module CUDAExt

export CuArrayDeviceProc

import Dagger, DaggerGPU, MemPool
import Distributed: myid, remotecall_fetch

const CPUProc = Union{Dagger.OSProc,Dagger.ThreadProc}

if isdefined(Base, :get_extension)
    import CUDA
else
    import ..CUDA
end
import CUDA: CuDevice, CuContext, CuArray, CUDABackend, devices, attribute

using UUIDs

"Represents a single CUDA GPU device."
struct CuArrayDeviceProc <: Dagger.Processor
    owner::Int
    device::Int
    device_uuid::UUID
end
DaggerGPU.@gpuproc(CuArrayDeviceProc, CuArray)
Dagger.get_parent(proc::CuArrayDeviceProc) = Dagger.OSProc(proc.owner)

# function can_access(this, peer)
#     status = Ref{Cint}()
#     CUDA.cuDeviceCanAccessPeer(status, this, peer)
#     return status[] == 1
# end

function Dagger.move(from::CuArrayDeviceProc, to::CuArrayDeviceProc, x::Dagger.Chunk{T}) where T<:CuArray
    if from == to
        # Same process and GPU, no change
        MemPool.poolget(x.handle)
    elseif from.owner == to.owner
        # Same process but different GPUs, use DtoD copy
        from_arr = MemPool.poolget(x.handle)
        to_arr = CUDA.device!(to.device) do
            CuArray{T,N}(undef, size)
        end
        copyto!(to_arr, from_arr)
        to_arr
    elseif Dagger.system_uuid(from.owner) == Dagger.system_uuid(to.owner)
        # Same node, we can use IPC
        ipc_handle, eT, shape = remotecall_fetch(from.owner, x.handle) do h
            arr = MemPool.poolget(h)
            ipc_handle_ref = Ref{CUDA.CUipcMemHandle}()
            GC.@preserve arr begin
                CUDA.cuIpcGetMemHandle(ipc_handle_ref, pointer(arr))
            end
            (ipc_handle_ref[], eltype(arr), size(arr))
        end
        r_ptr = Ref{CUDA.CUdeviceptr}()
        CUDA.device!(from.device) do # FIXME: Assumes that device IDs are identical across processes
            CUDA.cuIpcOpenMemHandle(r_ptr, ipc_handle, CUDA.CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS)
        end
        ptr = Base.unsafe_convert(CUDA.CuPtr{eT}, r_ptr[])
        arr = unsafe_wrap(CuArray, ptr, shape; own=false)
        finalizer(arr) do arr
            CUDA.cuIpcCloseMemHandle(pointer(arr))
        end
        if from.device_uuid != to.device_uuid
            CUDA.device!(to.device) do
                to_arr = similar(arr)
                copyto!(to_arr, arr)
                to_arr
            end
        else
            arr
        end
    else
        # Different node, use DtoH, serialization, HtoD
        # TODO UCX
        CuArray(remotecall_fetch(from.owner, x.handle) do h
            Array(MemPool.poolget(h))
        end)
    end
end

function Dagger.move(from_proc::CPUProc, to_proc::CuArrayDeviceProc, x::CuArray)
    # TODO: No extra allocations here
    if CUDA.device(x) == collect(CUDA.devices())[to_proc.device+1]
        return x
    end
    DaggerGPU.with_device(to_proc) do
        _x = similar(x)
        copyto!(_x, x)
        return _x
    end
end

function Dagger.move(from_proc::CuArrayDeviceProc, to_proc::CPUProc, x::CuArray{T,N}) where {T,N}
    _x = Array{T,N}(undef, size(x))
    copyto!(_x, x)
    return _x
end

function Dagger.execute!(proc::CuArrayDeviceProc, f, args...; kwargs...)
    @nospecialize f args kwargs
    tls = Dagger.get_tls()
    task = Threads.@spawn begin
        Dagger.set_tls!(tls)
        CUDA.device!(proc.device)
        result = Base.@invokelatest f(args...; kwargs...)
        CUDA.synchronize()
        return result
    end

    try
        fetch(task)
    catch err
        stk = current_exceptions(task)
        err, frames = stk[1]
        rethrow(CapturedException(err, frames))
    end
end
Base.show(io::IO, proc::CuArrayDeviceProc) =
    print(io, "CuArrayDeviceProc(worker $(proc.owner), device $(proc.device), uuid $(proc.device_uuid))")

DaggerGPU.processor(::Val{:CUDA}) = CuArrayDeviceProc
DaggerGPU.cancompute(::Val{:CUDA}) = CUDA.has_cuda()
DaggerGPU.kernel_backend(::CuArrayDeviceProc) = CUDABackend()
DaggerGPU.with_device(f, proc::CuArrayDeviceProc) =
    CUDA.device!(f, proc.device)

function Dagger.to_scope(::Val{:cuda_gpu}, sc::NamedTuple)
    worker = get(sc, :worker, 1)
    dev_id = sc.cuda_gpu
    dev = collect(CUDA.devices())[dev_id]
    return Dagger.ExactScope(CuArrayDeviceProc(worker, dev_id-1, CUDA.uuid(dev)))
end
Dagger.scope_key_precedence(::Val{:cuda_gpu}) = 1

function __init__()
    if CUDA.has_cuda()
        for dev in CUDA.devices()
            @debug "Registering CUDA GPU processor with Dagger: $dev"
            Dagger.add_processor_callback!("cuarray_device_$(dev.handle)") do
                CuArrayDeviceProc(myid(), dev.handle, CUDA.uuid(dev))
            end
        end
    end
end

end # module CUDAExt
