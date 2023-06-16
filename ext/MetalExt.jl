module MetalExt

export MtlArrayDeviceProc

import Dagger, DaggerGPU
import Distributed: myid

const CPUProc = Union{Dagger.OSProc,Dagger.ThreadProc}

if isdefined(Base, :get_extension)
    import Metal
else
    import ..Metal
end
import Metal: MtlArray, MetalBackend
const MtlDevice = Metal.MTL.MTLDeviceInstance

struct MtlArrayDeviceProc <: Dagger.Processor
    owner::Int
    device_id::UInt64
end

DaggerGPU.@gpuproc(MtlArrayDeviceProc, MtlArray)
Dagger.get_parent(proc::MtlArrayDeviceProc) = Dagger.OSProc(proc.owner)

function DaggerGPU.move_optimized(
    from_proc::CPUProc,
    to_proc::MtlArrayDeviceProc,
    x::Array
)
    # FIXME
    return nothing

    # If we have unified memory, we can try casting the `Array` to `MtlArray`.
    device = _get_metal_device(to_proc)

    if (device !== nothing) && device.hasUnifiedMemory
        marray = _cast_array_to_mtlarray(x, device)
        marray !== nothing && return marray
    end

    return nothing
end


function DaggerGPU.move_optimized(
    from_proc::MtlArrayDeviceProc,
    to_proc::CPUProc,
    x::Array
)
    # FIXME
    return nothing

    # If we have unified memory, we can just cast the `MtlArray` to an `Array`.
    device = _get_metal_device(from_proc)

    if (device !== nothing) && device.hasUnifiedMemory
        return unsafe_wrap(Array{T}, x, size(x))
    end

    return nothing
end

function Dagger.execute!(proc::MtlArrayDeviceProc, f, args...; kwargs...)
    @nospecialize f args kwargs
    tls = Dagger.get_tls()
    task = Threads.@spawn begin
        Dagger.set_tls!(tls)
        result = Base.@invokelatest f(args...; kwargs...)
        Metal.synchronize()
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

function Base.show(io::IO, proc::MtlArrayDeviceProc)
    print(io, "MtlArrayDeviceProc(worker $(proc.owner), device $(something(_get_metal_device(proc)).name))")
end

DaggerGPU.processor(::Val{:Metal}) = MtlArrayDeviceProc
DaggerGPU.cancompute(::Val{:Metal}) = Metal.functional()
DaggerGPU.kernel_backend(proc::MtlArrayDeviceProc) = MetalBackend()
# TODO: Switch devices
DaggerGPU.with_device(f, proc::MtlArrayDeviceProc) = f()

function Dagger.to_scope(::Val{:metal_gpu}, sc::NamedTuple)
    worker = get(sc, :worker, 1)
    dev_id = sc.metal_gpu
    dev = Metal.devices()[dev_id]
    return Dagger.ExactScope(MtlArrayDeviceProc(worker, dev.registryID))
end
Dagger.scope_key_precedence(::Val{:metal_gpu}) = 1

function __init__()
    for dev in Metal.devices()
        @debug "Registering Metal GPU processor with Dagger: $dev"
        Dagger.add_processor_callback!("metal_device_$(dev.registryID)") do
            MtlArrayDeviceProc(myid(), dev.registryID)
        end
    end
end


################################################################################
#                              Private functions
################################################################################

# Try casting the array `x` to an `MtlArray`. If the casting is not possible,
# return `nothing`.
function _cast_array_to_mtlarray(x::Array{T,N}, device::MtlDevice) where {T,N}
    # Try creating the buffer without copying.
    dims = size(x)
    nbytes_array = prod(dims) * sizeof(T)
    pagesize = ccall(:getpagesize, Cint, ())
    num_pages = div(nbytes_array, pagesize, RoundUp)
    nbytes = num_pages * pagesize

    pbuf = Metal.MTL.mtDeviceNewBufferWithBytesNoCopy(
        device,
        pointer(x),
        nbytes,
        Metal.Shared | Metal.MTL.DefaultTracking | Metal.MTL.DefaultCPUCache
    )

    if pbuf != C_NULL
        buf = MtlBuffer(pbuf)
        marray = MtlArray{T,N}(buf, dims)
        return marray
    end

    # If we reached here, the conversion was not possible.
    return nothing
end

# Return the Metal device handler given the ID recorded in `proc`.
function _get_metal_device(proc::MtlArrayDeviceProc)
    devices = Metal.devices()
    id = findfirst(dev -> dev.registryID == proc.device_id, devices)

    if devices === nothing
        return nothing
    else
        return devices[id]
    end
end

end # module MetalExt
