using .Metal
import .Metal: MtlArray, MtlDevice

struct MtlArrayDeviceProc <: Dagger.Processor
    owner::Int
    device_id::UInt64
end

# Assume that we can run anything.
Dagger.iscompatible_func(proc::MtlArrayDeviceProc, opts, f) = true
Dagger.iscompatible_arg(proc::MtlArrayDeviceProc, opts, x) = true

# CPUs shouldn't process our array type.
Dagger.iscompatible_arg(proc::Dagger.ThreadProc, opts, x::MtlArray) = false

function Dagger.move(from_proc::OSProc, to_proc::MtlArrayDeviceProc, x::Chunk)
    from_pid = from_proc.pid
    to_pid = Dagger.get_parent(to_proc).pid
    @assert myid() == to_pid

    return Dagger.move(from_proc, to_proc, remotecall_fetch(x->poolget(x.handle), from_pid, x))
end

function Dagger.move(from_proc::MtlArrayDeviceProc, to_proc::OSProc, x::Chunk)
    from_pid = Dagger.get_parent(from_proc).pid
    to_pid = to_proc.pid
    @assert myid() == to_pid

    return remotecall_fetch(from_pid, x) do x
        mtlarray = poolget(x.handle)
        return Dagger.move(from_proc, to_proc, mtlarray)
    end
end

function Dagger.move(
    from_proc::OSProc,
    to_proc::MtlArrayDeviceProc,
    x::Array{T, N}
) where {T, N}
    # If we have unified memory, we can try casting the `Array` to `MtlArray`.
    device = _get_metal_device(to_proc)

    if (device !== nothing) && device.hasUnifiedMemory
        marray = _cast_array_to_mtlarray(x, device)
        marray !== nothing && return marray
    end

    return adapt(MtlArray, x)
end

function Dagger.move(from_proc::OSProc, to_proc::MtlArrayDeviceProc, x)
    adapt(MtlArray, x)
end

function Dagger.move(
    from_proc::MtlArrayDeviceProc,
    to_proc::OSProc,
    x::Array{T, N}
) where {T, N}
    # If we have unified memory, we can just cast the `MtlArray` to an `Array`.
    device = _get_metal_device(from_proc)

    if (device !== nothing) && device.hasUnifiedMemory
        return unsafe_wrap(Array{T}, x, size(x))
    else
        return adapt(Array, x)
    end
end

function Dagger.move(from_proc::MtlArrayDeviceProc, to_proc::OSProc, x)
    adapt(Array, x)
end

Dagger.get_parent(proc::MtlArrayDeviceProc) = Dagger.OSProc(proc.owner)

function Dagger.execute!(proc::MtlArrayDeviceProc, func, args...)
    tls = Dagger.get_tls()
    task = Threads.@spawn begin
        Dagger.set_tls!(tls)
        Metal.@sync func(args...)
    end

    try
        fetch(task)
    catch err
        @static if VERSION >= v"1.1"
            stk = Base.catch_stack(task)
            err, frames = stk[1]
            rethrow(CapturedException(err, frames))
        else
            rethrow(task.result)
        end
    end
end

function Base.show(io::IO, proc::MtlArrayDeviceProc)
    print(io, "MtlArrayDeviceProc on worker $(proc.owner), device ($(something(_get_metal_device(proc)).name))")
end

processor(::Val{:Metal}) = MtlArrayDeviceProc
cancompute(::Val{:Metal}) = length(Metal.devices()) >= 1
kernel_backend(::MtlArrayDeviceProc) = MtlArrayDeviceProc

for dev in Metal.devices()
    Dagger.add_processor_callback!("metal_device_$(dev.registryID)") do
        MtlArrayDeviceProc(Distributed.myid(), dev.registryID)
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
