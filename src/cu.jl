using .CUDA
import .CUDA: CuDevice, CuContext, devices, attribute

export CuArrayDeviceProc

"Represents a single CUDA GPU device."
struct CuArrayDeviceProc <: Dagger.Processor
    device::Int
    owner::Int
end

@gpuproc(CuArrayDeviceProc, CuArray)
Dagger.get_parent(proc::CuArrayDeviceProc) = Dagger.OSProc(proc.owner)

function validate_arr_loc(dev::CUDA.CuDevice, arr::CuArray)
    @assert Array(arr)[1] !== nothing # access test
    devs = collect(CUDA.devices())
    ptr = arr.baseptr
    alloc_devs = filter(d->haskey(CUDA.allocated[d],ptr), devs)
    if !(dev in alloc_devs)
        dev_str = sprint(io->Base.show(io, dev))
        alloc_devs_str = sprint(io->Base.show(io, alloc_devs))
        Core.println("dev: $dev_str, alloc_devs: $alloc_devs_str")
    end
    @assert dev in alloc_devs
end
function Dagger.execute!(proc::CuArrayDeviceProc, func, args...)
    @info "Executing on $proc: $func $(typeof.(args))"
    mydev = CUDA.CuDevice(proc.device)
    for arg in filter(x->x isa CuArray, args)
        validate_arr_loc(mydev, arg)
    end
    tls = Dagger.get_tls()
    task = Threads.@spawn begin
        Dagger.set_tls!(tls)
        CUDA.device!(proc.device)
        CUDA.@sync func(args...)
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
# FIXME: DtoD copies and CUDA IPC
function Dagger.move(from_proc::OSProc, to_proc::CuArrayDeviceProc, x::CuArray)
    #Core.println("1")
    @assert from_proc.pid == to_proc.owner # FIXME
    x_cpu = adapt(Array, x)
    @assert x_cpu isa Array
    x_gpu = CUDA.device!(to_proc.device) do
        adapt(CuArray, x_cpu)
    end
    @assert x_gpu isa CuArray
    validate_arr_loc(CUDA.CuDevice(to_proc.device), x_gpu)
    return x_gpu
end
function Dagger.move(from_proc::OSProc, to_proc::CuArrayDeviceProc, x::Chunk)
    @assert from_proc.pid == to_proc.owner # FIXME
    x_cpu = remotecall_fetch(from_proc.pid) do
        adapt(Array, poolget(x.handle))
    end
    x_gpu = CUDA.device!(to_proc.device) do
        adapt(CuArray, x_cpu)
    end
    if x_gpu isa CuArray
        validate_arr_loc(CUDA.CuDevice(to_proc.device), x_gpu)
    end
    return x_gpu
end
function Dagger.move(from_proc::CuArrayDeviceProc, to_proc::CuArrayDeviceProc, x::CuArray)
    x_cpu = remotecall_fetch(from_proc.owner) do
        adapt(Array, x)
    end
    x_gpu = CUDA.device!(to_proc.device) do
        adapt(CuArray, x_cpu)
    end
    @assert x_gpu isa CuArray
    validate_arr_loc(CUDA.CuDevice(to_proc.device), x_gpu)
    return x_gpu
end
function Dagger.move(from_proc::CuArrayDeviceProc, to_proc::CuArrayDeviceProc, x::Chunk)
    #=
    if from_proc.owner == to_proc.owner
        x_from = poolget(x.handle)
        if from_proc.device == to_proc.device
            return x_from
        else
            x_to = CUDA.device!(to_proc.device) do
                similar(x_from)
            end
            copyto!(x_to, x_from)
            return x_to
        end
    else
    =#
        x_cpu = remotecall_fetch(from_proc.owner) do
            adapt(Array, poolget(x.handle))
        end
        x_gpu = CUDA.device!(to_proc.device) do
            adapt(CuArray, x_cpu)
        end
        if x_gpu isa CuArray
            validate_arr_loc(CUDA.CuDevice(to_proc.device), x_gpu)
        end
        return x_gpu
    #end
end
Base.show(io::IO, proc::CuArrayDeviceProc) =
    print(io, "CuArrayDeviceProc on worker $(proc.owner), device $(proc.device)")

processor(::Val{:CUDA}) = CuArrayDeviceProc
cancompute(::Val{:CUDA}) = CUDA.has_cuda()
kernel_backend(::CuArrayDeviceProc) = CUDADevice()

if CUDA.has_cuda()
    for dev in devices()
        Dagger.add_callback!(proc -> begin
            return CuArrayDeviceProc(dev.handle, myid())
        end)
    end
end
