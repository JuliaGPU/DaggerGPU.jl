using .CUDA
import .CUDA: CuDevice, CuContext, devices, attribute

export CuArrayDeviceProc

"Represents a single CUDA GPU device."
struct CuArrayDeviceProc <: Dagger.Processor
    owner::Int
    #ctx::CuContext
    device::Int
end
@gpuproc(CuArrayDeviceProc, CuArray)
#= FIXME: DtoD copies and CUDA IPC
function Dagger.move(from::CuArrayDeviceProc, to::CuArrayDeviceProc, x)
    if from === to
        return x
    else
        error("Not implemented")
    end
end
=#
function Dagger.execute!(proc::CuArrayDeviceProc, func, args...)
    fetch(Threads.@spawn begin
        task_local_storage(:processor, proc)
        CUDA.device!(proc.device)
        CUDA.@sync func(args...)
    end)
end
Base.show(io::IO, proc::CuArrayDeviceProc) =
    print(io, "CuArrayDeviceProc on worker $(proc.owner), device $(proc.device)")

processor(::Val{:CUDA}) = CuArrayDeviceProc
cancompute(::Val{:CUDA}) = CUDA.has_cuda()
kernel_backend(::CuArrayDeviceProc) = CUDADevice()

if CUDA.has_cuda()
    for dev in devices()
        Dagger.add_callback!(proc -> begin
            return CuArrayDeviceProc(Distributed.myid(), #=CuContext(dev),=# dev.handle)
        end)
    end
end
