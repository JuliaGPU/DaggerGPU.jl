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
    tls = Dagger.get_tls()
    fetch(Threads.@spawn begin
        Dagger.set_tls!(tls)
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
            return CuArrayDeviceProc(dev.handle, myid())
        end)
    end
end
