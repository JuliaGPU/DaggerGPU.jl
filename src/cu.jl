using .CUDA
import .CUDA: CuDevice, CuContext, devices, attribute

export CuArrayProc, CuArrayDeviceProc, CuArraySMProc

"Represents a single CUDA GPU device."
struct CuArrayDeviceProc <: Dagger.Processor
    owner::Int
    #ctx::CuContext
    device::CuDevice
end
@gpuproc(CuArrayDeviceProc, CuArray)
const CuArrayProc = CuArrayDeviceProc
#= FIXME: CUDA IPC
function Dagger.move(from::CuArrayDeviceProc, to::CuArrayDeviceProc, x)
    if from === to
        return x
    else
        error("Not implemented")
    end
end
=#
function Dagger.execute!(proc::CuArrayDeviceProc, func, args...)
    #CUDA.context!(proc.ctx)
    CUDA.@sync func(args...)
end

"Represents a single CUDA GPU Streaming Multiprocessor."
struct CuArraySMProc <: Dagger.Processor
    owner::Int
    #ctx::CuContext
    device::CuDevice
    sm::Int
end
@gpuproc(CuArraySMProc, CuArray)
#= FIXME: CUDA IPC
function Dagger.move(from::CuArraySMProc, to::CuArraySMProc, x)
    if from.device === to.device
        return x
    else
        error("Not implemented")
    end
end
=#
function Dagger.execute!(proc::CuArraySMProc, func, args...)
    #CUDA.context!(proc.ctx)
    CUDA.@sync func(args...)
end

processor(::Val{:CUDA}) = CuArrayDeviceProc
cancompute(::Val{:CUDA}) = CUDA.has_cuda()
# TODO: CuArraySMProc

if CUDA.has_cuda()
    for dev in devices()
        Dagger.add_callback!(proc -> begin
            return CuArrayDeviceProc(Distributed.myid(), #=CuContext(dev),=# dev)
        end)
        for i in 1:attribute(dev, CUDA.CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
            Dagger.add_callback!(proc -> begin
                return CuArraySMProc(Distributed.myid(), #=CuContext(dev),=# dev, i)
            end)
        end
    end
end
