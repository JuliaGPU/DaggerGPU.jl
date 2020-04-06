using .CuArrays
import .CuArrays: CUDAapi, CUDAdrv

struct CuArrayProc <: Dagger.Processor
    device
end

@gpuproc(CuArrayProc, CuArray)

processor(::Val{:CUDA}) = CuArrayProc
cancompute(::Val{:CUDA}) = CUDAapi.has_cuda()

push!(Dagger.PROCESSOR_CALLBACKS, proc -> begin
    if CUDAapi.has_cuda()
        return CuArrayProc(first(CUDAdrv.devices()))
    end
end)
