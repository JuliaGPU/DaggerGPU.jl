using .CuArrays
import .CuArrays: CUDAapi, CUDAdrv

struct CuArrayProc <: Dagger.Processor
    device
end

@gpuproc(CuArrayProc, CuArray)


push!(Dagger.PROCESSOR_CALLBACKS, proc -> begin
    if CUDAapi.has_cuda()
        @eval processor(::Val{:CUDA}) = CuArrayProc
        return CuArrayProc(first(devices()))
    end
end)
