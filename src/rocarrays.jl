using .ROCArrays
import .ROCArrays: AMDGPUnative.HSARuntime

struct ROCArrayProc <: Dagger.Processor
    device
end

@gpuproc(ROCArrayProc, ROCArray)


push!(Dagger.PROCESSOR_CALLBACKS, proc -> begin
    if ROCArrays.configured
        @eval processor(::Val{:ROC}) = ROCArrayProc
        return ROCArrayProc(HSARuntime.get_default_agent())
    end
end)
