using .ROCArrays
import .ROCArrays: AMDGPUnative.HSARuntime

struct ROCArrayProc <: Dagger.Processor
    device
end

@gpuproc(ROCArrayProc, ROCArray)

processor(::Val{:ROC}) = ROCArrayProc
cancompute(::Val{:ROC}) = ROCArrays.configured

push!(Dagger.PROCESSOR_CALLBACKS, proc -> begin
    if ROCArrays.configured
        return ROCArrayProc(HSARuntime.get_default_agent())
    end
end)
