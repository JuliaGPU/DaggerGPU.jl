using .AMDGPU

struct ROCArrayProc <: Dagger.Processor
    device
end

@gpuproc(ROCArrayProc, ROCArray)

Dagger.execute!(proc::ROCArrayProc, func, args...) = func(args...)

processor(::Val{:ROC}) = ROCArrayProc
cancompute(::Val{:ROC}) = AMDGPU.configured
# FIXME: kernel_backend(::ROCDevice) = ROCArrayProc

if AMDGPU.configured
    Dagger.add_callback!(proc -> begin
        return ROCArrayProc(AMDGPU.get_default_agent())
    end)
end
