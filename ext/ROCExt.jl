module ROCExt

export ROCArrayDeviceProc

import Dagger, DaggerGPU
import Distributed: myid

const CPUProc = Union{Dagger.OSProc,Dagger.ThreadProc}

if isdefined(Base, :get_extension)
    import AMDGPU
else
    import ..AMDGPU
end
import AMDGPU: ROCDevice, ROCArray, ROCBackend

struct ROCArrayDeviceProc <: Dagger.Processor
    owner::Int
    device_id::Int
end
DaggerGPU.@gpuproc(ROCArrayDeviceProc, ROCArray)
Dagger.get_parent(proc::ROCArrayDeviceProc) = Dagger.OSProc(proc.owner)

function Dagger.execute!(proc::ROCArrayDeviceProc, f, args...; kwargs...)
    @nospecialize f args kwargs
    tls = Dagger.get_tls()
    task = Threads.@spawn begin
        Dagger.set_tls!(tls)
        AMDGPU.device!(AMDGPU.devices()[proc.device_id])
        result = Base.@invokelatest f(args...; kwargs...)
        AMDGPU.synchronize()
        return result
    end

    try
        fetch(task)
    catch err
        stk = current_exceptions(task)
        err, frames = stk[1]
        rethrow(CapturedException(err, frames))
    end
end
Base.show(io::IO, proc::ROCArrayDeviceProc) =
    print(io, "ROCArrayDeviceProc(worker $(proc.owner), device $(AMDGPU.devices()[proc.device_id]))")

DaggerGPU.processor(::Val{:ROC}) = ROCArrayDeviceProc
DaggerGPU.cancompute(::Val{:ROC}) = AMDGPU.functional()
DaggerGPU.kernel_backend(proc::ROCArrayDeviceProc) = ROCBackend()
DaggerGPU.with_device(f, proc::ROCArrayDeviceProc) =
    AMDGPU.device!(f, AMDGPU.devices()[proc.device_id])

function Dagger.to_scope(::Val{:rocm_gpu}, sc::NamedTuple)
    worker = get(sc, :worker, 1)
    @assert 1 <= sc.rocm_gpu <= length(AMDGPU.devices())
    return Dagger.ExactScope(ROCArrayDeviceProc(worker, sc.rocm_gpu))
end
Dagger.scope_key_precedence(::Val{:rocm_gpu}) = 1

function __init__()
    if AMDGPU.functional()
        for device_id in 1:length(AMDGPU.devices())
            @debug "Registering ROCm GPU processor with Dagger: $(AMDGPU.devices(device_id))"
            Dagger.add_processor_callback!("rocarray_device_$device_id") do
                ROCArrayDeviceProc(myid(), device_id)
            end
        end
    end
end

end # module ROCExt
