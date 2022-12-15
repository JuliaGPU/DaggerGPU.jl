using .Metal
import .Metal: MtlArray, MtlDevice

struct MtlArrayDeviceProc <: Dagger.Processor
    owner::Int
    device::MtlDevice
end

@gpuproc(MtlArrayDeviceProc, MtlArray)
Dagger.get_parent(proc::MtlArrayDeviceProc) = Dagger.OSProc(proc.owner)

function Dagger.execute!(proc::MtlArrayDeviceProc, func, args...)
    tls = Dagger.get_tls()
    task = Threads.@spawn begin
        Dagger.set_tls!(tls)
        Metal.@sync func(args...)
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

function Base.show(io::IO, proc::MtlArrayDeviceProc)
    print(io, "MtlArrayDeviceProc on worker $(proc.owner), device ($(proc.device.name))")
end

processor(::Val{:Metal}) = MtlArrayDeviceProc
cancompute(::Val{:Metal}) = length(Metal.devices()) >= 1
kernel_backend(::MtlArrayDeviceProc) = Metal.current_device()

for dev in Metal.devices()
    Dagger.add_processor_callback!("metal_device_$(dev.registryID)") do
        MtlArrayDeviceProc(Distributed.myid(), Metal.current_device())
    end
end
