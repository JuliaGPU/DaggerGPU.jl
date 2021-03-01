using .AMDGPU

struct RemoteROCDevice
    handle::UInt64
    owner::Int
end
RemoteROCDevice(handle::UInt64) = RemoteROCDevice(handle, myid())
# TODO: Instead take hash of identifying fields
RemoteROCDevice(agent::AMDGPU.HSAAgent, pid=myid()) =
    RemoteROCDevice(agent.agent.handle, pid)
function agent(rrd::RemoteROCDevice)
    rrd.owner == myid() || error("Cannot get agent of remote worker")
    AMDGPU.HSAAgent(AMDGPU.HSA.Agent(rrd.handle))
end

struct ROCArrayProc <: Dagger.Processor
    device::RemoteROCDevice
    owner::Int
end

@gpuproc(ROCArrayProc, ROCArray)
Dagger.get_parent(proc::ROCArrayProc) = Dagger.OSProc(proc.owner)

function Dagger.execute!(proc::ROCArrayProc, func, args...)
    tls = Dagger.get_tls()
    task = Threads.@spawn begin
        Dagger.set_tls!(tls)
        set_default_agent(agent(proc.device)) do
            func(args...)
        end
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
function Dagger.move(from_proc::ROCArrayProc, to_proc::ROCArrayProc, x::Chunk)
    if from_proc.owner == to_proc.owner
        x_from = poolget(x.handle)
        if from_proc.device.handle == to_proc.device.handle
            return x_from
        else
            to_agent = agent(to_proc.device)
            x_to = set_default_agent(to_agent) do
                similar(x_from)
            end
            copyto!(x_to, x_from)
            return x_to
        end
    else
        x_cpu = remotecall_fetch(from_proc.owner) do
            Dagger.move(from_proc, Dagger.OSProc(), x)
        end
        @assert !(x_cpu isa ROCArray)
        return Dagger.move(OSProc(), to_proc, x_cpu)
    end
end
Base.show(io::IO, proc::ROCArrayProc) =
    print(io, "ROCArrayProc on worker $(proc.owner), device $(proc.owner == myid() ? agent(proc.device) : proc.device)")


processor(::Val{:ROC}) = ROCArrayProc
cancompute(::Val{:ROC}) = AMDGPU.configured
# FIXME: kernel_backend(::ROCDevice) = ROCArrayProc

if AMDGPU.configured
    for agent in AMDGPU.get_agents(:gpu)
        Dagger.add_callback!(proc -> begin
            return ROCArrayProc(RemoteROCDevice(agent), myid())
        end)
    end
end
