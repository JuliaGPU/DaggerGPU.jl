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

function validate_arr_loc(agent::HSAAgent, arr::ROCArray)
    @assert Array(arr)[1] !== nothing # access test
    if arr.buf.agent != agent
        agent_str = sprint(io->Base.show(io, agent))
        arr_agent_str = sprint(io->Base.show(io, arr.buf.agent))
        Core.println("agent: $agent_str, arr_agent: $arr_agent_str")
    end
    @assert arr.buf.agent == agent
end
function Dagger.execute!(proc::ROCArrayProc, func, args...)
    @info "Executing on $proc: $func $(typeof.(args))"
    mydev = CUDA.CuDevice(proc.device)
    for arg in filter(x->x isa ROCArray, args)
        validate_arr_loc(mydev, arg)
    end
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
function Dagger.move(from_proc::OSProc, to_proc::ROCArrayProc, x::ROCArray)
    @assert from_proc.pid == to_proc.owner # FIXME
    x_cpu = adapt(Array, x)
    @assert x_cpu isa Array
    x_gpu = set_default_agent(agent(to_proc.device)) do
        adapt(ROCArray, x_cpu)
    end
    @assert x_gpu isa ROCArray
    validate_arr_loc(agent(to_proc.device), x_gpu)
    return x_gpu
end
function Dagger.move(from_proc::OSProc, to_proc::ROCArrayProc, x::Chunk)
    @assert from_proc.pid == to_proc.owner # FIXME
    x_cpu = remotecall_fetch(from_proc.pid) do
        adapt(Array, poolget(x.handle))
    end
    x_gpu = set_default_agent(agent(to_proc.device)) do
        adapt(ROCArray, x_cpu)
    end
    if x_gpu isa ROCArray
        validate_arr_loc(agent(to_proc.device), x_gpu)
    end
    return x_gpu
end
function Dagger.move(from_proc::ROCArrayProc, to_proc::ROCArrayProc, x::ROCArray)
    x_cpu = remotecall_fetch(from_proc.owner) do
        adapt(Array, x)
    end
    x_gpu = set_default_agent(agent(to_proc.device)) do
        adapt(ROCArray, x_cpu)
    end
    @assert x_gpu isa ROCArray
    validate_arr_loc(agent(to_proc.device), x_gpu)
    return x_gpu
end
function Dagger.move(from_proc::ROCArrayProc, to_proc::ROCArrayProc, x::Chunk)
    #=
    if from_proc.owner == to_proc.owner
        x_from = poolget(x.handle)
        if from_proc.device == to_proc.device
            return x_from
        else
            x_to = set_default_agent(to_proc.device) do
                similar(x_from)
            end
            copyto!(x_to, x_from)
            return x_to
        end
    else
    =#
        x_cpu = remotecall_fetch(from_proc.owner) do
            adapt(Array, poolget(x.handle))
        end
        x_gpu = set_default_agent(agent(to_proc.device)) do
            adapt(ROCArray, x_cpu)
        end
        if x_gpu isa ROCArray
            validate_arr_loc(agent(to_proc.device), x_gpu)
        end
        return x_gpu
    #end
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
