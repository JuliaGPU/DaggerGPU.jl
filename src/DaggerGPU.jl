module DaggerGPU

using Dagger, MemPool
using Distributed
using KernelAbstractions, Adapt

import Dagger: Chunk

const CPUProc = Union{OSProc, Dagger.ThreadProc}

macro gpuproc(PROC, T)
    PROC = esc(PROC)
    T = esc(T)
    quote
        # Assume that we can run anything
        Dagger.iscompatible_func(proc::$PROC, opts, f) = true
        Dagger.iscompatible_arg(proc::$PROC, opts, x) = true

        # CPUs shouldn't process our array type
        Dagger.iscompatible_arg(proc::Dagger.ThreadProc, opts, x::$T) = false

        # Adapt to/from the appropriate type
        function Dagger.move(from_proc::CPUProc, to_proc::$PROC, x::Chunk)
            from_pid = Dagger.get_parent(from_proc).pid
            to_pid = Dagger.get_parent(to_proc).pid
            @assert myid() == to_pid
            cpu_data = remotecall_fetch(from_pid, x) do x
                poolget(x.handle)
            end
            return DaggerGPU.with_device(to_proc) do
                adapt($T, cpu_data)
            end
        end
        function Dagger.move(from_proc::$PROC, to_proc::CPUProc, x::Chunk)
            from_pid = Dagger.get_parent(from_proc).pid
            to_pid = Dagger.get_parent(to_proc).pid
            @assert myid() == to_pid
            remotecall_fetch(from_pid, x) do x
                # FIXME: Need to switch devices
                DaggerGPU.with_device(from_proc) do
                    adapt(Array, poolget(x.handle))
                end
            end
        end
        function Dagger.move(from_proc::CPUProc, to_proc::$PROC, x)
            DaggerGPU.with_device(to_proc) do
                x_opt = DaggerGPU.move_optimized(from_proc, to_proc, x)
                if x_opt !== nothing
                    return x_opt
                end
                return adapt($T, x)
            end
        end
        function Dagger.move(from_proc::$PROC, to_proc::CPUProc, x)
            DaggerGPU.with_device(from_proc) do
                x_opt = DaggerGPU.move_optimized(from_proc, to_proc, x)
                if x_opt !== nothing
                    return x_opt
                end
                return adapt(Array, x)
            end
        end
    end
end

processor(kind::Symbol) = processor(Val(kind))
processor(::Val) = Dagger.ThreadProc
cancompute(kind::Symbol) = cancompute(Val(kind))
cancompute(::Val) = false
function with_device end

move_optimized(from_proc::Dagger.Processor,
               to_proc::Dagger.Processor,
               x) = nothing

kernel_backend() = kernel_backend(Dagger.Sch.thunk_processor())
kernel_backend(::Dagger.ThreadProc) = CPU()

using Requires
@static if !isdefined(Base, :get_extension)
function __init__()
    @require CUDA="052768ef-5323-5732-b1bb-66c8b64840ba" begin
        include(joinpath(@__DIR__, "..", "ext", "CUDAExt.jl"))
    end
    @require AMDGPU="21141c5a-9bdb-4563-92ae-f87d6854732e" begin
        include(joinpath(@__DIR__, "..", "ext", "ROCExt.jl"))
    end
    @require Metal="dde4c033-4e86-420c-a63e-0dd931031962" begin
        include(joinpath(@__DIR__, "..", "ext", "MetalExt.jl"))
    end
end
end

end
