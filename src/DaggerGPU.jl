module DaggerGPU

using Dagger, Requires, Adapt
import Dagger: Chunk
import Dagger.MemPool: DRef, poolget
using Distributed
using KernelAbstractions

macro gpuproc(PROC, T)
    quote
        # Assume that we can run anything
        Dagger.iscompatible_func(proc::$PROC, opts, f) = true
        Dagger.iscompatible_arg(proc::$PROC, opts, x) = true

        # CPUs shouldn't process our array type
        Dagger.iscompatible_arg(proc::Dagger.ThreadProc, opts, x::$T) = false

        # Adapt to/from the appropriate type
        Dagger.move(from_proc::OSProc, to_proc::$PROC, x::Chunk) = Dagger.move(from_proc, to_proc, x.handle)
        Dagger.move(from_proc::$PROC, to_proc::OSProc, x::Chunk) = Dagger.move(from_proc, to_proc, x.handle)
        Dagger.move(from_proc::OSProc, to_proc::$PROC, x::DRef) = Dagger.move(from_proc, to_proc, poolget(x))
        Dagger.move(from_proc::$PROC, to_proc::OSProc, x::DRef) = Dagger.move(from_proc, to_proc, poolget(x))
        Dagger.move(from_proc::OSProc, to_proc::$PROC, x) = adapt($T, x)
        Dagger.move(from_proc::$PROC, to_proc::OSProc, x) = adapt(Array, x)
    end
end

processor(kind::Symbol) = processor(Val(kind))
processor(::Val) = Dagger.ThreadProc
cancompute(kind::Symbol) = cancompute(Val(kind))
cancompute(::Val) = false

kernel_backend() = kernel_backend(Dagger.Sch.thunk_processor())
kernel_backend(::Dagger.ThreadProc) = CPU()

function __init__()
    @require CUDA="052768ef-5323-5732-b1bb-66c8b64840ba" begin
        include("cu.jl")
    end
    @require ROCArrays="ddf941ca-5d6a-11e9-36cc-a3fed13dd2fc" begin
        include("roc.jl")
    end
end

end
