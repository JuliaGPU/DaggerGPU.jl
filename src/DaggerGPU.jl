module DaggerGPU

using Dagger, Requires

macro gpuproc(PROC, T)
    quote
Dagger.iscompatible(proc::$PROC, opts, x::AbstractArray{AT}) where AT =
    isbitstype(AT)
Dagger.move(ctx, from_proc::OSProc, to_proc::$PROC, x::AbstractArray) =
    $T(x)
Dagger.move(ctx, from_proc::$PROC, to_proc::OSProc, x) = x
Dagger.move(ctx, from_proc::$PROC, to_proc::OSProc, x::$T) =
    collect(x)
Dagger.execute!(proc::$PROC, func, args...) = func(args...)
    end
end

processor(kind::Symbol) = processor(Val(kind))
processor(::Val) = Dagger.ThreadProc

function __init__()
    @require CuArrays="3a865a2d-5b23-5a0f-bc46-62713ec82fae" begin
        include("cuarrays.jl")
    end
    @require ROCArrays="ddf941ca-5d6a-11e9-36cc-a3fed13dd2fc" begin
        include("rocarrays.jl")
    end
end

end
