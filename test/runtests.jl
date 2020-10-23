using Distributed
using Test
addprocs(2)

@everywhere begin

using Distributed, Dagger, DaggerGPU
using CUDA, AMDGPU

function myfunc(X)
    @assert !(X isa Array)
    X
end

end

function generate_thunks()
    as = [delayed(x->x+1)(1) for i in 1:10]
    delayed((xs...)->[sum(xs)])(as...)
end

@test DaggerGPU.cancompute(:CUDA) || DaggerGPU.cancompute(:ROC)

@testset "CUDA" begin
    if !DaggerGPU.cancompute(:CUDA)
        @warn "No CUDA devices available, skipping tests"
    else
        didtest = true
        cuproc = DaggerGPU.processor(:CUDA)
        b = generate_thunks()
        opts = Dagger.Sch.ThunkOptions(;proctypes=[cuproc])
        c_pre = delayed(myfunc; options=opts)(b)
        c = delayed(sum; options=opts)(b)

        opts = Dagger.Sch.ThunkOptions(;proctypes=[Dagger.ThreadProc])
        d = delayed(identity; options=opts)(c)
        @test collect(d) == 20
    end
end

@testset "ROCm" begin
    if !DaggerGPU.cancompute(:ROC)
        @warn "No ROCm devices available, skipping tests"
    else
        didtest = true
        rocproc = DaggerGPU.processor(:ROC)
        b = generate_thunks()
        opts = Dagger.Sch.ThunkOptions(;proctypes=[rocproc])
        c_pre = delayed(myfunc; options=opts)(b)
        c = delayed(sum; options=opts)(b)

        opts = Dagger.Sch.ThunkOptions(;proctypes=[Dagger.ThreadProc])
        d = delayed(identity; options=opts)(c)
        @test collect(d) == 20
    end
end
