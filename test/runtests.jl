using Distributed
using Test
addprocs(2)

@everywhere begin

using Distributed, Dagger, DaggerGPU
using CuArrays, ROCArrays

end

cuproc = DaggerGPU.processor(:CUDA)
rocproc = DaggerGPU.processor(:ROC)

if !DaggerGPU.cancompute(:CUDA)
    @warn "No CUDA devices available, falling back to ThreadProc"
    cuproc = Dagger.ThreadProc
end
if !DaggerGPU.cancompute(:ROC)
    @warn "No ROCm devices available, falling back to ThreadProc"
    rocproc = Dagger.ThreadProc
end

as = [delayed(x->x+1)(1) for i in 1:10]
b = delayed((xs...)->[sum(xs)])(as...)

opts = Dagger.Sch.ThunkOptions(;proctypes=[cuproc])
c1 = delayed(sum; options=opts)(b)
opts = Dagger.Sch.ThunkOptions(;proctypes=[rocproc])
c2 = delayed(sum; options=opts)(b)

opts = Dagger.Sch.ThunkOptions(;proctypes=[Dagger.ThreadProc])
d = delayed((x,y)->x+y; options=opts)(c1,c2)
@test collect(d) == 40
