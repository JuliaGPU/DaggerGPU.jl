using Distributed
using Test
addprocs(2)

@everywhere begin

using Distributed, Dagger, DaggerGPU
using CuArrays, ROCArrays

end

cuproc = DaggerGPU.processor(:CUDA)
rocproc = DaggerGPU.processor(:ROC)

cuproc === Dagger.ThreadProc && @warn "No CUDA devices available"
rocproc === Dagger.ThreadProc && @warn "No ROCm devices available"

as = [delayed(x->x+1)(1) for i in 1:10]
b = delayed((xs...)->[sum(xs)])(as...)

opts = Dagger.Sch.ThunkOptions(;proctypes=[cuproc])
c1 = delayed(sum; options=opts)(b)
opts = Dagger.Sch.ThunkOptions(;proctypes=[rocproc])
c2 = delayed(sum; options=opts)(b)

opts = Dagger.Sch.ThunkOptions(;proctypes=[Dagger.ThreadProc])
d = delayed((x,y)->x+y; options=opts)(c1,c2)
@test collect(d) == 40
