# DaggerGPU

**GPU integrations for Dagger.jl**

DaggerGPU.jl makes use of the `Dagger.Processor` infrastructure to dispatch Dagger kernels to NVIDIA, AMD, and Apple GPUs, via CUDA.jl, AMDGPU.jl, and Metal.jl respectively. Usage is simple: `add` or `dev` DaggerGPU.jl and CUDA.jl/AMDGPU.jl/Metal.jl appropriately, load it with `using DaggerGPU`, and add the appropriate GPU scope to the Dagger scope options, for example

``` julia
using CUDA, Dagger, DaggerGPU
sc = Dagger.scope(cuda_gpu=1)

# two large matrices
A = rand(1000, 1000); B = rand(1000, 1000)
# move them to gpu and multiply there
A_gpu = Dagger.@spawn scope=sc CUDA.CuMatrix(A); B_gpu = Dagger.@spawn scope=sc CUDA.CuMatrix(B)
C_gpu = Dagger.@spawn scope=sc A_gpu*B_gpu
# move back to cpu to use there.
C = Dagger.@spawn scope=sc Matrix(C_gpu) 
```

and similarly for `rocm_gpu` and `metal_gpu`.

DaggerGPU.jl is still experimental, but we welcome GPU-owning users to try it out and report back on any issues or sharp edges that they encounter. When filing an issue about DaggerGPU.jl, please provide:
- The complete error message and backtrace
- Julia version
- GPU vendor and model
- CUDA/AMDGPU version(s)
