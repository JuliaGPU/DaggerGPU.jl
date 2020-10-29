# DaggerGPU

**GPU integrations for Dagger.jl**

DaggerGPU.jl makes use of the `Dagger.Processor` infrastructure to dispatch Dagger kernels to NVIDIA and AMD GPUs, via CUDA.jl and AMDGPU.jl respectively. Usage is simple: `add` or `dev` DaggerGPU.jl and CUDA.jl/AMDGPU.jl appropriately, load it with `using DaggerGPU`, and add `DaggerGPU.CuArrayDeviceProc`/`DaggerGPU.ROCArrayProc` to your scheduler or thunk options (see Dagger.jl documentation for details on how to do this).

DaggerGPU.jl is still experimental, but we welcome GPU-owning users to try it out and report back on any issues or sharp edges that they encounter. When filing an issue about DaggerGPU.jl, please provide:
- The complete error message and backtrace
- Julia version
- GPU vendor and model
- CUDA/AMDGPU version(s)
