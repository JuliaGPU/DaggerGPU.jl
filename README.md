# DaggerGPU

**GPU integrations for Dagger.jl**

## Deprecation Notice

DaggerGPU's logic and extensions have been merged upstream into Dagger.jl, making DaggerGPU no longer necessary. You can now load CUDA logic with `using Dagger, CUDA`, and similarly for all other backends. Please report all issues to the [Dagger issue tracker](https://github.com/JuliaParallel/Dagger.jl/issues).

## Original README

DaggerGPU.jl makes use of the `Dagger.Processor` infrastructure to dispatch Dagger kernels to NVIDIA, AMD, and Apple GPUs, via CUDA.jl, AMDGPU.jl, and Metal.jl respectively. Usage is simple: `add` or `dev` DaggerGPU.jl and CUDA.jl/AMDGPU.jl/Metal.jl appropriately, load it with `using DaggerGPU`, and add `DaggerGPU.CuArrayDeviceProc`/`DaggerGPU.ROCArrayProc`/`DaggerGPU.MtlArrayDeviceProc` to your scheduler or thunk options (see Dagger.jl documentation for details on how to do this).

DaggerGPU.jl is still experimental, but we welcome GPU-owning users to try it out and report back on any issues or sharp edges that they encounter. When filing an issue about DaggerGPU.jl, please provide:
- The complete error message and backtrace
- Julia version
- GPU vendor and model
- CUDA/AMDGPU version(s)
