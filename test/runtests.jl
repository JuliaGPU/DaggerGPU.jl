using Test
using Distributed
addprocs(2, exeflags="--project")

@everywhere begin
    try using CUDA
    catch end

    try using AMDGPU
    catch end

    try using Metal
    catch end

    using Distributed, Dagger, DaggerGPU
    import DaggerGPU: Kernel
    using KernelAbstractions
end
@everywhere begin
    function isongpu(X)
        return !(X isa Array)
    end

    @kernel function fill_kernel(A, x)
        idx = @index(Global, Linear)
        A[idx] = x
    end
    function fill_thunk(A, x)
        backend = DaggerGPU.kernel_backend()
        k = fill_kernel(backend, 8)
        k(A, x; ndrange=8)
        KernelAbstractions.synchronize(backend)
        return A, typeof(A)
    end

    @kernel function copy_kernel(B, A)
        idx = @index(Global, Linear)
        B[idx] = A[idx]
    end

    # Create a function to perform an in-place operation.
    function addarray!(x)
        x .= x .+ 1.0f0
    end
end

function generate_thunks()
    as = [Dagger.spawn(x->x+1, 1) for i in 1:10]
    Dagger.spawn((xs...)->[sum(xs)], as...)
end

@test DaggerGPU.cancompute(:CUDA) ||
      DaggerGPU.cancompute(:ROC)  ||
      DaggerGPU.cancompute(:Metal)

@testset "CPU" begin
    @testset "KernelAbstractions" begin
        A = rand(Float32, 8)
        DA, T = fetch(Dagger.@spawn fill_thunk(A, 2.3f0))
        @test all(DA .== 2.3f0)
        @test T <: Array

        A = rand(Float64, 128)
        B = zeros(Float64, 128)
        Dagger.with_options(scope=Dagger.scope(worker=1,thread=1)) do
            fetch(Dagger.@spawn Kernel(copy_kernel)(B, A; ndrange=length(A)))
        end
        @test all(B .== A)
    end
end

@testset "CUDA" begin
    if !DaggerGPU.cancompute(:CUDA)
        @warn "No CUDA devices available, skipping tests"
    else
        cuproc = if isdefined(Base, :get_extension)
            Base.get_extension(DaggerGPU, :CUDAExt).CuArrayDeviceProc
        else
            CuArrayDeviceProc
        end
        @test DaggerGPU.processor(:CUDA) === cuproc
        b = generate_thunks()
        c = Dagger.with_options(;scope=Dagger.scope(cuda_gpu=1)) do
            @test fetch(Dagger.@spawn isongpu(b))
            Dagger.@spawn sum(b)
        end
        @test !fetch(Dagger.@spawn isongpu(b))
        @test fetch(Dagger.@spawn identity(c)) == 20

        @testset "KernelAbstractions" begin
            A = rand(Float32, 8)
            DA, T = Dagger.with_options(;scope=Dagger.scope(cuda_gpu=1)) do
                fetch(Dagger.@spawn fill_thunk(A, 2.3f0))
            end
            @test all(DA .== 2.3f0)
            @test T <: CuArray

            A = CUDA.rand(128)
            B = CUDA.zeros(128)
            Dagger.with_options(;scope=Dagger.scope(worker=1,cuda_gpu=1)) do
                fetch(Dagger.@spawn Kernel(copy_kernel)(B, A; ndrange=length(A)))
            end
            @test all(B .== A)
        end
    end
end

@testset "ROCm" begin
    if !DaggerGPU.cancompute(:ROC)
        @warn "No ROCm devices available, skipping tests"
    else
        rocproc = if isdefined(Base, :get_extension)
            Base.get_extension(DaggerGPU, :ROCExt).ROCArrayDeviceProc
        else
            ROCArrayDeviceProc
        end
        @test DaggerGPU.processor(:ROC) === rocproc
        b = generate_thunks()
        c = Dagger.with_options(;scope=Dagger.scope(rocm_gpu=1)) do
            @test fetch(Dagger.@spawn isongpu(b))
            Dagger.@spawn sum(b)
        end
        @test !fetch(Dagger.@spawn isongpu(b))
        @test fetch(Dagger.@spawn identity(c)) == 20

        @testset "KernelAbstractions" begin
            A = rand(Float32, 8)
            DA, T = Dagger.with_options(;scope=Dagger.scope(rocm_gpu=1)) do
                fetch(Dagger.@spawn fill_thunk(A, 2.3f0))
            end
            @test all(DA .== 2.3f0)
            @test T <: ROCArray

            A = AMDGPU.rand(128)
            B = AMDGPU.zeros(128)
            Dagger.with_options(;scope=Dagger.scope(worker=1,rocm_gpu=1)) do
                fetch(Dagger.@spawn Kernel(copy_kernel)(B, A; ndrange=length(A)))
            end
            @test all(B .== A)
        end
    end
end

@testset "Metal" begin
    if !DaggerGPU.cancompute(:Metal)
        @warn "No Metal devices available, skipping tests"
    else
        mtlproc = if isdefined(Base, :get_extension)
            Base.get_extension(DaggerGPU, :MetalExt).MtlArrayDeviceProc
        else
            MtlArrayDeviceProc
        end
        @test DaggerGPU.processor(:Metal) === mtlproc
        b = generate_thunks()
        c = Dagger.with_options(;scope=Dagger.scope(metal_gpu=1)) do
            @test fetch(Dagger.@spawn isongpu(b))
            Dagger.@spawn sum(b)
        end
        @test !fetch(Dagger.@spawn isongpu(b))
        @test fetch(Dagger.@spawn identity(c)) == 20

        @testset "KernelAbstractions" begin
            A = rand(Float32, 8)
            DA, T = Dagger.with_options(;scope=Dagger.scope(metal_gpu=1)) do
                fetch(Dagger.@spawn fill_thunk(A, 2.3f0))
            end
            @test all(DA .== 2.3f0)
            @test T <: MtlArray

            A = Metal.rand(128)
            B = Metal.zeros(128)
            Dagger.with_options(;scope=Dagger.scope(worker=1,metal_gpu=1)) do
                fetch(Dagger.@spawn Kernel(copy_kernel)(B, A; ndrange=length(A)))
            end
            @test all(B .== A)
        end

        @testset "In-place operations" begin
            # Create a page-aligned array.
            dims = (2, 2)
            T = Float32
            pagesize = ccall(:getpagesize, Cint, ())
            addr = Ref(C_NULL)

            ccall(
                :posix_memalign,
                Cint,
                (Ptr{Ptr{Cvoid}}, Csize_t, Csize_t), addr,
                pagesize,
                prod(dims) * sizeof(T)
            )

            array = unsafe_wrap(
                Array{T, length(dims)},
                reinterpret(Ptr{T}, addr[]),
                dims,
                own = false
            )

            # Initialize the array.
            array[1, 1] = 1
            array[1, 2] = 2
            array[2, 1] = 3
            array[2, 2] = 4

            # Perform the computation only on a local `MtlArrayDeviceProc`
            t = Dagger.@spawn scope=Dagger.scope(worker=myid(), metal_gpu=1) addarray!(array)

            # Fetch and check the results.
            ret = fetch(t)

            @test ret[1, 1] == 2.0f0
            @test ret[1, 2] == 3.0f0
            @test ret[2, 1] == 4.0f0
            @test ret[2, 2] == 5.0f0

            # Check if the operation happened in-place.
            @test_broken array[1, 1] == 2.0f0
            @test_broken array[1, 2] == 3.0f0
            @test_broken array[2, 1] == 4.0f0
            @test_broken array[2, 2] == 5.0f0
        end
    end
end
