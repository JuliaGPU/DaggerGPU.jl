using Distributed
using Test
addprocs(2, exeflags="--project")

@everywhere begin
    using CUDA, AMDGPU, Metal, KernelAbstractions
    using Distributed, Dagger, DaggerGPU
end
@everywhere begin
    function myfunc(X)
        @assert !(X isa Array)
        X
    end

    KernelAbstractions.@kernel function fill_kernel(A, x)
        idx = @index(Global, Linear)
        A[idx] = x
    end
    function fill_thunk(A, x)
        k = fill_kernel(DaggerGPU.kernel_backend(), 8)
        wait(k(A, x; ndrange=8))
        @show A
        A
    end

    # Create a function to perform an in-place operation.
    function addarray!(x)
        x .= x .+ 1.0f0
    end
end

function generate_thunks()
    as = [delayed(x->x+1)(1) for i in 1:10]
    delayed((xs...)->[sum(xs)])(as...)
end

@test DaggerGPU.cancompute(:CUDA) ||
      DaggerGPU.cancompute(:ROC)  ||
      DaggerGPU.cancompute(:Metal)

@testset "CPU" begin
    @testset "KernelAbstractions" begin
        A = rand(Float32, 8)
        _A = collect(delayed(fill_thunk)(A, 2.3f0))
        @test all(_A .== 2.3f0)
    end
end

@testset "CUDA" begin
    if !DaggerGPU.cancompute(:CUDA)
        @warn "No CUDA devices available, skipping tests"
    else
        cuproc = DaggerGPU.processor(:CUDA)
        b = generate_thunks()
        opts = Dagger.Sch.ThunkOptions(;proclist=[cuproc])
        c_pre = delayed(myfunc; options=opts)(b)
        c = delayed(sum; options=opts)(b)

        opts = Dagger.Sch.ThunkOptions(;proclist=[Dagger.ThreadProc])
        d = delayed(identity; options=opts)(c)
        @test collect(d) == 20

        @test_skip "KernelAbstractions"
        #= FIXME
        @testset "KernelAbstractions" begin
            cuproc = DaggerGPU.processor(:CUDA)
            opts = Dagger.Sch.ThunkOptions(;proclist=[cuproc])
            A = rand(Float32, 8)
            _A = collect(delayed(fill_thunk)(A, 2.3); options=opts)
            @test all(_A .== 2.3)
        end
        =#
    end
end

@testset "ROCm" begin
    if !DaggerGPU.cancompute(:ROC)
        @warn "No ROCm devices available, skipping tests"
    else
        rocproc = DaggerGPU.processor(:ROC)
        b = generate_thunks()
        opts = Dagger.Sch.ThunkOptions(;proclist=[rocproc])
        c_pre = delayed(myfunc; options=opts)(b)
        c = delayed(sum; options=opts)(b)

        opts = Dagger.Sch.ThunkOptions(;proclist=[Dagger.ThreadProc])
        d = delayed(identity; options=opts)(c)
        @test collect(d) == 20

        @test_skip "KernelAbstractions"
        #= FIXME
        @testset "KernelAbstractions" begin
            rocproc = DaggerGPU.processor(:ROC)
            opts = Dagger.Sch.ThunkOptions(;proclist=[rocproc])
            A = rand(Float32, 8)
            _A = collect(delayed(fill_thunk)(A, 2.3); options=opts)
            @test all(_A .== 2.3)
        end
        =#
    end
end

@testset "Metal" begin
    if !DaggerGPU.cancompute(:Metal)
        @warn "No Metal devices available, skipping tests"
    else
        metalproc = DaggerGPU.processor(:Metal)
        b = generate_thunks()
        opts = Dagger.Sch.ThunkOptions(;proclist = [metalproc])
        c_pre = delayed(myfunc; options = opts)(b)
        c = delayed(sum; options = opts)(b)

        opts = Dagger.Sch.ThunkOptions(;proclist = [Dagger.ThreadProc])
        d = delayed(identity; options = opts)(c)
        @test collect(d) == 20

        # It seems KernelAbstractions does not support Metal.jl.
        @test_skip "KernelAbstractions"

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
            t = Dagger.@spawn single=myid() proclist = [metalproc] addarray!(array)

            # Fetch and check the results.
            ret = fetch(t)

            @test ret[1, 1] == 2.0f0
            @test ret[1, 2] == 3.0f0
            @test ret[2, 1] == 4.0f0
            @test ret[2, 2] == 5.0f0

            # Check if the operation happened in-place.
            @test array[1, 1] == 2.0f0
            @test array[1, 2] == 3.0f0
            @test array[2, 1] == 4.0f0
            @test array[2, 2] == 5.0f0
        end
    end
end
