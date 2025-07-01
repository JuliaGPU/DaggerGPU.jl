module DaggerGPU

import Dagger: Kernel, gpu_processor, gpu_can_compute, with_device, move_optimized, gpu_kernel_backend

@deprecate processor gpu_processor
@deprecate cancompute gpu_can_compute
@deprecate kernel_backend gpu_kernel_backend

end
