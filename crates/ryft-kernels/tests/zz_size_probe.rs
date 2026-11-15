#[test]
fn size_probe() {
    use std::mem::size_of;
    eprintln!("SIZE ProgramError {}", size_of::<ryft_core::ProgramError>());
    eprintln!("SIZE TypeError {}", size_of::<ryft_core::TypeError>());
    eprintln!("SIZE KernelError {}", size_of::<ryft_kernels::calls::KernelError>());
    eprintln!("SIZE GridError {}", size_of::<ryft_kernels::GridError>());
    eprintln!("SIZE BlockMappingError {}", size_of::<ryft_kernels::BlockMappingError>());
    eprintln!("SIZE KernelValidationError {}", size_of::<ryft_kernels::validation::KernelValidationError>());
    eprintln!("SIZE KernelMemoryError {}", size_of::<ryft_kernels::memory::KernelMemoryError>());
    eprintln!("SIZE KernelInitializationError {}", size_of::<ryft_kernels::KernelInitializationError>());
    eprintln!("SIZE KernelSerializationError {}", size_of::<ryft_kernels::KernelSerializationError>());
    eprintln!("SIZE KernelTransformError {}", size_of::<ryft_kernels::KernelTransformError>());
    eprintln!("SIZE KernelCompilationError {}", size_of::<ryft_kernels::KernelCompilationError>());
    eprintln!("SIZE KernelSchedulingError {}", size_of::<ryft_kernels::KernelSchedulingError>());
}
