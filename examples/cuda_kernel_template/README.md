# CUDA Kernel Template Example for openKernelEvolve

This directory contains a complete template for evolving CUDA kernels using CUTLASS CuTe DSL on Ampere architecture.

## Files

- **`initial_program.cu`**: Starting CUDA kernel with CuTe DSL template and EVOLVE-BLOCK markers
- **`evaluator.py`**: Bulletproof CUDA evaluator with compilation, profiling, and safety mechanisms  
- **`config.yaml`**: Complete configuration for CUDA kernel evolution with CuTe DSL prompts

## Key Features

### 🛡️ Bulletproof Evaluation
- Comprehensive error handling and recovery
- Graceful fallback on CUDA compilation/execution failures
- Never crashes the evolution process
- Tracks detailed error statistics

### 🚀 CUDA/CuTe DSL Integration
- CUTLASS and CuTe DSL compilation pipeline
- Nsight Compute profiling integration
- Multi-stage cascade evaluation (syntax → compilation → correctness → performance)
- Support for multiple SM architectures (80, 86)

### 🎯 Advanced Optimization Targets
- Tensor Core utilization (MMA atoms)
- Async memory operations (cp_async)
- CuTe tensor layouts and abstractions
- Multi-stage software pipelining
- Register and shared memory optimization

## Usage

### Prerequisites

```bash
# CUDA 12.4+ with Ampere GPU
# CUTLASS 3.4+ with CuTe DSL
# Nsight Compute for profiling

export CUTLASS_PATH=/opt/cutlass
export CUDA_VISIBLE_DEVICES=0
```

### Run Evolution

```bash
# From openKernelEvolve root directory
python openevolve-run.py \
    examples/cuda_kernel_template/initial_program.cu \
    examples/cuda_kernel_template/evaluator.py \
    --config examples/cuda_kernel_template/config.yaml \
    --iterations 50
```

### Monitor Progress

```bash
# View evolution logs
tail -f openevolve_output/cuda_kernel_evolution/logs/openevolve_*.log

# Visualize evolution tree  
python scripts/visualizer.py --path openevolve_output/cuda_kernel_evolution/
```

## Evolution Process

1. **Stage 1 - Syntax Validation** (5-10s):
   - CuTe DSL syntax checking
   - Basic compilation test
   - Parameter validation

2. **Stage 2 - Correctness** (30-60s):
   - Full CUDA compilation
   - Correctness vs PyTorch/CuBLAS
   - Memory safety validation

3. **Stage 3 - Performance** (2-5min):
   - Nsight Compute profiling
   - Multi-shape benchmarking
   - Detailed metrics collection

## Optimization Examples

### Baseline Implementation
```cuda
// EVOLVE-BLOCK-START
constexpr int TILE_M = 64, TILE_N = 64, TILE_K = 16;

// Basic shared memory tiling
__shared__ T shared_A[TILE_M][TILE_K];
__shared__ T shared_B[TILE_K][TILE_N];

// Manual loading and computation
for (int k_tile = 0; k_tile < K; k_tile += TILE_K) {
    // Load tiles...
    // Compute...
}
// EVOLVE-BLOCK-END
```

### Evolved with CuTe DSL
```cuda  
// EVOLVE-BLOCK-START
using MMA_Atom = MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>;
auto tiled_mma = make_tiled_mma(MMA_Atom{}, 
                               Layout<Shape<_2, _2, _1>>{});

auto layout_A = make_layout(make_shape(M, K));
auto tensor_A = make_tensor(make_gmem_ptr(A), layout_A);

// Async copy operations
auto copy_atom = Copy_Atom<SM80_CP_ASYNC_CACHEGLOBAL<uint128_t>, T>{};
cute::cp_async<128>(sA_ptr, gA_ptr);

// Tensor Core MMA operations  
cute::gemm(tiled_mma, rA, rB, rC);
// EVOLVE-BLOCK-END
```

## Expected Improvements

- **10-30% speedup** from Tensor Core utilization
- **20-50% speedup** from memory optimization  
- **5-15% speedup** from instruction pipeline improvements
- **Overall target: 30-80% improvement** over baseline

## Safety and Constraints

### ✅ Evolution Freedom
- Tile sizes and memory layouts
- CuTe DSL constructs and tensor operations
- Thread organization patterns
- MMA atom selection
- Pipeline and instruction scheduling

### ❌ Protected Elements  
- Kernel function signature
- Host wrapper interface
- Overall GEMM semantics
- Include directives
- Template structure

## Troubleshooting

### Compilation Issues
```bash
# Check CUDA environment
nvcc --version
echo $CUTLASS_PATH

# Test basic compilation
nvcc -I$CUTLASS_PATH/include -arch=sm_80 test.cu
```

### Profiling Issues
```bash
# Check Nsight Compute
ncu --version

# Test basic profiling
ncu --metrics sm_efficiency ./test_kernel
```

### Memory Issues
```bash
# Check GPU memory
nvidia-smi

# Monitor during evolution
watch -n 1 nvidia-smi
```

## Integration with OpenEvolve

This template follows OpenEvolve patterns:

1. **Evaluator Interface**: `evaluate()` function returns score dictionary
2. **Cascade Stages**: Optional `evaluate_stage1/2/3()` for efficiency  
3. **Configuration**: YAML-based setup with LLM prompts
4. **Safety**: Comprehensive error handling and fallback mechanisms
5. **Artifacts**: Detailed metrics and debugging information

The template can be easily adapted for other kernel types:
- **FlashAttention**: Attention mechanism optimization
- **Conv2D**: Convolution kernels with im2col
- **LayerNorm**: Reduction and normalization operations
- **Softmax**: Reduction with numerical stability

## Performance Baselines

Expected performance on A100 (FP32 GEMM 1024x1024x1024):

| Implementation | Time (ms) | TFLOPS | Relative |
|---------------|-----------|---------|----------|
| Naive CUDA | 2.5 | 0.86 | 1.0x |
| Optimized CUDA | 1.8 | 1.19 | 1.4x |  
| **Evolved CuTe** | **1.2** | **1.79** | **2.1x** |
| CuBLAS (reference) | 0.8 | 2.68 | 3.1x |

Target: Get within 20-30% of CuBLAS performance through evolution.