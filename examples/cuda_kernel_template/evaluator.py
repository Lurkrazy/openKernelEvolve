"""
Example CUDA/CuTe DSL Evaluator for openKernelEvolve

This is a template demonstrating how to implement a CUDA kernel evaluator
following the OpenEvolve patterns from the MLX Metal example.
"""

import os
import sys
import json
import time
import tempfile
import subprocess
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path

class CUDAKernelSafetyError(Exception):
    """CUDA kernel safety violation"""
    pass

class CUDACompilationError(Exception):
    """CUDA compilation error"""
    pass

class CUDAExecutionError(Exception):
    """CUDA kernel execution error"""
    pass

class CUDAEvaluator:
    """
    Bulletproof CUDA kernel evaluator for openKernelEvolve
    
    Follows the same safety patterns as the MLX Metal evaluator:
    - Never crashes the evolution process
    - Graceful fallback on any CUDA failure
    - Comprehensive error tracking and recovery
    - Performance profiling with Nsight Compute
    """
    
    def __init__(self):
        # CUDA environment configuration
        self.cuda_version = self._detect_cuda_version()
        self.sm_targets = ["80", "86"]  # Ampere architectures
        self.cutlass_path = os.environ.get("CUTLASS_PATH", "/opt/cutlass")
        
        # Safety and retry configuration
        self.max_retry_attempts = 3
        self.compilation_timeout = 300  # 5 minutes
        self.execution_timeout = 60     # 1 minute
        self.profiling_timeout = 120    # 2 minutes
        
        # Error tracking
        self.cuda_compilation_errors = 0
        self.cuda_execution_errors = 0
        self.cuda_memory_errors = 0
        self.successful_fallbacks = 0
        
        # Performance baselines
        self.baseline_metrics = None
        
        print("🛡️ BULLETPROOF CUDA KERNEL EVALUATOR INITIALIZED")
        print(f"🔧 CUDA Version: {self.cuda_version}")
        print(f"🏗️ SM Targets: {self.sm_targets}")
        print(f"📁 CUTLASS Path: {self.cutlass_path}")
    
    def _detect_cuda_version(self) -> str:
        """Detect CUDA version from nvcc"""
        try:
            result = subprocess.run(
                ["nvcc", "--version"], 
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                # Parse CUDA version from nvcc output
                for line in result.stdout.splitlines():
                    if "release" in line:
                        return line.split("release")[1].split(",")[0].strip()
            return "unknown"
        except:
            return "not_found"
    
    def compile_cuda_kernel(self, source_code: str, kernel_name: str) -> Optional[str]:
        """
        Compile CuTe DSL kernel to CUDA binary
        
        Returns: Path to compiled binary or None on failure
        """
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                # Write source code
                source_file = Path(temp_dir) / f"{kernel_name}.cu"
                with open(source_file, 'w') as f:
                    f.write(source_code)
                
                # Compile with nvcc
                binary_file = Path(temp_dir) / f"{kernel_name}.bin"
                compile_cmd = [
                    "nvcc",
                    f"-arch=sm_{self.sm_targets[0]}",  # Primary target
                    "--ptxas-options=-v",             # Verbose ptxas output
                    "-O3",                            # Optimization
                    "-std=c++17",                     # C++17 support
                    f"-I{self.cutlass_path}/include", # CUTLASS headers
                    "-lcuda", "-lcudart",             # CUDA libraries
                    str(source_file),
                    "-o", str(binary_file)
                ]
                
                result = subprocess.run(
                    compile_cmd,
                    capture_output=True, 
                    text=True, 
                    timeout=self.compilation_timeout
                )
                
                if result.returncode == 0:
                    # Copy binary to persistent location
                    persistent_binary = f"/tmp/{kernel_name}_{int(time.time())}.bin"
                    subprocess.run(["cp", str(binary_file), persistent_binary])
                    return persistent_binary
                else:
                    self.cuda_compilation_errors += 1
                    print(f"❌ CUDA compilation failed: {result.stderr}")
                    return None
                    
        except subprocess.TimeoutExpired:
            self.cuda_compilation_errors += 1
            print(f"⏰ CUDA compilation timeout after {self.compilation_timeout}s")
            return None
        except Exception as e:
            self.cuda_compilation_errors += 1
            print(f"💥 CUDA compilation error: {e}")
            return None
    
    def profile_kernel_performance(self, binary_path: str, test_inputs: Dict) -> Dict[str, float]:
        """
        Profile kernel performance using Nsight Compute
        
        Returns: Dictionary of performance metrics
        """
        try:
            # Nsight Compute profiling command
            ncu_cmd = [
                "ncu",
                "--metrics", "sm_efficiency.avg,dram_throughput.avg,achieved_occupancy.avg",
                "--csv",
                "--target-processes", "all",
                binary_path
            ]
            
            result = subprocess.run(
                ncu_cmd,
                capture_output=True,
                text=True,
                timeout=self.profiling_timeout
            )
            
            if result.returncode == 0:
                return self._parse_ncu_output(result.stdout)
            else:
                # Fallback to basic timing
                return self._fallback_timing_measurement(binary_path, test_inputs)
                
        except subprocess.TimeoutExpired:
            print(f"⏰ Nsight Compute timeout after {self.profiling_timeout}s")
            return self._fallback_timing_measurement(binary_path, test_inputs)
        except Exception as e:
            print(f"💥 Profiling error: {e}")
            return {"kernel_time_ms": 999.0, "error": str(e)}
    
    def _parse_ncu_output(self, ncu_output: str) -> Dict[str, float]:
        """Parse Nsight Compute CSV output"""
        metrics = {}
        try:
            lines = ncu_output.strip().split('\n')
            if len(lines) >= 2:
                headers = lines[0].split(',')
                values = lines[1].split(',')
                
                for header, value in zip(headers, values):
                    try:
                        metrics[header.strip()] = float(value.strip())
                    except:
                        metrics[header.strip()] = value.strip()
                        
        except Exception as e:
            print(f"⚠️ NCU output parsing failed: {e}")
            
        return metrics
    
    def _fallback_timing_measurement(self, binary_path: str, test_inputs: Dict) -> Dict[str, float]:
        """Fallback timing measurement using CUDA events"""
        try:
            # Simple CUDA event timing (implementation depends on specific kernel interface)
            start_time = time.time()
            
            # Execute kernel (simplified - actual implementation needs proper CUDA integration)
            result = subprocess.run([binary_path], timeout=self.execution_timeout)
            
            end_time = time.time()
            kernel_time = (end_time - start_time) * 1000  # Convert to milliseconds
            
            return {
                "kernel_time_ms": kernel_time,
                "sm_efficiency": 0.0,  # Unknown without profiling
                "achieved_occupancy": 0.0,
                "fallback_timing": True
            }
            
        except Exception as e:
            return {"kernel_time_ms": 999.0, "error": str(e)}
    
    def validate_correctness(self, cuda_output, reference_output, tolerance: float = 1e-3) -> Dict[str, Any]:
        """
        Validate CUDA kernel correctness against reference implementation
        
        Args:
            cuda_output: Output from CUDA kernel
            reference_output: Reference implementation output (PyTorch/CuBLAS)
            tolerance: Acceptable numerical difference
            
        Returns:
            Correctness metrics
        """
        try:
            # Compute element-wise differences
            diff = abs(cuda_output - reference_output)
            max_diff = float(diff.max())
            mean_diff = float(diff.mean())
            
            # Correctness score (1.0 = perfect, 0.0 = completely wrong)
            correctness_score = max(0.0, 1.0 - (max_diff / tolerance))
            
            return {
                "correctness_score": correctness_score,
                "max_absolute_error": max_diff,
                "mean_absolute_error": mean_diff,
                "tolerance": tolerance,
                "passed": max_diff <= tolerance
            }
            
        except Exception as e:
            return {
                "correctness_score": 0.0,
                "error": str(e),
                "passed": False
            }
    
    def safe_kernel_evaluation(self, source_code: str, kernel_name: str) -> Dict[str, Any]:
        """
        Safely evaluate a CUDA kernel with comprehensive error handling
        
        This is the main entry point that mirrors the MLX evaluator pattern
        """
        evaluation_result = {
            "score": 0.0,
            "correctness": 0.0,
            "performance": 0.0,
            "compilation_success": False,
            "execution_success": False,
            "artifacts": {}
        }
        
        for attempt in range(self.max_retry_attempts):
            try:
                # Stage 1: Compilation
                print(f"🔨 Attempt {attempt + 1}: Compiling CUDA kernel...")
                binary_path = self.compile_cuda_kernel(source_code, kernel_name)
                
                if not binary_path:
                    continue  # Retry compilation
                
                evaluation_result["compilation_success"] = True
                
                # Stage 2: Correctness validation
                print(f"✅ Validating correctness...")
                # NOTE: Actual implementation would run kernel and compare with reference
                correctness_metrics = {"correctness_score": 0.8, "passed": True}  # Placeholder
                evaluation_result["correctness"] = correctness_metrics["correctness_score"]
                
                if not correctness_metrics["passed"]:
                    continue  # Retry if correctness fails
                
                # Stage 3: Performance profiling
                print(f"📊 Profiling performance...")
                performance_metrics = self.profile_kernel_performance(binary_path, {})
                
                # Calculate performance score relative to baseline
                if "kernel_time_ms" in performance_metrics and self.baseline_metrics:
                    speedup = self.baseline_metrics["kernel_time_ms"] / performance_metrics["kernel_time_ms"]
                    performance_score = min(1.0, speedup / 2.0)  # Cap at 2x speedup = 1.0 score
                else:
                    performance_score = 0.5  # Default if no baseline
                
                evaluation_result["performance"] = performance_score
                evaluation_result["execution_success"] = True
                
                # Combined score (weighted correctness + performance)
                evaluation_result["score"] = (
                    0.7 * evaluation_result["correctness"] + 
                    0.3 * evaluation_result["performance"]
                )
                
                # Store artifacts
                evaluation_result["artifacts"] = {
                    "binary_path": binary_path,
                    "performance_metrics": performance_metrics,
                    "correctness_metrics": correctness_metrics,
                    "attempt_number": attempt + 1,
                    "compilation_time": performance_metrics.get("compilation_time", 0.0)
                }
                
                print(f"🎯 Evaluation successful: score={evaluation_result['score']:.3f}")
                return evaluation_result
                
            except CUDAKernelSafetyError as e:
                print(f"🛡️ Safety violation in attempt {attempt + 1}: {e}")
                continue
            except Exception as e:
                print(f"💥 Unexpected error in attempt {attempt + 1}: {e}")
                continue
        
        # All attempts failed - return failure with fallback
        print(f"❌ All {self.max_retry_attempts} attempts failed, falling back to baseline")
        self.successful_fallbacks += 1
        
        evaluation_result["score"] = 0.1  # Small penalty for failure
        evaluation_result["artifacts"]["error"] = "All evaluation attempts failed"
        
        return evaluation_result

# OpenEvolve integration function
def evaluate(program_path: str) -> Dict[str, Any]:
    """
    Main evaluator function for OpenEvolve integration
    
    This function will be called by OpenEvolve for each candidate kernel.
    It must return a dictionary with at least a 'score' field.
    """
    try:
        # Load the evolved CUDA kernel source code
        with open(program_path, 'r') as f:
            source_code = f.read()
        
        # Extract kernel name from filename
        kernel_name = Path(program_path).stem
        
        # Initialize evaluator
        evaluator = CUDAEvaluator()
        
        # Perform safe evaluation
        result = evaluator.safe_kernel_evaluation(source_code, kernel_name)
        
        # Return in OpenEvolve expected format
        return {
            "score": result["score"],
            "correctness": result["correctness"], 
            "performance": result["performance"],
            "compilation_success": result["compilation_success"],
            "execution_success": result["execution_success"],
            "artifacts": result["artifacts"]
        }
        
    except Exception as e:
        # Ultimate fallback - never crash OpenEvolve
        return {
            "score": 0.0,
            "error": str(e),
            "artifacts": {"evaluation_failed": True}
        }

# Optional cascade evaluation stages for efficiency
def evaluate_stage1(program_path: str) -> Dict[str, Any]:
    """Stage 1: Quick validation - syntax and basic compilation check"""
    try:
        with open(program_path, 'r') as f:
            source_code = f.read()
        
        # Basic syntax validation
        if "#include" not in source_code or "template" not in source_code:
            return {"score": 0.0, "stage": 1, "error": "Invalid CuTe DSL syntax"}
        
        # Quick compilation test
        evaluator = CUDAEvaluator()
        kernel_name = Path(program_path).stem + "_stage1"
        binary_path = evaluator.compile_cuda_kernel(source_code, kernel_name)
        
        if binary_path:
            return {"score": 0.6, "stage": 1, "compilation_success": True}
        else:
            return {"score": 0.0, "stage": 1, "compilation_failed": True}
            
    except Exception as e:
        return {"score": 0.0, "stage": 1, "error": str(e)}

if __name__ == "__main__":
    # Demo usage
    print("🚀 CUDA Kernel Evaluator Demo")
    
    # Example CuTe DSL kernel source
    demo_kernel = '''
    #include <cute/tensor.hpp>
    #include <cuda_runtime.h>
    
    template<typename T>
    __global__ void demo_gemm_kernel(T* A, T* B, T* C, int M, int N, int K) {
        // Simple GEMM implementation for demo
        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;
        
        if (row < M && col < N) {
            T sum = 0;
            for (int k = 0; k < K; k++) {
                sum += A[row * K + k] * B[k * N + col];
            }
            C[row * N + col] = sum;
        }
    }
    '''
    
    # Save demo kernel
    with tempfile.NamedTemporaryFile(mode='w', suffix='.cu', delete=False) as f:
        f.write(demo_kernel)
        demo_path = f.name
    
    # Evaluate demo kernel
    result = evaluate(demo_path)
    print(f"📊 Demo evaluation result: {json.dumps(result, indent=2)}")
    
    # Cleanup
    os.unlink(demo_path)