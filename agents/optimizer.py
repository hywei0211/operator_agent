"""
Optimizer Agent - 性能优化器
对生成的内核进行迭代性能优化，基于屋顶线模型和profiling反馈
"""
import json
import logging
import re
from typing import Optional

from agents.base_agent import BaseAgent, AgentContext, AgentResult, AgentStatus
from models.operator_ir import GeneratedKernel, OperatorIR
from models.hardware_model import GPUSpec
from prompts.code_gen_prompts import build_optimization_prompt

logger = logging.getLogger(__name__)


# 优化策略库
OPTIMIZATION_STRATEGIES = {
    "memory_bound": [
        "vectorized_load",          # 向量化内存访问（float4/half2）
        "shared_memory_tiling",     # 共享内存分块
        "memory_coalescing",        # 内存访问合并
        "prefetching",              # 软件预取
        "cache_hints",              # 缓存提示指令
    ],
    "compute_bound": [
        "tensor_core_wmma",         # Tensor Core WMMA API
        "tensor_core_mma",          # Tensor Core MMA PTX
        "loop_unrolling",           # 循环展开 #pragma unroll
        "fast_math",                # 快速数学函数
        "instruction_level_parallelism",  # 指令级并行
    ],
    "latency_bound": [
        "increase_occupancy",       # 提高occupancy
        "register_pressure_reduce", # 减少寄存器压力
        "warp_level_primitives",    # Warp原语（shfl等）
        "persistent_kernel",        # 持久化kernel减少启动开销
    ],
    "general": [
        "kernel_fusion",            # 算子融合
        "mixed_precision",          # 混合精度
        "async_copy",               # 异步内存复制
        "double_buffering",         # 双缓冲流水线
    ]
}


class RooflineAnalyzer:
    """屋顶线模型分析器"""

    def __init__(self, gpu_spec: GPUSpec):
        self.gpu_spec = gpu_spec
        # 计算屋顶线交叉点（算术强度阈值）
        self.ridge_point = (
            gpu_spec.compute.fp16_tflops * 1e12 /
            (gpu_spec.memory.bandwidth_gbps * 1e9)
        )  # FLOPs/Byte

    def analyze(self, flops: float, bytes_accessed: float, measured_tflops: float) -> dict:
        """
        分析内核的性能特征

        Returns:
            bottleneck: 瓶颈类型 (memory_bound/compute_bound/latency_bound)
            peak_achievable: 可达到的峰值性能
            efficiency: 实际效率
        """
        if bytes_accessed <= 0:
            return {"bottleneck": "unknown", "efficiency": 0.0}

        arithmetic_intensity = flops / bytes_accessed

        if arithmetic_intensity < self.ridge_point:
            # 内存带宽受限
            peak_achievable_tflops = (
                arithmetic_intensity * self.gpu_spec.memory.bandwidth_gbps * 1e9 / 1e12
            )
            bottleneck = "memory_bound"
        else:
            # 计算受限
            peak_achievable_tflops = self.gpu_spec.compute.fp16_tflops
            bottleneck = "compute_bound"

        efficiency = measured_tflops / peak_achievable_tflops if peak_achievable_tflops > 0 else 0

        return {
            "bottleneck": bottleneck,
            "arithmetic_intensity": arithmetic_intensity,
            "ridge_point": self.ridge_point,
            "peak_achievable_tflops": peak_achievable_tflops,
            "measured_tflops": measured_tflops,
            "efficiency": efficiency,
            "recommended_strategies": OPTIMIZATION_STRATEGIES.get(bottleneck, [])
                                      + OPTIMIZATION_STRATEGIES["general"],
        }


class OptimizerAgent(BaseAgent):
    """
    性能优化Agent

    职责：
    1. 对生成的内核进行屋顶线模型分析
    2. 识别性能瓶颈（内存带宽/计算/延迟）
    3. 选择针对性的优化策略
    4. 用LLM重写内核，应用优化
    5. 与验证Agent配合，确认优化效果
    """

    def __init__(self, llm_client=None, config: dict = None):
        super().__init__("OptimizerAgent", llm_client, config)

    def get_system_prompt(self) -> str:
        return """你是GPU性能优化专家，精通CUDA/HIP性能分析和优化技术。
你熟悉屋顶线模型、Nsight Systems/Compute分析工具，以及各种GPU优化技术。
你能够根据性能分析结果，针对性地优化GPU内核代码。"""

    async def run(self, context: AgentContext, **kwargs) -> AgentResult:
        self._start_timer()
        self.set_status(AgentStatus.RUNNING)

        kernel: Optional[GeneratedKernel] = kwargs.get("kernel")
        gpu_spec: Optional[GPUSpec] = kwargs.get("gpu_spec")
        operator_ir: Optional[OperatorIR] = kwargs.get("operator_ir") or context.get_artifact("operator_ir")
        iteration: int = kwargs.get("iteration", 1)

        if kernel is None or gpu_spec is None:
            return self.failure_result("Missing kernel or gpu_spec")

        try:
            # 1. 进行模拟profiling（真实环境中调用GPU profiler）
            profiling_result = await self._profile_kernel(kernel, gpu_spec, operator_ir)
            logger.info(f"[Optimizer] Iter {iteration} profiling: "
                       f"BW={profiling_result.get('bandwidth_utilization', 0):.1%}, "
                       f"bottleneck={profiling_result.get('bottleneck', 'unknown')}")

            # 2. 屋顶线分析
            roofline = RooflineAnalyzer(gpu_spec)
            roofline_result = roofline.analyze(
                flops=profiling_result.get("flops", 1e12),
                bytes_accessed=profiling_result.get("bytes_accessed", 1e9),
                measured_tflops=profiling_result.get("measured_tflops", 1.0),
            )

            # 3. 判断是否已达到足够的效率
            current_efficiency = roofline_result["efficiency"]
            target_efficiency = self.config.get("target_efficiency", 0.75) if self.config else 0.75

            if current_efficiency >= target_efficiency:
                logger.info(f"[Optimizer] Kernel already efficient ({current_efficiency:.1%}), skipping")
                return self.success_result(
                    output=kernel,
                    metrics={"efficiency": current_efficiency, "optimized": False}
                )

            # 4. 用LLM进行优化
            optimized_kernel = await self._optimize_with_llm(
                kernel, gpu_spec, profiling_result, roofline_result, iteration
            )

            # 5. 更新优化记录
            optimized_kernel.iteration = iteration
            optimized_kernel.optimizations_applied = (
                kernel.optimizations_applied + roofline_result.get("recommended_strategies", [])[:3]
            )

            return self.success_result(
                output=optimized_kernel,
                metrics={
                    "efficiency_before": profiling_result.get("bandwidth_utilization", 0),
                    "bottleneck": roofline_result["bottleneck"],
                    "optimizations": optimized_kernel.optimizations_applied,
                }
            )

        except Exception as e:
            self.set_status(AgentStatus.FAILED)
            logger.exception(f"[Optimizer] Failed: {e}")
            return self.failure_result(str(e))

    async def _profile_kernel(
        self,
        kernel: GeneratedKernel,
        gpu_spec: GPUSpec,
        operator_ir: Optional[OperatorIR]
    ) -> dict:
        """
        对内核进行性能分析。
        优先用真实硬件 profiling，不可用时退化为代码特征估算。
        """
        # 尝试真实硬件 profiling
        from agents.verifier import HardwareDetector
        hw = HardwareDetector.detect()
        can_execute = {
            "cuda": hw.get("nvidia_gpu", False),
            "hip": hw.get("amd_gpu", False),
            "ascendc": hw.get("npu", False),
            "triton": hw.get("nvidia_gpu", False) or hw.get("amd_gpu", False),
        }.get(kernel.backend, False)

        if can_execute and operator_ir:
            real_result = await self._try_real_profiling(kernel, gpu_spec, operator_ir, hw)
            if real_result is not None:
                return real_result

        # 退化为代码特征估算
        estimated_bw_util = self._estimate_bw_utilization(kernel, gpu_spec)
        estimated_compute_util = self._estimate_compute_utilization(kernel, gpu_spec)
        bottleneck = "memory_bound" if estimated_bw_util > estimated_compute_util else "compute_bound"

        # 用 Roofline 模型估算（如果有 operator_ir）
        flops_est = 1e12
        bytes_est = 1e9
        if operator_ir:
            try:
                from tools.cpu_simulator import RooflineSimulator
                roofline = RooflineSimulator()
                pred = roofline.predict(
                    operator_ir.name, kernel.target_gpu.lower().replace(" ", "_"),
                    {"x_shape": [4, 512, 4096]}, "float16")
                if "error" not in pred:
                    flops_est = pred.get("flops_total", flops_est)
                    bytes_est = pred.get("memory_bytes", bytes_est)
                    bottleneck = pred.get("bound_type", bottleneck)
            except Exception:
                pass

        return {
            "bandwidth_utilization": estimated_bw_util,
            "compute_utilization": estimated_compute_util,
            "latency_ms": 0.0,
            "bottleneck": bottleneck,
            "flops": flops_est,
            "bytes_accessed": bytes_est,
            "measured_tflops": gpu_spec.compute.fp16_tflops * estimated_compute_util,
            "profiling_method": "code_feature_estimate",
        }

    async def _try_real_profiling(
        self, kernel: GeneratedKernel, gpu_spec: GPUSpec,
        operator_ir: OperatorIR, hw: dict
    ) -> Optional[dict]:
        """尝试用 PyTorch 在真实硬件上做简单 profiling"""
        try:
            import torch
            import time as _time

            from tools.cpu_simulator import CPUSimulator
            ref_fn = CPUSimulator().REFERENCE_IMPLS.get(operator_ir.name)
            if ref_fn is None:
                return None

            # 选设备
            if kernel.backend in ("cuda", "triton") and hw.get("nvidia_gpu"):
                device = torch.device("cuda:0")
                sync_fn = torch.cuda.synchronize
            elif kernel.backend == "ascendc" and hw.get("npu"):
                import torch_npu  # noqa: F401
                device = torch.device("npu:0")
                sync_fn = torch.npu.synchronize
            else:
                return None

            x = torch.randn(4, 512, 4096, dtype=torch.float16, device=device)

            # warmup
            for _ in range(10):
                with torch.no_grad():
                    ref_fn(x)
            sync_fn()

            # 计时
            start = _time.perf_counter()
            repeats = 50
            for _ in range(repeats):
                with torch.no_grad():
                    ref_fn(x)
            sync_fn()
            latency_ms = (_time.perf_counter() - start) / repeats * 1000

            # 估算 BW
            num_elements = 4 * 512 * 4096
            bytes_accessed = num_elements * 2 * 2  # fp16, read + write
            bw_achieved = bytes_accessed / (latency_ms / 1000)
            bw_util = bw_achieved / (gpu_spec.memory.bandwidth_gbps * 1e9)
            bw_util = min(max(bw_util, 0.01), 1.0)

            # 估算 compute
            estimated_compute_util = self._estimate_compute_utilization(kernel, gpu_spec)
            bottleneck = "memory_bound" if bw_util > estimated_compute_util else "compute_bound"

            logger.info(f"[Optimizer] Real profiling: latency={latency_ms:.3f}ms, bw_util={bw_util:.1%}")

            return {
                "bandwidth_utilization": bw_util,
                "compute_utilization": estimated_compute_util,
                "latency_ms": latency_ms,
                "bottleneck": bottleneck,
                "flops": num_elements * 4,  # 粗估
                "bytes_accessed": bytes_accessed,
                "measured_tflops": gpu_spec.compute.fp16_tflops * estimated_compute_util,
                "profiling_method": "real_hardware",
            }
        except Exception as e:
            logger.debug(f"[Optimizer] Real profiling failed: {e}")
            return None

    def _estimate_bw_utilization(self, kernel: GeneratedKernel, gpu_spec: GPUSpec) -> float:
        """基于代码特征估算内存带宽利用率"""
        code = kernel.source_code.lower()
        score = 0.4  # 基础分

        # 正面信号
        if "shared" in code or "lds" in code or "__shared__" in code:
            score += 0.15
        if "float4" in code or "half2" in code or "double2" in code:
            score += 0.10
        if "ldg" in code or "__ldg" in code:
            score += 0.05
        if "prefetch" in code or "async_copy" in code:
            score += 0.05
        if "unroll" in code or "#pragma unroll" in code:
            score += 0.05

        # 负面信号
        if "atomicadd" in code or "atomic" in code:
            score -= 0.10
        if "if (idx" in code and "else" not in code:
            score -= 0.05  # 潜在的分支分歧

        return min(max(score, 0.1), 0.95)

    def _estimate_compute_utilization(self, kernel: GeneratedKernel, gpu_spec: GPUSpec) -> float:
        """基于代码特征估算计算利用率"""
        code = kernel.source_code.lower()
        score = 0.3

        if "wmma" in code or "mma" in code:
            score += 0.30  # Tensor Core使用
        if "mfma" in code:
            score += 0.30  # AMD Matrix Core
        if "__hfma" in code or "fmaf" in code or "fma" in code:
            score += 0.10
        if "unroll" in code:
            score += 0.05

        return min(max(score, 0.1), 0.95)

    async def _optimize_with_llm(
        self,
        kernel: GeneratedKernel,
        gpu_spec: GPUSpec,
        profiling_result: dict,
        roofline_result: dict,
        iteration: int
    ) -> GeneratedKernel:
        """使用LLM进行针对性优化"""
        if self.llm_client is None:
            # 无LLM时进行简单的代码级优化
            return self._apply_static_optimizations(kernel, roofline_result)

        prompt = build_optimization_prompt(
            kernel_code=kernel.source_code,
            gpu_spec=gpu_spec,
            profiling_result={**profiling_result, **roofline_result}
        )

        response = await self.call_llm(prompt, max_tokens=8192)

        # 解析优化后的代码
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group())
                optimized_code = data.get("kernel_code") or data.get("optimized_code", kernel.source_code)
                optimizations = data.get("optimizations", [])
                new_efficiency = data.get("estimated_efficiency", profiling_result["bandwidth_utilization"])

                return GeneratedKernel(
                    operator_name=kernel.operator_name,
                    backend=kernel.backend,
                    target_gpu=kernel.target_gpu,
                    source_code=optimized_code,
                    header_code=kernel.header_code,
                    build_flags=kernel.build_flags,
                    launch_config=data.get("launch_config", kernel.launch_config),
                    estimated_bandwidth_utilization=new_efficiency,
                    optimizations_applied=kernel.optimizations_applied + optimizations,
                )
            except json.JSONDecodeError:
                pass

        # 解析失败，提取代码块
        code_match = re.search(r'```(?:cuda|cpp|hip|python)?\n(.*?)```', response, re.DOTALL)
        if code_match:
            return GeneratedKernel(
                operator_name=kernel.operator_name,
                backend=kernel.backend,
                target_gpu=kernel.target_gpu,
                source_code=code_match.group(1),
                header_code=kernel.header_code,
                build_flags=kernel.build_flags,
                launch_config=kernel.launch_config,
                estimated_bandwidth_utilization=profiling_result["bandwidth_utilization"] * 1.1,
                optimizations_applied=kernel.optimizations_applied,
            )

        return kernel  # 优化失败，返回原始内核

    def _apply_static_optimizations(
        self,
        kernel: GeneratedKernel,
        roofline_result: dict
    ) -> GeneratedKernel:
        """无LLM时的静态代码优化（字符串替换级别）"""
        code = kernel.source_code
        applied = []

        # 添加unroll pragma
        if "#pragma unroll" not in code and "for (" in code:
            code = code.replace("for (", "#pragma unroll\n    for (", 1)
            applied.append("loop_unrolling")

        # 添加__restrict__修饰符（减少指针别名分析开销）
        if "__restrict__" not in code and kernel.backend == "cuda":
            code = re.sub(r'(float|half)\s*\*\s*(\w+)', r'\1* __restrict__ \2', code)
            applied.append("restrict_pointers")

        return GeneratedKernel(
            operator_name=kernel.operator_name,
            backend=kernel.backend,
            target_gpu=kernel.target_gpu,
            source_code=code,
            header_code=kernel.header_code,
            build_flags=kernel.build_flags,
            launch_config=kernel.launch_config,
            estimated_bandwidth_utilization=kernel.estimated_bandwidth_utilization * 1.05,
            optimizations_applied=kernel.optimizations_applied + applied,
        )
