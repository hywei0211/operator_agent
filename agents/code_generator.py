"""
Code Generator Agent - 代码生成器
为不同GPU后端生成算子内核代码，支持CUDA/HIP/SYCL/Triton
"""
import json
import logging
import re
from typing import Optional

from agents.base_agent import BaseAgent, AgentContext, AgentResult, AgentStatus
from models.operator_ir import OperatorIR, GeneratedKernel
from models.hardware_model import GPUSpec, GPUBackend, GPUVendor
from prompts.code_gen_prompts import (
    build_cuda_codegen_prompt,
    build_hip_codegen_prompt,
    build_triton_codegen_prompt,
    build_ascendc_codegen_prompt,
)

logger = logging.getLogger(__name__)


class CodeGenAgent(BaseAgent):
    """
    代码生成Agent

    职责：
    1. 根据算子IR和目标GPU规格选择合适的后端
    2. 构造精确的LLM代码生成提示词
    3. 解析LLM返回的代码
    4. 进行基本的代码合法性检查
    5. 返回可编译的GeneratedKernel

    支持的后端：CUDA / HIP / SYCL / Triton
    """

    # 后端选择策略：优先级顺序
    BACKEND_PRIORITY = {
        GPUVendor.NVIDIA: [GPUBackend.CUDA, GPUBackend.TRITON],
        GPUVendor.AMD: [GPUBackend.HIP, GPUBackend.TRITON],
        GPUVendor.INTEL: [GPUBackend.SYCL, GPUBackend.TRITON],
        GPUVendor.APPLE: [GPUBackend.METAL, GPUBackend.TRITON],
        GPUVendor.HUAWEI: [GPUBackend.ASCENDC],
    }

    def __init__(self, llm_client=None, config: dict = None):
        super().__init__("CodeGenAgent", llm_client, config)
        self._use_triton_for_cross_platform = config.get("prefer_triton", False) if config else False

    def get_system_prompt(self) -> str:
        return """你是一位专业的GPU内核工程师，精通CUDA、HIP、SYCL和Triton。
你能够根据算子的数学定义和目标硬件规格，生成高性能的GPU内核代码。
你深刻理解GPU的内存层次、并行计算模型和硬件特性。
生成的代码必须：正确、高效、可编译、有清晰注释。"""

    async def run(self, context: AgentContext, **kwargs) -> AgentResult:
        self._start_timer()
        self.set_status(AgentStatus.RUNNING)

        operator_ir: Optional[OperatorIR] = kwargs.get("operator_ir")
        gpu_spec: Optional[GPUSpec] = kwargs.get("gpu_spec")

        if operator_ir is None:
            operator_ir = context.get_artifact("operator_ir")
        if operator_ir is None:
            return self.failure_result("No OperatorIR provided")
        if gpu_spec is None:
            return self.failure_result("No GPUSpec provided")

        # 提取修复上下文（ReviewLoop 迭代修复时传入）
        fix_context = context.get_artifact("fix_context")

        try:
            # 选择最优后端
            backend = self._select_backend(gpu_spec)
            logger.info(f"[CodeGen] Generating {backend.value} kernel for {gpu_spec.model_name}")

            # 调用对应后端的生成方法
            kernel = await self._generate_kernel(operator_ir, gpu_spec, backend, fix_context)

            logger.info(f"[CodeGen] Generated kernel: {len(kernel.source_code)} chars, "
                       f"backend={backend.value}, optimizations={kernel.optimizations_applied}")

            return self.success_result(
                output=kernel,
                metrics={
                    "backend": backend.value,
                    "code_length": len(kernel.source_code),
                    "estimated_efficiency": kernel.estimated_bandwidth_utilization,
                }
            )

        except Exception as e:
            self.set_status(AgentStatus.FAILED)
            logger.exception(f"[CodeGen] Failed: {e}")
            return self.failure_result(str(e))

    def _select_backend(self, gpu_spec: GPUSpec) -> GPUBackend:
        """根据GPU厂商选择最优后端"""
        if self._use_triton_for_cross_platform and gpu_spec.supports_triton():
            return GPUBackend.TRITON

        priority_list = self.BACKEND_PRIORITY.get(gpu_spec.vendor, [GPUBackend.OPENCL])
        for backend in priority_list:
            if backend in gpu_spec.supported_backends:
                return backend

        # 退化到Triton或OpenCL
        if GPUBackend.TRITON in gpu_spec.supported_backends:
            return GPUBackend.TRITON
        return GPUBackend.OPENCL

    async def _generate_kernel(
        self,
        op_ir: OperatorIR,
        gpu_spec: GPUSpec,
        backend: GPUBackend,
        fix_context: dict = None,
    ) -> GeneratedKernel:
        """根据后端生成内核代码"""
        if backend == GPUBackend.CUDA:
            return await self._generate_cuda(op_ir, gpu_spec, fix_context)
        elif backend == GPUBackend.HIP:
            return await self._generate_hip(op_ir, gpu_spec)
        elif backend == GPUBackend.TRITON:
            return await self._generate_triton(op_ir, gpu_spec)
        elif backend == GPUBackend.SYCL:
            return await self._generate_sycl(op_ir, gpu_spec)
        elif backend == GPUBackend.ASCENDC:
            return await self._generate_ascendc(op_ir, gpu_spec)
        else:
            raise NotImplementedError(f"Backend {backend.value} not yet supported")

    async def _generate_cuda(self, op_ir: OperatorIR, gpu_spec: GPUSpec, fix_context: dict = None) -> GeneratedKernel:
        """生成CUDA内核（支持自动降级到简单版本）"""
        if self.llm_client:
            prompt = build_cuda_codegen_prompt(op_ir, gpu_spec)
            # 如果是 ReviewLoop 修复模式，附加历史上下文
            if fix_context:
                history = fix_context.get("history_summary", "")
                guidance = fix_context.get("fix_guidance", "")
                if history:
                    prompt += f"\n\n--- PREVIOUS ATTEMPTS (避免重复错误) ---\n{history}\n"
                if guidance:
                    prompt += f"\n当前需修复的问题：\n{guidance}\n"
                # 如果已经失败多次，自动降级到简单版本
                num_attempts = len(fix_context.get("iteration_history", []))
                if num_attempts >= 2:
                    logger.info(f"[CodeGen] Auto-downgrading to simple mode for {op_ir.name} after {num_attempts} failures")
                    from prompts.code_gen_prompts import build_cuda_simple_prompt
                    prompt = build_cuda_simple_prompt(op_ir, gpu_spec)
            response = await self.call_llm(prompt, max_tokens=8192)
            logger.debug(f"[CodeGen] raw response for {op_ir.name} (first 200): {response[:200]!r}")
            return self._parse_kernel_response(response, op_ir.name, "cuda", gpu_spec.model_name)
        else:
            return self._generate_cuda_template(op_ir, gpu_spec)

    async def _generate_hip(self, op_ir: OperatorIR, gpu_spec: GPUSpec) -> GeneratedKernel:
        """生成HIP内核"""
        if self.llm_client:
            prompt = build_hip_codegen_prompt(op_ir, gpu_spec)
            response = await self.call_llm(prompt, max_tokens=8192)
            return self._parse_kernel_response(response, op_ir.name, "hip", gpu_spec.model_name)
        else:
            return self._generate_hip_template(op_ir, gpu_spec)

    async def _generate_triton(self, op_ir: OperatorIR, gpu_spec: GPUSpec) -> GeneratedKernel:
        """生成Triton内核（跨硬件）"""
        if self.llm_client:
            prompt = build_triton_codegen_prompt(op_ir, [gpu_spec])
            response = await self.call_llm(prompt, max_tokens=8192)
            return self._parse_triton_response(response, op_ir.name, gpu_spec.model_name)
        else:
            return self._generate_triton_template(op_ir, gpu_spec)

    async def _generate_sycl(self, op_ir: OperatorIR, gpu_spec: GPUSpec) -> GeneratedKernel:
        """生成SYCL内核（Intel GPU）"""
        prompt = f"""为Intel GPU生成SYCL内核实现以下算子：

算子: {op_ir.name}
数学定义: {op_ir.math_description}
目标GPU: {gpu_spec.model_name}

要求：使用SYCL 2020标准，利用nd_range并行，使用local_accessor实现局部内存共享。

返回JSON格式：{{"kernel_code": "...", "build_flags": [...], "optimizations": [...]}}"""

        if self.llm_client:
            response = await self.call_llm(prompt, max_tokens=6144)
            return self._parse_kernel_response(response, op_ir.name, "sycl", gpu_spec.model_name)
        else:
            return GeneratedKernel(
                operator_name=op_ir.name,
                backend="sycl",
                target_gpu=gpu_spec.model_name,
                source_code=f"// SYCL kernel placeholder for {op_ir.name}\n// TODO: implement",
                build_flags=["-fsycl", "-O3"],
            )

    async def _generate_ascendc(self, op_ir: OperatorIR, gpu_spec: GPUSpec) -> GeneratedKernel:
        """生成 AscendC 内核（华为昇腾）"""
        if self.llm_client:
            prompt = build_ascendc_codegen_prompt(op_ir, gpu_spec)
            response = await self.call_llm(prompt, max_tokens=8192)
            return self._parse_kernel_response(response, op_ir.name, "ascendc", gpu_spec.model_name)
        else:
            return self._generate_ascendc_template(op_ir, gpu_spec)

    # 合法的 CUDA/HIP nvcc/hipcc build flags 白名单前缀
    _VALID_FLAG_PREFIXES = (
        "-O", "-arch=", "-gencode", "--use_fast_math", "-std=", "-I",
        "-D", "-G", "-lineinfo", "-Xcompiler", "--offload-arch=",
        "-fPIC", "--shared", "-o", "-c", "-rdc=",
    )

    def _sanitize_build_flags(self, flags: list[str], backend: str) -> list[str]:
        """过滤掉 LLM 可能生成的非法 build flags"""
        result = []
        for f in flags:
            if any(f.startswith(p) for p in self._VALID_FLAG_PREFIXES):
                result.append(f)
            else:
                logger.warning(f"[CodeGen] Dropping invalid build flag: {f!r}")
        return result

    @staticmethod
    def _strip_markdown_fences(text: str) -> str:
        """剥离 LLM 可能在代码值里插入的 markdown 代码块标记"""
        text = re.sub(r'^```[a-zA-Z+]*\n?', '', text.strip())
        text = re.sub(r'\n?```$', '', text.strip())
        return text.strip()

    def _parse_kernel_response(
        self,
        response: str,
        op_name: str,
        backend: str,
        gpu_name: str
    ) -> GeneratedKernel:
        """解析LLM返回的内核代码"""
        # 尝试直接解析 JSON（LLM 有时把整个 JSON 包在 ```json ... ``` 里，先剥离）
        for candidate in [response, self._strip_markdown_fences(response)]:
            json_match = re.search(r'\{.*\}', candidate, re.DOTALL)
            if not json_match:
                continue
            try:
                data = json.loads(json_match.group())
                kernel_code = data.get("kernel_code", "")
                # 剥离 kernel_code 值里可能的 markdown 代码块标记
                kernel_code = self._strip_markdown_fences(kernel_code)
                if kernel_code:
                    return GeneratedKernel(
                        operator_name=op_name,
                        backend=backend,
                        target_gpu=gpu_name,
                        source_code=kernel_code,
                        header_code=data.get("header_code", ""),
                        build_flags=self._sanitize_build_flags(data.get("build_flags", []), backend),
                        launch_config=data.get("launch_config", {}),
                        estimated_bandwidth_utilization=data.get("estimated_efficiency", 0.0),
                        optimizations_applied=data.get("optimizations", []),
                    )
            except json.JSONDecodeError:
                continue

        # JSON解析失败，直接提取代码块（优先匹配 cuda/cpp/c++ fence，再匹配任意 fence）
        code_match = (
            re.search(r'```(?:cuda|cpp|c\+\+)\n(.*?)```', response, re.DOTALL) or
            re.search(r'```[a-zA-Z]*\n(.*?)```', response, re.DOTALL)
        )
        if code_match:
            code = code_match.group(1).strip()
        else:
            # 没有代码块 — 可能是裸 JSON 或纯代码
            code = None

            # 尝试从 response 中提取 kernel_code（即使 JSON 整体无法解析）
            # 方案1: 正则提取 "kernel_code": "..." 字段（处理转义的 JSON 字符串）
            kc_match = re.search(
                r'"kernel_code"\s*:\s*"((?:[^"\\]|\\.)*)"',
                response, re.DOTALL
            )
            if kc_match:
                raw = kc_match.group(1)
                code = (raw.replace('\\n', '\n')
                           .replace('\\"', '"')
                           .replace('\\\\', '\\')
                           .replace('\\t', '\t'))
                code = self._strip_markdown_fences(code)

            # 方案2: 用 json.loads 解析裸 JSON
            if not code:
                try:
                    json_start = response.find('{')
                    json_end = response.rfind('}')
                    if json_start >= 0 and json_end > json_start:
                        data = json.loads(response[json_start:json_end+1])
                        kc = data.get("kernel_code", "")
                        if kc:
                            code = self._strip_markdown_fences(kc)
                except (json.JSONDecodeError, Exception):
                    pass

            # 方案3: 找第一个 #include 开始的内容（可能嵌在 JSON 值里的转义文本）
            if not code:
                # 从 response 中查找 #include 开头（可能是转义的 \\n#include 或 \n#include）
                raw_code_match = re.search(r'(#include\s*<[^>]+>.*)', response, re.DOTALL)
                if raw_code_match:
                    code = raw_code_match.group(1)
                    # 清理可能的 JSON 结尾 (如 末尾的 ", ... })
                    code = re.sub(r'",\s*"(header_code|build_flags|launch_config|optimizations|estimated_efficiency)".*$', '', code, flags=re.DOTALL)

            if not code:
                code = response.strip()
                logger.error(f"[CodeGen] Could not extract kernel code from response for {op_name}, using raw response")

        # 最终安全检查：如果 code 以 { 开头，说明仍是 JSON 而非 C/CUDA 代码
        # 做最后一次尝试提取
        if code.lstrip().startswith('{') and '"kernel_code"' in code:
            kc_last = re.search(r'"kernel_code"\s*:\s*"((?:[^"\\]|\\.)*)"', code, re.DOTALL)
            if kc_last:
                raw = kc_last.group(1)
                code = (raw.replace('\\n', '\n').replace('\\"', '"')
                           .replace('\\\\', '\\').replace('\\t', '\t'))
                logger.info(f"[CodeGen] Recovered kernel_code from raw JSON for {op_name} ({len(code)} chars)")

        logger.warning(f"[CodeGen] JSON parse failed for {op_name}, using code block fallback ({len(code)} chars)")

        return GeneratedKernel(
            operator_name=op_name,
            backend=backend,
            target_gpu=gpu_name,
            source_code=code,
            build_flags=["-O3"],
        )

    def _parse_triton_response(self, response: str, op_name: str, gpu_name: str) -> GeneratedKernel:
        """解析Triton内核响应"""
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group())
                full_code = data.get("kernel_code", "")
                wrapper = data.get("python_wrapper", "")
                if wrapper:
                    full_code = full_code + "\n\n" + wrapper

                return GeneratedKernel(
                    operator_name=op_name,
                    backend="triton",
                    target_gpu=gpu_name,
                    source_code=full_code,
                    build_flags=[],
                    launch_config={"autotune_configs": data.get("autotune_configs", [])},
                    estimated_bandwidth_utilization=data.get("estimated_efficiency", 0.0),
                    optimizations_applied=data.get("optimizations", []),
                )
            except json.JSONDecodeError:
                pass

        code_match = re.search(r'```python\n(.*?)```', response, re.DOTALL)
        code = code_match.group(1) if code_match else response

        return GeneratedKernel(
            operator_name=op_name,
            backend="triton",
            target_gpu=gpu_name,
            source_code=code,
        )

    # ---- 无LLM时的模板兜底实现 ----

    def _generate_cuda_template(self, op_ir: OperatorIR, gpu_spec: GPUSpec) -> GeneratedKernel:
        """无LLM时生成CUDA模板代码"""
        sm = gpu_spec.compute_capability.replace('.', '') if gpu_spec.compute_capability else "80"
        code = f"""// Auto-generated CUDA kernel for {op_ir.name}
// Target: {gpu_spec.model_name} (SM {gpu_spec.compute_capability})
// Math: {op_ir.math_description}

#include <cuda_fp16.h>
#include <cuda_runtime.h>

// TODO: Implement {op_ir.name} kernel
__global__ void {op_ir.name}_kernel(
    // inputs: {', '.join(t.name for t in op_ir.inputs)}
    // outputs: {', '.join(t.name for t in op_ir.outputs)}
    int N
) {{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {{
        // Implement: {op_ir.math_description}
    }}
}}

void launch_{op_ir.name}(/* params */, cudaStream_t stream = nullptr) {{
    dim3 block(256);
    dim3 grid((N + 255) / 256);
    {op_ir.name}_kernel<<<grid, block, 0, stream>>>(/* args */);
}}
"""
        return GeneratedKernel(
            operator_name=op_ir.name,
            backend="cuda",
            target_gpu=gpu_spec.model_name,
            source_code=code,
            build_flags=[f"-arch=sm_{sm}", "-O3", "-use_fast_math"],
            launch_config={"block_size": 256},
        )

    def _generate_hip_template(self, op_ir: OperatorIR, gpu_spec: GPUSpec) -> GeneratedKernel:
        code = f"""// Auto-generated HIP kernel for {op_ir.name}
// Target: {gpu_spec.model_name} (CDNA/RDNA)

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

__global__ void {op_ir.name}_kernel(int N) {{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {{
        // Implement: {op_ir.math_description}
    }}
}}

void launch_{op_ir.name}(int N, hipStream_t stream = nullptr) {{
    dim3 block(256);
    dim3 grid((N + 255) / 256);
    hipLaunchKernelGGL({op_ir.name}_kernel, grid, block, 0, stream, N);
}}
"""
        return GeneratedKernel(
            operator_name=op_ir.name,
            backend="hip",
            target_gpu=gpu_spec.model_name,
            source_code=code,
            build_flags=["--offload-arch=gfx942", "-O3"],
            launch_config={"block_size": 256},
        )

    def _generate_triton_template(self, op_ir: OperatorIR, gpu_spec: GPUSpec) -> GeneratedKernel:
        code = f"""# Auto-generated Triton kernel for {op_ir.name}
import triton
import triton.language as tl
import torch

@triton.autotune(
    configs=[
        triton.Config({{'BLOCK_SIZE': 128}}, num_warps=4),
        triton.Config({{'BLOCK_SIZE': 256}}, num_warps=8),
    ],
    key=['N'],
)
@triton.jit
def {op_ir.name}_kernel(
    # TODO: add tensor pointers
    N,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    # Implement: {op_ir.math_description}

def {op_ir.name}(/* inputs */):
    N = ...  # total elements
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)
    {op_ir.name}_kernel[grid](N)
"""
        return GeneratedKernel(
            operator_name=op_ir.name,
            backend="triton",
            target_gpu=gpu_spec.model_name,
            source_code=code,
        )

    def _generate_ascendc_template(self, op_ir: OperatorIR, gpu_spec: GPUSpec) -> GeneratedKernel:
        """无LLM时生成 AscendC 模板代码"""
        code = f"""// Auto-generated AscendC kernel for {op_ir.name}
// Target: {gpu_spec.model_name} ({gpu_spec.architecture})
// Math: {op_ir.math_description}

#include "kernel_operator.h"
using namespace AscendC;

constexpr int32_t TILE_NUM = 8;

class {op_ir.name.capitalize()}Kernel {{
public:
    __aicore__ inline {op_ir.name.capitalize()}Kernel() {{}}

    __aicore__ inline void Init(
        GM_ADDR input, GM_ADDR output, uint32_t totalLength, uint32_t tileLength)
    {{
        this->tileLength = tileLength;
        this->tileNum = totalLength / tileLength;

        inputGm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(input), totalLength);
        outputGm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(output), totalLength);

        pipe.InitBuffer(inQueue, TILE_NUM, tileLength * sizeof(half));
        pipe.InitBuffer(outQueue, TILE_NUM, tileLength * sizeof(half));
    }}

    __aicore__ inline void Process()
    {{
        int32_t loopCount = this->tileNum;
        for (int32_t i = 0; i < loopCount; i++) {{
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }}
    }}

private:
    __aicore__ inline void CopyIn(int32_t progress)
    {{
        LocalTensor<half> inLocal = inQueue.AllocTensor<half>();
        DataCopy(inLocal, inputGm[progress * tileLength], tileLength);
        inQueue.EnQue(inLocal);
    }}

    __aicore__ inline void Compute(int32_t progress)
    {{
        LocalTensor<half> inLocal = inQueue.DeQue<half>();
        LocalTensor<half> outLocal = outQueue.AllocTensor<half>();
        // TODO: implement {op_ir.math_description}
        // Example: Abs(outLocal, inLocal, tileLength);
        outQueue.EnQue(outLocal);
        inQueue.FreeTensor(inLocal);
    }}

    __aicore__ inline void CopyOut(int32_t progress)
    {{
        LocalTensor<half> outLocal = outQueue.DeQue<half>();
        DataCopy(outputGm[progress * tileLength], outLocal, tileLength);
        outQueue.FreeTensor(outLocal);
    }}

private:
    TPipe pipe;
    TBuf<QuePosition::VECIN>  inQueue;
    TBuf<QuePosition::VECOUT> outQueue;
    GlobalTensor<half> inputGm, outputGm;
    uint32_t tileLength;
    uint32_t tileNum;
}};

extern "C" __global__ __aicore__ void {op_ir.name}_kernel(
    GM_ADDR input, GM_ADDR output, uint32_t totalLength, uint32_t tileLength)
{{
    {op_ir.name.capitalize()}Kernel op;
    op.Init(input, output, totalLength, tileLength);
    op.Process();
}}
"""
        return GeneratedKernel(
            operator_name=op_ir.name,
            backend="ascendc",
            target_gpu=gpu_spec.model_name,
            source_code=code,
            build_flags=["--target=ascend910b", "-O2", "-std=c++17"],
            launch_config={
                "aiv_num": gpu_spec.compute_units,
                "tile_length": 21760,
                "double_buffer": True,
            },
        )
