"""
Operator Spec Analyzer Agent - 算子规格解析器
将用户的自然语言算子描述转换为标准化的OperatorIR
"""
import json
import logging
import re
from typing import Any

from agents.base_agent import BaseAgent, AgentContext, AgentResult, AgentStatus
from models.operator_ir import (
    OperatorIR, TensorSpec, DataType, OperatorCategory,
    OperatorConstraints, ParallelStrategy
)

logger = logging.getLogger(__name__)

# 内置算子模板库（常见大模型算子）
OPERATOR_TEMPLATES: dict[str, dict[str, Any]] = {
    "flash_attention": {
        "category": OperatorCategory.ATTENTION,
        "description": "Memory-efficient attention with online softmax",
        "inputs": [
            {"name": "Q", "shape": ["batch", "num_heads", "seq_len", "head_dim"], "dtype": DataType.FP16},
            {"name": "K", "shape": ["batch", "num_heads", "seq_len", "head_dim"], "dtype": DataType.FP16},
            {"name": "V", "shape": ["batch", "num_heads", "seq_len", "head_dim"], "dtype": DataType.FP16},
        ],
        "outputs": [
            {"name": "O", "shape": ["batch", "num_heads", "seq_len", "head_dim"], "dtype": DataType.FP16},
        ],
        "math_description": "O = softmax(QK^T / sqrt(d)) * V",
        "flops_formula": "4 * batch * num_heads * seq_len * seq_len * head_dim",
        "memory_reads_formula": "3 * batch * num_heads * seq_len * head_dim",
        "memory_writes_formula": "batch * num_heads * seq_len * head_dim",
        "reference_impl": """
import torch
import torch.nn.functional as F
def flash_attention_ref(Q, K, V, scale=None):
    if scale is None:
        scale = Q.shape[-1] ** -0.5
    attn = torch.matmul(Q, K.transpose(-2, -1)) * scale
    attn = F.softmax(attn, dim=-1)
    return torch.matmul(attn, V)
""",
        "tags": ["transformer", "attention", "memory_efficient"],
    },
    "rmsnorm": {
        "category": OperatorCategory.NORMALIZATION,
        "description": "Root Mean Square Layer Normalization",
        "inputs": [
            {"name": "x", "shape": ["batch", "seq_len", "hidden"], "dtype": DataType.FP16},
            {"name": "weight", "shape": ["hidden"], "dtype": DataType.FP16},
        ],
        "outputs": [
            {"name": "y", "shape": ["batch", "seq_len", "hidden"], "dtype": DataType.FP16},
        ],
        "math_description": "y = x / RMS(x) * weight, where RMS(x) = sqrt(mean(x^2) + eps)",
        "flops_formula": "5 * batch * seq_len * hidden",
        "memory_reads_formula": "2 * batch * seq_len * hidden",
        "memory_writes_formula": "batch * seq_len * hidden",
        "reference_impl": """
import torch
def rmsnorm_ref(x, weight, eps=1e-6):
    rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + eps)
    return x / rms * weight
""",
        "tags": ["normalization", "llama", "transformer"],
    },
    "gelu": {
        "category": OperatorCategory.ELEMENTWISE,
        "description": "Gaussian Error Linear Unit activation",
        "inputs": [
            {"name": "x", "shape": ["batch", "seq_len", "hidden"], "dtype": DataType.FP16},
        ],
        "outputs": [
            {"name": "y", "shape": ["batch", "seq_len", "hidden"], "dtype": DataType.FP16},
        ],
        "math_description": "y = x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))",
        "flops_formula": "8 * batch * seq_len * hidden",
        "memory_reads_formula": "batch * seq_len * hidden",
        "memory_writes_formula": "batch * seq_len * hidden",
        "reference_impl": """
import torch
def gelu_ref(x):
    return torch.nn.functional.gelu(x)
""",
        "tags": ["activation", "gpt", "bert"],
    },
    "fused_moe": {
        "category": OperatorCategory.FUSED,
        "description": "Fused Mixture of Experts with top-k routing",
        "inputs": [
            {"name": "hidden_states", "shape": ["batch", "seq_len", "hidden"], "dtype": DataType.FP16},
            {"name": "router_logits", "shape": ["batch", "seq_len", "num_experts"], "dtype": DataType.FP32},
            {"name": "w1", "shape": ["num_experts", "intermediate", "hidden"], "dtype": DataType.FP16},
            {"name": "w2", "shape": ["num_experts", "hidden", "intermediate"], "dtype": DataType.FP16},
        ],
        "outputs": [
            {"name": "output", "shape": ["batch", "seq_len", "hidden"], "dtype": DataType.FP16},
        ],
        "math_description": "MoE with top-k expert selection, gating and expert computation",
        "flops_formula": "2 * top_k * batch * seq_len * hidden * intermediate",
        "memory_reads_formula": "top_k * batch * seq_len * hidden + num_experts * intermediate * hidden",
        "memory_writes_formula": "batch * seq_len * hidden",
        "tags": ["moe", "mixtral", "expert_parallel"],
    },
}


class OperatorSpecAgent(BaseAgent):
    """
    算子规格解析Agent

    职责：
    1. 解析用户的自然语言算子描述
    2. 匹配内置算子模板库
    3. 用LLM提取算子数学语义和计算复杂度
    4. 生成标准化的OperatorIR
    5. 分析并行化可能性
    """

    def __init__(self, llm_client=None, config: dict = None):
        super().__init__("OperatorSpecAgent", llm_client, config)

    def get_system_prompt(self) -> str:
        template_list = "\n".join(f"  - {k}" for k in OPERATOR_TEMPLATES)
        return f"""你是一个GPU算子专家，专注于将算子描述转化为精确的规格定义。

内置算子模板：
{template_list}

你的任务是：
1. 理解用户描述的算子计算语义
2. 如果是已知算子，匹配对应模板
3. 如果是新算子，提取其数学定义、输入输出张量规格、FLOPs复杂度
4. 判断是否可以并行化，以及适合的并行策略
5. 生成严格的算子规格JSON

规格JSON格式：
{{
  "name": "算子名称",
  "category": "matmul|attention|normalization|elementwise|reduction|fused",
  "description": "描述",
  "inputs": [{{"name": "...", "shape": [...], "dtype": "fp16|bf16|fp32"}}],
  "outputs": [{{"name": "...", "shape": [...], "dtype": "..."}}],
  "math_description": "数学公式",
  "flops_formula": "FLOPs计算公式（用Python表达式）",
  "memory_reads_formula": "内存读取量公式",
  "memory_writes_formula": "内存写入量公式",
  "parallel_strategy": "none|tensor_parallel|data_parallel|sequence",
  "reference_impl": "PyTorch参考实现代码",
  "tags": [...]
}}"""

    async def run(self, context: AgentContext, **kwargs) -> AgentResult:
        self._start_timer()
        self.set_status(AgentStatus.RUNNING)

        request: str = kwargs.get("request", "")
        hardware_profiles: dict = kwargs.get("hardware_profiles", {})

        if not request:
            return self.failure_result("No operator request provided")

        try:
            # 1. 首先尝试匹配模板
            template_match = self._match_template(request)
            if template_match:
                logger.info(f"[SpecAnalyzer] Matched template: {template_match}")
                operator_ir = self._build_from_template(template_match, request)
            elif self.llm_client:
                # 2. 使用LLM解析
                operator_ir = await self._parse_with_llm(request, hardware_profiles)
            else:
                return self.failure_result("Cannot parse operator: no template match and no LLM client")

            # 3. 验证IR完整性
            issues = self._validate_ir(operator_ir)
            if issues:
                logger.warning(f"[SpecAnalyzer] IR validation issues: {issues}")

            context.add_artifact("operator_ir", operator_ir)
            logger.info(f"[SpecAnalyzer] Generated OperatorIR: {operator_ir.name} ({operator_ir.category.value})")

            return self.success_result(
                output=operator_ir,
                metrics={
                    "template_matched": template_match is not None,
                    "num_inputs": len(operator_ir.inputs),
                    "num_outputs": len(operator_ir.outputs),
                }
            )

        except Exception as e:
            self.set_status(AgentStatus.FAILED)
            return self.failure_result(str(e))

    def _match_template(self, request: str) -> str | None:
        """基于关键词匹配内置模板"""
        request_lower = request.lower()
        keyword_to_template = {
            "flash_attention": "flash_attention",
            "flashattention": "flash_attention",
            "attention": "flash_attention",
            "rmsnorm": "rmsnorm",
            "rms norm": "rmsnorm",
            "layer norm": "rmsnorm",
            "gelu": "gelu",
            "moe": "fused_moe",
            "mixture of experts": "fused_moe",
        }
        for keyword, template in keyword_to_template.items():
            if keyword in request_lower:
                return template
        return None

    def _build_from_template(self, template_name: str, request: str) -> OperatorIR:
        """从模板构建OperatorIR"""
        template = OPERATOR_TEMPLATES[template_name]

        inputs = [
            TensorSpec(
                name=t["name"],
                shape=t["shape"],
                dtype=t["dtype"],
                is_input=True,
            )
            for t in template["inputs"]
        ]
        outputs = [
            TensorSpec(
                name=t["name"],
                shape=t["shape"],
                dtype=t["dtype"],
                is_input=False,
            )
            for t in template["outputs"]
        ]

        return OperatorIR(
            name=template_name,
            category=template["category"],
            description=template["description"],
            inputs=inputs,
            outputs=outputs,
            math_description=template.get("math_description", ""),
            reference_impl=template.get("reference_impl", ""),
            flops_formula=template.get("flops_formula", ""),
            memory_reads_formula=template.get("memory_reads_formula", ""),
            memory_writes_formula=template.get("memory_writes_formula", ""),
            tags=template.get("tags", []),
        )

    async def _parse_with_llm(self, request: str, hardware_profiles: dict) -> OperatorIR:
        """用LLM解析算子描述"""
        hw_context = ""
        if hardware_profiles:
            hw_context = f"\n目标硬件: {', '.join(hardware_profiles.keys())}"

        prompt = f"""请分析以下算子需求，生成完整的算子规格JSON：

算子需求：
{request}
{hw_context}

请返回严格的JSON格式（不要包含其他内容）："""

        response = await self.call_llm(prompt)

        # 提取JSON
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if not json_match:
            raise ValueError(f"LLM did not return valid JSON: {response[:200]}")

        spec_dict = json.loads(json_match.group())
        return self._dict_to_ir(spec_dict)

    def _dict_to_ir(self, spec: dict) -> OperatorIR:
        """将JSON字典转换为OperatorIR"""
        dtype_map = {d.value: d for d in DataType}
        category_map = {c.value: c for c in OperatorCategory}
        parallel_map = {p.value: p for p in ParallelStrategy}

        inputs = [
            TensorSpec(
                name=t["name"],
                shape=t["shape"],
                dtype=dtype_map.get(t.get("dtype", "fp16"), DataType.FP16),
                is_input=True,
            )
            for t in spec.get("inputs", [])
        ]
        outputs = [
            TensorSpec(
                name=t["name"],
                shape=t["shape"],
                dtype=dtype_map.get(t.get("dtype", "fp16"), DataType.FP16),
                is_input=False,
            )
            for t in spec.get("outputs", [])
        ]

        return OperatorIR(
            name=spec.get("name", "custom_operator"),
            category=category_map.get(spec.get("category", "elementwise"), OperatorCategory.ELEMENTWISE),
            description=spec.get("description", ""),
            inputs=inputs,
            outputs=outputs,
            math_description=spec.get("math_description", ""),
            reference_impl=spec.get("reference_impl", ""),
            flops_formula=spec.get("flops_formula", ""),
            memory_reads_formula=spec.get("memory_reads_formula", ""),
            memory_writes_formula=spec.get("memory_writes_formula", ""),
            parallel_strategy=parallel_map.get(spec.get("parallel_strategy", "none"), ParallelStrategy.NONE),
            tags=spec.get("tags", []),
        )

    def _validate_ir(self, ir: OperatorIR) -> list[str]:
        """验证OperatorIR的完整性"""
        issues = []
        if not ir.name:
            issues.append("Missing operator name")
        if not ir.inputs:
            issues.append("No input tensors defined")
        if not ir.outputs:
            issues.append("No output tensors defined")
        if not ir.math_description:
            issues.append("Missing math description")
        if not ir.flops_formula:
            issues.append("Missing FLOPs formula")
        return issues
