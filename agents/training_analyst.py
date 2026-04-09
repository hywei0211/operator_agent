"""
Training Analyst Agent - 训练代码分析器
解析用户的训练脚本，提取所需算子依赖，生成执行计划
"""
import ast
import logging
import re
from dataclasses import dataclass, field
from typing import Optional

from agents.base_agent import BaseAgent, AgentContext, AgentResult, AgentStatus

logger = logging.getLogger(__name__)

# 训练代码中常见的算子关键词映射
OP_KEYWORD_MAP = {
    # 注意力
    r"flash.?attention|scaled_dot_product_attention|MultiheadAttention|self\.attn": "flash_attention",
    r"FlashAttention|flash_attn": "flash_attention",
    # 归一化
    r"RMSNorm|rms_norm|LlamaRMSNorm": "rmsnorm",
    r"LayerNorm|layer_norm": "layernorm",
    # 激活
    r"\bgelu\b|GELU|F\.gelu": "gelu",
    r"\bsilu\b|SiLU|F\.silu|swiglu": "silu",
    r"\brelu\b|ReLU|F\.relu": "relu",
    # MoE
    r"MixtureOfExperts|MoE|fused_moe|TopKGating": "fused_moe",
    # 矩阵
    r"torch\.matmul|torch\.bmm|F\.linear|nn\.Linear": "matmul",
    # 嵌入
    r"nn\.Embedding|embed_tokens|token_embed": "embedding",
    # 规约
    r"softmax|F\.softmax": "softmax",
    # 通信（分布式）
    r"all_reduce|AllReduce|dist\.all_reduce": "allreduce",
    r"all_gather|AllGather": "allgather",
}

# 常见模型架构及其使用的算子
ARCHITECTURE_OP_MAP = {
    "llama": ["flash_attention", "rmsnorm", "silu", "matmul", "embedding", "softmax"],
    "gpt2": ["flash_attention", "layernorm", "gelu", "matmul", "embedding"],
    "mistral": ["flash_attention", "rmsnorm", "silu", "matmul", "fused_moe"],
    "mixtral": ["flash_attention", "rmsnorm", "silu", "fused_moe", "matmul"],
    "bert": ["flash_attention", "layernorm", "gelu", "matmul"],
    "qwen": ["flash_attention", "rmsnorm", "silu", "matmul", "embedding"],
    "deepseek": ["flash_attention", "rmsnorm", "silu", "fused_moe", "matmul"],
    "transformer": ["flash_attention", "layernorm", "gelu", "matmul"],
}


@dataclass
class TrainingPlan:
    """训练执行计划"""
    # 算子依赖
    required_operators: list[str] = field(default_factory=list)
    critical_operators: list[str] = field(default_factory=list)   # 关键路径，优先生成
    optional_operators: list[str] = field(default_factory=list)   # 可 fallback

    # 模型信息
    model_architecture: str = "unknown"
    estimated_params: str = "unknown"
    uses_distributed: bool = False
    parallel_hints: list[str] = field(default_factory=list)

    # 训练配置
    batch_size: Optional[int] = None
    seq_length: Optional[int] = None
    hidden_size: Optional[int] = None
    num_layers: Optional[int] = None
    dtype: str = "float16"

    # 框架信息
    framework: str = "pytorch"
    uses_amp: bool = False
    uses_gradient_checkpointing: bool = False

    def all_operators(self) -> list[str]:
        seen = set()
        result = []
        for op in self.critical_operators + self.required_operators + self.optional_operators:
            if op not in seen:
                seen.add(op)
                result.append(op)
        return result


class TrainingAnalystAgent(BaseAgent):
    """
    训练代码分析 Agent

    职责：
    1. 静态分析训练代码，提取算子依赖
    2. 识别模型架构（LLaMA / GPT / Mistral 等）
    3. 提取训练超参数（batch_size、seq_len、hidden_size）
    4. 判断是否需要分布式训练
    5. 输出优先级排序的算子列表
    """

    def __init__(self, llm_client=None, config: dict = None):
        super().__init__("TrainingAnalystAgent", llm_client, config)

    def get_system_prompt(self) -> str:
        return """你是深度学习训练专家，能够分析PyTorch训练代码并提取关键信息。
请分析代码，提取：使用的算子、模型架构、超参数、分布式设置。
以JSON格式返回分析结果。"""

    async def run(self, context: AgentContext, **kwargs) -> AgentResult:
        self._start_timer()
        self.set_status(AgentStatus.RUNNING)

        training_code: str = kwargs.get("training_code", "")
        if not training_code:
            return self.failure_result("No training code provided")

        try:
            # 静态分析
            plan = self._static_analyze(training_code)

            # LLM 深度分析（补充静态分析遗漏的信息）
            if self.llm_client:
                plan = await self._llm_enhance(training_code, plan)

            context.add_artifact("training_plan", plan)
            logger.info(
                f"[TrainingAnalyst] Operators found: {plan.all_operators()}\n"
                f"  Architecture: {plan.model_architecture}\n"
                f"  Distributed: {plan.uses_distributed}"
            )

            return self.success_result(
                output=plan,
                metrics={
                    "operators_found": len(plan.all_operators()),
                    "architecture": plan.model_architecture,
                    "distributed": plan.uses_distributed,
                }
            )
        except Exception as e:
            self.set_status(AgentStatus.FAILED)
            return self.failure_result(str(e))

    def _static_analyze(self, code: str) -> TrainingPlan:
        """静态代码分析"""
        plan = TrainingPlan()

        # 1. 识别使用的算子
        found_ops = set()
        for pattern, op_name in OP_KEYWORD_MAP.items():
            if re.search(pattern, code, re.IGNORECASE):
                found_ops.add(op_name)

        # 2. 识别模型架构
        arch = self._detect_architecture(code)
        plan.model_architecture = arch
        if arch in ARCHITECTURE_OP_MAP:
            for op in ARCHITECTURE_OP_MAP[arch]:
                found_ops.add(op)

        # 3. 划分关键/可选算子
        critical = {"flash_attention", "matmul", "rmsnorm", "layernorm"}
        plan.critical_operators = [op for op in found_ops if op in critical]
        plan.required_operators = [op for op in found_ops if op not in critical]

        # 4. 提取超参数
        plan.batch_size = self._extract_int(code, r"batch_size\s*=\s*(\d+)")
        plan.seq_length = self._extract_int(code, r"(?:seq_len|max_seq_len|sequence_length)\s*=\s*(\d+)")
        plan.hidden_size = self._extract_int(code, r"hidden_(?:size|dim)\s*=\s*(\d+)")
        plan.num_layers = self._extract_int(code, r"num_(?:hidden_)?layers\s*=\s*(\d+)")
        plan.dtype = "bfloat16" if "bfloat16" in code else "float16"

        # 5. 检测分布式
        dist_patterns = [
            r"torch\.distributed", r"DistributedDataParallel", r"DDP",
            r"deepspeed", r"megatron", r"dist\.init_process_group"
        ]
        plan.uses_distributed = any(re.search(p, code) for p in dist_patterns)

        # 6. 检测 AMP 和梯度检查点
        plan.uses_amp = bool(re.search(r"autocast|GradScaler|amp\.", code))
        plan.uses_gradient_checkpointing = bool(re.search(r"gradient_checkpointing|checkpoint_sequential", code))

        return plan

    def _detect_architecture(self, code: str) -> str:
        code_lower = code.lower()
        arch_keywords = {
            "llama": ["llama", "llamaforsequenceclassification", "llamamodel"],
            "mistral": ["mistral"],
            "mixtral": ["mixtral", "mixtralfor"],
            "gpt2": ["gpt2", "gpt-2"],
            "bert": ["bert", "bertmodel"],
            "qwen": ["qwen"],
            "deepseek": ["deepseek"],
            "transformer": ["transformer", "transformermodel"],
        }
        for arch, keywords in arch_keywords.items():
            if any(kw in code_lower for kw in keywords):
                return arch
        return "unknown"

    def _extract_int(self, code: str, pattern: str) -> Optional[int]:
        match = re.search(pattern, code, re.IGNORECASE)
        if match:
            try:
                return int(match.group(1))
            except ValueError:
                pass
        return None

    async def _llm_enhance(self, code: str, plan: TrainingPlan) -> TrainingPlan:
        """用 LLM 补充静态分析未能识别的信息"""
        prompt = f"""分析以下 PyTorch 训练代码（前 2000 字符），补充这些信息：
1. 还有哪些算子被使用（我已找到：{plan.all_operators()}）？
2. 模型参数量大概多少？
3. 并行策略建议？

代码:
```python
{code[:2000]}
```

以JSON返回: {{"additional_operators": [], "estimated_params": "7B", "parallel_hints": []}}"""

        try:
            resp = await self.call_llm(prompt, temperature=0.0, max_tokens=500)
            import json
            json_match = re.search(r'\{.*\}', resp, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                for op in data.get("additional_operators", []):
                    if op not in plan.required_operators and op not in plan.critical_operators:
                        plan.optional_operators.append(op)
                plan.estimated_params = data.get("estimated_params", plan.estimated_params)
                plan.parallel_hints = data.get("parallel_hints", [])
        except Exception as e:
            logger.warning(f"[TrainingAnalyst] LLM enhancement failed: {e}")

        return plan
