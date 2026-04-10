"""
Intent Parser Agent — LLM 驱动的意图解析器

将用户的自然语言输入解析为结构化的算子生成请求。
支持：
- 任意算子名称（不限于预定义列表）
- 模糊输入识别
- 缺失信息检测 → 生成追问问题
- 多轮对话上下文
"""
import json
import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)

# 系统支持的 GPU 列表（用于 prompt 注入）
SUPPORTED_GPUS = {
    "ascend_910b": "华为昇腾 910B (AscendC)",
    "ascend_910c": "华为昇腾 910C (AscendC)",
    "rtx_4090": "NVIDIA RTX 4090 (CUDA)",
    "rtx_3090": "NVIDIA RTX 3090 (CUDA)",
    "h100_sxm5": "NVIDIA H100 SXM5 (CUDA)",
    "a100_80gb": "NVIDIA A100 80GB (CUDA)",
    "mi300x": "AMD MI300X (HIP)",
}

BACKEND_FOR_GPU = {
    "ascend_910b": "ascendc", "ascend_910c": "ascendc",
    "rtx_4090": "cuda", "rtx_3090": "cuda",
    "h100_sxm5": "cuda", "a100_80gb": "cuda",
    "mi300x": "hip",
}

SYSTEM_PROMPT = """你是一个算子生成系统的意图解析器。用户会用自然语言描述他们想要生成的 GPU 算子。

你的任务是从用户输入中提取以下信息，返回严格的 JSON：

```json
{
  "status": "ready" | "need_clarification",
  "operator": "算子名称（英文小写，如 rope, gelu, flash_attention, custom_xxx）",
  "operator_description": "算子的简要数学描述（用于后续代码生成）",
  "gpus": ["目标GPU列表"],
  "backend": "编程后端 (cuda/hip/ascendc/triton)",
  "dtype": "数据类型 (fp16/bf16/fp32)，默认 fp16",
  "questions": ["需要向用户追问的问题列表（仅 status=need_clarification 时）"],
  "confidence": 0.0-1.0
}
```

规则：
1. 如果用户明确说了算子名称和目标硬件，status="ready"
2. 如果缺少关键信息，status="need_clarification"，并在 questions 中列出需要追问的问题
3. 必须追问的情况：
   - 用户没有指定目标硬件/GPU → 追问想在哪个硬件上运行
   - 用户描述的算子语义不明确 → 追问具体的数学定义或行为
4. 不需要追问的情况：
   - dtype 未指定 → 默认 fp16
   - backend 未指定 → 根据 GPU 自动推断
5. 算子名称不限于预定义列表，任何合理的计算算子都可以
6. 对于常见算子（如 gelu, silu, rmsnorm, matmul, flash_attention, rope, layernorm, softmax 等），
   你应该能直接识别，不需要追问其数学定义
7. 对于不常见或自定义算子，需要追问其数学定义

系统支持的 GPU：
SUPPORTED_GPUS_PLACEHOLDER

追问时要友好、简洁，像一个专业的同事在和你确认需求。"""


class IntentParser:
    """LLM 驱动的意图解析器"""

    def __init__(self, llm_client):
        self.llm_client = llm_client
        self.conversation_history: list[dict] = []

    def _build_system_prompt(self) -> str:
        gpu_list = "\n".join(f"  - {k}: {v}" for k, v in SUPPORTED_GPUS.items())
        return SYSTEM_PROMPT.replace("SUPPORTED_GPUS_PLACEHOLDER", gpu_list)

    async def parse(self, user_input: str, context: dict = None) -> dict:
        """
        解析用户输入，返回结构化结果。

        返回:
          status="ready" → 信息完整，可以开始生成
          status="need_clarification" → 需要追问，questions 里有问题列表
        """
        # 构建 prompt
        if context and self.conversation_history:
            # 多轮对话：附加历史上下文
            history_str = "\n".join(
                f"{'用户' if h['role']=='user' else '系统'}: {h['content']}"
                for h in self.conversation_history[-6:]  # 最多保留 3 轮
            )
            user_prompt = f"""对话历史：
{history_str}

用户最新输入：{user_input}

之前已经提取的信息：
{json.dumps(context, ensure_ascii=False, indent=2)}

请根据用户最新输入更新解析结果。如果信息已经完整，status 设为 "ready"。"""
        else:
            user_prompt = f"用户输入：{user_input}"

        self.conversation_history.append({"role": "user", "content": user_input})

        try:
            response = await self.llm_client.chat(
                system=self._build_system_prompt(),
                user=user_prompt,
                temperature=0.1,
                max_tokens=1024,
            )

            result = self._extract_json(response)

            # 自动推断 backend
            if result.get("gpus") and not result.get("backend"):
                result["backend"] = BACKEND_FOR_GPU.get(result["gpus"][0])

            # 默认 dtype
            if not result.get("dtype"):
                result["dtype"] = "fp16"

            # 记录系统回复
            self.conversation_history.append({"role": "assistant", "content": response})

            return result

        except Exception as e:
            logger.error(f"[IntentParser] LLM parse failed: {e}")
            # 降级到规则解析
            return self._fallback_parse(user_input)

    def _extract_json(self, response: str) -> dict:
        """从 LLM 响应中提取 JSON"""
        # 尝试匹配 ```json ... ``` 代码块
        m = re.search(r"```json\s*\n?(.*?)\n?\s*```", response, re.DOTALL)
        if m:
            return json.loads(m.group(1))

        # 尝试匹配裸 JSON
        m = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", response, re.DOTALL)
        if m:
            return json.loads(m.group(0))

        raise ValueError(f"Cannot extract JSON from response: {response[:200]}")

    def _fallback_parse(self, text: str) -> dict:
        """规则降级解析（LLM 失败时使用）"""
        text_lower = text.lower()
        result = {
            "status": "need_clarification",
            "operator": None,
            "operator_description": "",
            "gpus": [],
            "backend": None,
            "dtype": "fp16",
            "questions": [],
            "confidence": 0.3,
        }

        # 简单关键词匹配算子
        known_ops = [
            "flash_attention", "flashattention", "rmsnorm", "layernorm",
            "gelu", "silu", "relu", "sigmoid", "tanh", "softmax",
            "matmul", "gemm", "rope", "rotary", "embedding", "dropout",
            "cross_entropy", "batchnorm", "groupnorm", "topk", "concat",
            "transpose", "mish", "add", "reduce_sum", "argmax",
        ]
        for op in sorted(known_ops, key=len, reverse=True):
            if op in text_lower:
                canonical = {
                    "flashattention": "flash_attention", "gemm": "matmul",
                    "rotary": "rope",
                }.get(op, op)
                result["operator"] = canonical
                break

        # 简单关键词匹配 GPU
        gpu_keywords = {
            "昇腾910b": "ascend_910b", "910b": "ascend_910b",
            "昇腾": "ascend_910b", "npu": "ascend_910b",
            "4090": "rtx_4090", "h100": "h100_sxm5",
            "a100": "a100_80gb", "mi300": "mi300x",
        }
        for kw, gpu_id in sorted(gpu_keywords.items(), key=lambda x: -len(x[0])):
            if kw in text_lower:
                result["gpus"].append(gpu_id)
                break

        # 判断缺什么
        if not result["operator"]:
            result["questions"].append("请问你想生成什么算子？可以描述算子的功能或数学定义。")
        if not result["gpus"]:
            gpu_list = ", ".join(f"{v}({k})" for k, v in SUPPORTED_GPUS.items())
            result["questions"].append(f"请问目标硬件是什么？支持: {gpu_list}")

        if result["operator"] and result["gpus"]:
            result["status"] = "ready"
            result["backend"] = BACKEND_FOR_GPU.get(result["gpus"][0])
            result["confidence"] = 0.7

        return result

    def reset(self):
        """重置对话历史"""
        self.conversation_history.clear()
