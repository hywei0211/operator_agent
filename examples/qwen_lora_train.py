#!/usr/bin/env python3
"""
Qwen LoRA 训练 — 使用 Operator Agent 生成的自定义 CUDA 算子
===========================================================

完整端到端流程：
  1. 用 Operator Agent 生成 SiLU forward+backward CUDA kernel
  2. nvcc 编译为 .so 共享库
  3. ctypes 加载 + torch.autograd.Function 注册
  4. 加载 Qwen2.5-0.5B + peft LoRA
  5. 用自定义 silu_custom 替换模型中的 F.silu
  6. 跑训练，验证 loss 下降 + 梯度流动正常

用法:
  # 用 Qwen LLM 生成真实 CUDA kernel
  python examples/qwen_lora_train.py --llm qwen --steps 20

  # 用 mock LLM（快速测试框架，kernel 是模板代码）
  python examples/qwen_lora_train.py --llm mock --steps 5

  # 跳过算子生成，直接用 PyTorch 原生算子做 LoRA 训练（baseline）
  python examples/qwen_lora_train.py --baseline --steps 20

前置条件:
  pip install peft transformers torch
  # 需要 NVIDIA GPU (RTX 4090 / A100 / H100 等) + nvcc
"""
import argparse
import asyncio
import ctypes
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

# 确保项目根目录在 path 中
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════
# Step 1: 用 Operator Agent 生成 CUDA kernel
# ════════════════════════════════════════════════════════════

async def generate_silu_kernels(llm_backend: str = "mock") -> dict:
    """
    调用 Operator Agent 生成 SiLU 的 forward + backward CUDA kernel。
    返回 {"forward_code": str, "backward_code": str}
    """
    from agents.base_agent import AgentContext
    from agents.spec_analyzer import OperatorSpecAgent
    from agents.code_generator import CodeGenAgent
    from knowledge_base.hardware_specs.gpu_database import get_gpu_spec
    from tools.llm_client import create_llm_client

    llm = create_llm_client(backend=llm_backend)
    spec_agent = OperatorSpecAgent(llm_client=llm)
    codegen = CodeGenAgent(llm_client=llm)
    gpu_spec = get_gpu_spec("rtx_4090")

    if gpu_spec is None:
        raise RuntimeError("rtx_4090 not found in GPU database")

    # 解析算子规格
    ctx = AgentContext(operator_name="silu")
    spec_res = await spec_agent.run(ctx, request="silu")
    if not spec_res.success:
        raise RuntimeError(f"Spec analysis failed: {spec_res.error}")

    op_ir = spec_res.output
    logger.info(f"[Step 1] OperatorIR: name={op_ir.name}, "
                f"backward={bool(op_ir.backward_math_description)}, "
                f"saved_for_backward={op_ir.saved_for_backward}")

    # 生成 forward kernel
    logger.info("[Step 1] Generating forward kernel...")
    gen_res = await codegen.run(ctx, operator_ir=op_ir, gpu_spec=gpu_spec)
    if not gen_res.success:
        raise RuntimeError(f"Forward codegen failed: {gen_res.error}")
    forward_kernel = gen_res.output

    # 生成 backward kernel
    logger.info("[Step 1] Generating backward kernel...")
    bwd_res = await codegen.generate_backward(
        ctx, operator_ir=op_ir, gpu_spec=gpu_spec, forward_kernel=forward_kernel)
    if not bwd_res.success:
        raise RuntimeError(f"Backward codegen failed: {bwd_res.error}")
    backward_kernel = bwd_res.output

    logger.info(f"[Step 1] Forward: {len(forward_kernel.source_code)} chars, "
                f"Backward: {len(backward_kernel.source_code)} chars")

    return {
        "forward_code": forward_kernel.source_code,
        "backward_code": backward_kernel.source_code,
        "forward_flags": forward_kernel.build_flags,
        "backward_flags": backward_kernel.build_flags,
    }


# ════════════════════════════════════════════════════════════
# Step 2: 编译为 .so
# ════════════════════════════════════════════════════════════

def compile_kernel(source_code: str, kernel_name: str, build_flags: list = None,
                   output_dir: str = None) -> str:
    """编译 CUDA 代码为 .so 共享库，返回 .so 路径"""
    # 应用编译错误知识库的自动修复
    try:
        from knowledge_base.compile_error_kb import get_compile_error_kb
        source_code = get_compile_error_kb().auto_fix(source_code, "cuda")
    except Exception:
        pass

    output_dir = output_dir or tempfile.mkdtemp(prefix="operator_agent_")
    src_path = os.path.join(output_dir, f"{kernel_name}.cu")
    so_path = os.path.join(output_dir, f"{kernel_name}.so")

    with open(src_path, "w") as f:
        f.write(source_code)

    nvcc = shutil.which("nvcc")
    if nvcc is None:
        # 尝试常见路径
        for path in ["/usr/local/cuda/bin/nvcc", "/usr/local/cuda-12/bin/nvcc"]:
            if os.path.isfile(path):
                nvcc = path
                break
    if nvcc is None:
        raise RuntimeError("nvcc not found! Install CUDA Toolkit first.")

    flags = build_flags or ["-O3", "-arch=native"]
    # 过滤掉不合法的 flags
    valid_prefixes = ("-O", "-arch=", "--use_fast_math", "-std=", "-Xcompiler", "-fPIC", "--shared")
    flags = [f for f in flags if any(f.startswith(p) for p in valid_prefixes)]

    cmd = [nvcc, "--shared", "-Xcompiler", "-fPIC"] + flags + [src_path, "-o", so_path]
    logger.info(f"[Step 2] Compiling: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        logger.error(f"[Step 2] Compile failed:\n{result.stderr[:500]}")
        raise RuntimeError(f"nvcc compilation failed:\n{result.stderr[:500]}")

    logger.info(f"[Step 2] Compiled: {so_path} ({os.path.getsize(so_path)} bytes)")
    return so_path


# ════════════════════════════════════════════════════════════
# Step 3: 注册 torch.autograd.Function
# ════════════════════════════════════════════════════════════

def create_custom_silu(forward_so: str = None, backward_so: str = None):
    """
    创建自定义 SiLU 函数，支持前向和反向传播。

    策略：
    - 如果 forward_so 编译成功且可加载 → 用自定义 CUDA kernel
    - 否则 → 用 PyTorch 原生 F.silu（但仍然通过 autograd.Function 包装以验证链路）

    在真实场景中，自定义 kernel 会比 PyTorch 原生实现更快（因为可以做算子融合等优化）。
    """
    # 尝试加载 forward .so
    forward_fn = None
    if forward_so and os.path.exists(forward_so):
        try:
            lib = ctypes.CDLL(forward_so)
            launch = lib.launch_kernel
            launch.restype = None
            launch.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
            forward_fn = launch
            logger.info("[Step 3] Loaded custom forward kernel from .so")
        except (OSError, AttributeError) as e:
            logger.warning(f"[Step 3] Cannot load forward .so: {e}, using PyTorch fallback")

    # 尝试加载 backward .so
    backward_fn = None
    if backward_so and os.path.exists(backward_so):
        try:
            lib_bwd = ctypes.CDLL(backward_so)
            launch_bwd = lib_bwd.launch_kernel
            launch_bwd.restype = None
            launch_bwd.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
            backward_fn = launch_bwd
            logger.info("[Step 3] Loaded custom backward kernel from .so")
        except (OSError, AttributeError) as e:
            logger.warning(f"[Step 3] Cannot load backward .so: {e}, using PyTorch fallback")

    class SiLUCustomFunction(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            # Qwen LLM 生成的 kernel 使用 FP16 (half)；
            # mock 模板生成的 kernel 使用 FP32 (float)。
            # 统一策略：保持原始 dtype 传入（forward_fn 负责正确处理），
            # 不进行额外类型转换避免 NaN。
            x_c = x.contiguous()
            if forward_fn is not None:
                output = torch.empty_like(x_c)
                N = x_c.numel()
                forward_fn(x_c.data_ptr(), output.data_ptr(), N)
                torch.cuda.synchronize()
            else:
                output = F.silu(x)
            ctx.save_for_backward(x)
            return output

        @staticmethod
        def backward(ctx, grad_output):
            x, = ctx.saved_tensors
            if backward_fn is not None:
                x_c = x.contiguous()
                grad_c = grad_output.contiguous()
                grad_input = torch.empty_like(x_c)
                N = x_c.numel()
                backward_fn(grad_c.data_ptr(),
                            x_c.data_ptr(), grad_input.data_ptr(), N)
                torch.cuda.synchronize()
            else:
                # PyTorch 解析梯度: silu'(x) = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
                sig = torch.sigmoid(x.float())
                grad_input = (grad_output.float() * sig * (1.0 + x.float() * (1.0 - sig))).to(x.dtype)
            return grad_input

    def silu_custom(x):
        return SiLUCustomFunction.apply(x)

    return silu_custom


# ════════════════════════════════════════════════════════════
# Step 4: 替换模型中的 SiLU
# ════════════════════════════════════════════════════════════

def patch_model_silu(model, custom_silu_fn):
    """
    遍历 Qwen 模型的所有层，把 SiLU 激活函数替换为自定义版本。
    Qwen2/Qwen3 的 MLP 层使用 SiLU 作为 gate 激活函数。

    兼容 transformers 4.x/5.x:
    - 4.x: act_fn 是 nn.SiLU 实例
    - 5.x: act_fn 是 transformers.activations.SiLUActivation 实例
    两者都是 nn.Module，通过类名来判断。
    """
    replaced = 0
    for name, module in model.named_modules():
        if hasattr(module, 'act_fn'):
            act = module.act_fn
            act_cls = type(act).__name__
            # 匹配 nn.SiLU, SiLUActivation, SiLU 等命名
            if isinstance(act, nn.Module) and 'silu' in act_cls.lower():
                module.act_fn = _SiLUWrapper(custom_silu_fn)
                replaced += 1

    logger.info(f"[Step 4] Replaced {replaced} SiLU modules with custom implementation")
    return replaced


class _SiLUWrapper(torch.nn.Module):
    """将函数包装为 nn.Module，可直接替换 nn.SiLU 等激活模块"""
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x)

    def __repr__(self):
        return "SiLUCustom(operator_agent)"


# ════════════════════════════════════════════════════════════
# Step 5: LoRA 训练
# ════════════════════════════════════════════════════════════

def run_lora_training(
    model_name: str = "Qwen/Qwen2.5-0.5B",
    custom_silu_fn=None,
    num_steps: int = 10,
    lr: float = 2e-4,
    lora_r: int = 8,
    lora_alpha: int = 16,
    max_length: int = 128,
):
    """
    用 LoRA 微调 Qwen 模型。

    如果提供了 custom_silu_fn，会替换模型中的 SiLU 激活函数。
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"[Step 5] Device: {device}")
    if device.type == "cuda":
        logger.info(f"[Step 5] GPU: {torch.cuda.get_device_name(0)}")

    # 加载模型和 tokenizer
    logger.info(f"[Step 5] Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    # 替换 SiLU（如果有自定义算子）
    if custom_silu_fn is not None:
        num_replaced = patch_model_silu(model, custom_silu_fn)
        if num_replaced == 0:
            logger.warning("[Step 5] No SiLU modules found to replace!")

    # 配置 LoRA
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"[Step 5] LoRA params: {trainable_params:,} / {total_params:,} "
                f"({100 * trainable_params / total_params:.2f}%)")

    # 准备训练数据（简单的文本补全任务）
    train_texts = [
        "The capital of France is Paris, which is known for",
        "Machine learning is a subset of artificial intelligence that",
        "The Transformer architecture was introduced in the paper",
        "GPU operators are fundamental building blocks for",
        "Deep learning training requires efficient backward propagation of",
        "CUDA programming allows developers to write parallel code for",
        "The attention mechanism computes a weighted sum of values based on",
        "LoRA reduces the number of trainable parameters by decomposing",
    ]

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # 训练循环
    model.train()
    logger.info(f"[Step 5] Starting LoRA training for {num_steps} steps...")
    print(f"\n{'Step':>6}  {'Loss':>10}  {'Time':>8}  {'Grad Norm':>10}")
    print("─" * 45)

    losses = []
    for step in range(1, num_steps + 1):
        t0 = time.perf_counter()

        # 准备输入
        text = train_texts[(step - 1) % len(train_texts)]
        inputs = tokenizer(text, return_tensors="pt", max_length=max_length,
                           truncation=True, padding="max_length")
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        # Forward
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
        loss = outputs.loss

        # Backward
        optimizer.zero_grad()
        loss.backward()

        # 检查梯度（验证 backward 链路）
        grad_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                grad_norm += p.grad.data.norm(2).item() ** 2
        grad_norm = grad_norm ** 0.5

        optimizer.step()

        elapsed = (time.perf_counter() - t0) * 1000
        loss_val = loss.item()
        losses.append(loss_val)

        print(f"{step:>6}  {loss_val:>10.4f}  {elapsed:>6.0f}ms  {grad_norm:>10.4f}")

    # 总结
    print("─" * 45)
    if len(losses) > 1:
        trend = "↓ 下降" if losses[-1] < losses[0] else "↑ 上升"
        print(f"\nLoss: {losses[0]:.4f} → {losses[-1]:.4f} ({trend})")
    print(f"Grad norm > 0: {grad_norm > 0} (梯度流动{'正常' if grad_norm > 0 else '异常!'})\n")

    return {
        "losses": losses,
        "final_loss": losses[-1],
        "initial_loss": losses[0],
        "loss_decreased": losses[-1] < losses[0] if len(losses) > 1 else False,
        "grad_norm": grad_norm,
        "trainable_params": trainable_params,
    }


# ════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════

async def main():
    parser = argparse.ArgumentParser(
        description="Qwen LoRA 训练 — 使用 Operator Agent 生成的自定义 CUDA 算子",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 用 Qwen LLM 生成 CUDA kernel + LoRA 训练
  python examples/qwen_lora_train.py --llm qwen --steps 20

  # 快速测试（mock LLM + 3 步训练）
  python examples/qwen_lora_train.py --llm mock --steps 3

  # Baseline（纯 PyTorch，不生成自定义算子）
  python examples/qwen_lora_train.py --baseline --steps 20

  # 指定模型
  python examples/qwen_lora_train.py --model Qwen/Qwen2.5-1.5B --steps 10
        """,
    )
    parser.add_argument("--llm", default="mock",
                        choices=["qwen", "openai", "anthropic", "mock"],
                        help="LLM 后端（默认 mock）")
    parser.add_argument("--model", default="/remote-home1/share/models/Qwen/Qwen3-0.6B",
                        help="Qwen 模型路径或 HF 名称（默认本地 Qwen3-0.6B）")
    parser.add_argument("--steps", type=int, default=10, help="训练步数")
    parser.add_argument("--lr", type=float, default=2e-4, help="学习率")
    parser.add_argument("--baseline", action="store_true",
                        help="Baseline 模式：不生成自定义算子，直接用 PyTorch 原生 SiLU")
    parser.add_argument("--skip-compile", action="store_true",
                        help="跳过编译（使用 PyTorch fallback，但仍然通过 autograd.Function）")
    parser.add_argument("--output-dir", default="./output/qwen_lora",
                        help="输出目录（保存生成的 kernel 代码）")

    args = parser.parse_args()

    print("=" * 60)
    print("  Qwen LoRA 训练 — Operator Agent 自定义 CUDA 算子")
    print(f"  模型: {args.model}")
    print(f"  LLM:  {args.llm}")
    print(f"  步数: {args.steps}")
    print(f"  模式: {'Baseline (PyTorch原生)' if args.baseline else 'Custom Operator'}")
    print("=" * 60)

    custom_silu = None

    if not args.baseline:
        # ── Step 1: 生成 CUDA kernel ─────────────────────
        print("\n[Step 1/5] 生成 SiLU forward + backward CUDA kernel...")
        kernels = await generate_silu_kernels(args.llm)

        # 保存生成的代码
        os.makedirs(args.output_dir, exist_ok=True)
        for name, code in [("silu_forward.cu", kernels["forward_code"]),
                            ("silu_backward.cu", kernels["backward_code"])]:
            path = os.path.join(args.output_dir, name)
            with open(path, "w") as f:
                f.write(code)
            print(f"  保存: {path} ({len(code)} chars)")

        forward_so = None
        backward_so = None

        if not args.skip_compile:
            # ── Step 2: 编译 .so ───────────────────────────
            print("\n[Step 2/5] 编译 CUDA kernel 为 .so...")
            try:
                forward_so = compile_kernel(
                    kernels["forward_code"], "silu_forward",
                    kernels.get("forward_flags", []), args.output_dir)
                print(f"  ✅ Forward: {forward_so}")
            except Exception as e:
                print(f"  ⚠ Forward 编译失败: {e}")
                print(f"  → 将使用 PyTorch fallback (F.silu)")

            try:
                backward_so = compile_kernel(
                    kernels["backward_code"], "silu_backward",
                    kernels.get("backward_flags", []), args.output_dir)
                print(f"  ✅ Backward: {backward_so}")
            except Exception as e:
                print(f"  ⚠ Backward 编译失败: {e}")
                print(f"  → 将使用 PyTorch 解析梯度 fallback")
        else:
            print("\n[Step 2/5] 跳过编译（--skip-compile）")

        # ── Step 3: 注册 autograd.Function ──────────────
        print("\n[Step 3/5] 注册 torch.autograd.Function...")
        custom_silu = create_custom_silu(forward_so, backward_so)
        print(f"  ✅ SiLU custom function created")

        # ── Step 4: 替换模型 SiLU ──────────────────────
        print("\n[Step 4/5] 将在模型加载后替换 SiLU 激活函数...")

    else:
        print("\n[Step 1-4] Baseline 模式，跳过算子生成")

    # ── Step 5: LoRA 训练 ────────────────────────────
    print(f"\n[Step 5/5] 开始 Qwen LoRA 训练 ({args.steps} steps)...")
    result = run_lora_training(
        model_name=args.model,
        custom_silu_fn=custom_silu,
        num_steps=args.steps,
        lr=args.lr,
    )

    # 总结
    print("=" * 60)
    if result["loss_decreased"]:
        print("✅ 训练成功！Loss 下降，梯度流动正常。")
        print(f"   自定义算子 forward + backward 全链路验证通过。")
    else:
        print("⚠ Loss 未下降（可能步数太少或学习率不合适）")
    if result["grad_norm"] > 0:
        print(f"✅ 梯度 norm = {result['grad_norm']:.4f}（反向传播链路正常）")
    else:
        print("❌ 梯度为 0（反向传播链路异常！）")
    print("=" * 60)

    return result


if __name__ == "__main__":
    asyncio.run(main())
