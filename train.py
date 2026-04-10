"""
train.py - 新主入口
用户通过此入口提交训练任务，系统自动处理 GPU 发现、算子生成、Review 和训练启动
"""
import asyncio
import json
import logging
import os
import sys
from pathlib import Path

from agents.base_agent import AgentContext
from orchestrator_v2 import MasterOrchestrator, SystemConfig


def setup_logging(level: str = "INFO"):
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stdout,
    )


# ════════════════════════════════════════════════════════════
# Python API（程序化调用）
# ════════════════════════════════════════════════════════════

async def train(
    training_code: str,
    gpu_list: list[str],
    config: SystemConfig = None,
    verbose: bool = True,
) -> dict:
    """
    主 API 入口

    Args:
        training_code: PyTorch 训练代码字符串
        gpu_list: 目标 GPU 列表，如 ["h100_sxm5", "mi300x"]
                  支持已知型号（直接训练）和未知型号（自动发现+生成）
        config: 系统配置
        verbose: 是否打印详细日志

    Returns:
        包含训练任务信息、生成的算子、监控报告的字典

    示例:
        result = await train(
            training_code=open("my_train.py").read(),
            gpu_list=["h100_sxm5", "mi300x"],
        )
    """
    if verbose:
        setup_logging("INFO")

    cfg = config or SystemConfig()
    orchestrator = MasterOrchestrator.create(cfg)

    context = AgentContext(
        operator_name="training",
        target_gpus=gpu_list,
    )

    result = await orchestrator.run(
        context,
        training_code=training_code,
        gpu_list=gpu_list,
    )

    if verbose:
        _print_result(result, gpu_list)

    return {
        "success": result.success,
        "output": result.output,
        "error": result.error,
        "metrics": result.metrics,
        "elapsed_seconds": result.elapsed_seconds,
    }


def _print_result(result, gpu_list: list[str]):
    """美化打印结果"""
    sep = "─" * 60
    print(f"\n{sep}")
    if result.success:
        output = result.output or {}
        print(f"✅  训练任务已就绪")
        print(f"   执行路径: {output.get('path', 'unknown')}")
        print(f"   目标 GPU: {gpu_list}")
        print(f"   生成算子: {list(output.get('kernels', {}).keys())}")

        job = output.get("training_job")
        if job:
            print(f"   任务 ID:  {job.job_id}")
            print(f"   状态:     {job.status}")
            if job.status == "dry_run_ready":
                script = f"./output/{job.job_id}/launch.sh"
                print(f"\n   启动命令: bash {script}")
                print(f"   (dry_run=True，请手动执行以上命令启动训练)")

        monitor = output.get("monitor")
        if monitor and monitor.alerts:
            print(f"\n   ⚠️  监控告警 ({len(monitor.alerts)}):")
            for alert in monitor.alerts[:3]:
                print(f"     [{alert['level'].upper()}] {alert['message']}")
            if monitor.recommendations:
                print(f"\n   建议:")
                for rec in monitor.recommendations[:3]:
                    print(f"     → {rec}")

        print(f"\n   耗时: {result.elapsed_seconds:.1f}s")
    else:
        print(f"❌  失败: {result.error}")
    print(sep)


# ════════════════════════════════════════════════════════════
# 命令行接口
# ════════════════════════════════════════════════════════════

async def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Operator Agent Train - 为异构 GPU 自动生成算子并启动训练",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  # 已知 GPU，直接训练
  python train.py --script my_train.py --gpus h100_sxm5 h100_sxm5

  # 未知 GPU，自动发现+生成算子
  python train.py --script my_train.py --gpus "Ascend 910C" "MI300X"

  # 四块异构 GPU
  python train.py --script llama_train.py \\
      --gpus h100_sxm5 h100_sxm5 mi300x "Ascend 910B" \\
      --llm openai

  # 使用内联代码（快速测试）
  python train.py --code "import torch; model=torch.nn.Linear(4096,4096).cuda()" \\
      --gpus h100_sxm5
        """
    )
    parser.add_argument("--script", "-s", help="训练脚本路径")
    parser.add_argument("--code", "-c", help="内联训练代码（用于快速测试）")
    parser.add_argument("--gpus", "-g", nargs="+", required=True, help="目标 GPU 型号列表")
    parser.add_argument("--llm", default="mock",
                        choices=["qwen", "openai", "anthropic", "mock"],
                        help="LLM 后端（默认 mock，不需要 API key）")
    parser.add_argument("--max-review-iters", type=int, default=5)
    parser.add_argument("--no-dry-run", action="store_true",
                        help="实际启动训练（默认只生成启动命令）")
    parser.add_argument("--output", default="./output", help="输出目录")
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--simulate", "-S", action="store_true",
                        help="模拟模式：无需 GPU，验证代码生成质量和数学正确性")
    parser.add_argument("--operators", nargs="+",
                        help="模拟模式下指定要验证的算子列表")

    args = parser.parse_args()

    # ── 模拟模式（无 GPU）────────────────────────────────────
    if args.simulate:
        ops = args.operators or ["flash_attention", "rmsnorm", "matmul", "gelu"]
        setup_logging(args.log_level)
        await simulate(operators=ops, gpu_list=args.gpus)
        return

    if not args.script and not args.code:
        parser.error("必须提供 --script 或 --code（或使用 --simulate 模式）")

    setup_logging(args.log_level)

    # 读取训练代码
    if args.script:
        training_code = Path(args.script).read_text()
    else:
        training_code = args.code

    # 构建系统配置
    config = SystemConfig(
        llm_backend=args.llm,
        max_review_iterations=args.max_review_iters,
        dry_run_training=not args.no_dry_run,
    )

    # 运行
    results = await train(
        training_code=training_code,
        gpu_list=args.gpus,
        config=config,
        verbose=True,
    )

    sys.exit(0 if results["success"] else 1)


# ════════════════════════════════════════════════════════════
# 快速示例函数（不需要真实训练代码）
# ════════════════════════════════════════════════════════════

EXAMPLE_LLAMA_TRAINING_CODE = '''
"""LLaMA 风格的简单训练循环（示例）"""
import torch
import torch.nn as nn
from torch.nn import functional as F

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

class LlamaAttention(nn.Module):
    def __init__(self, hidden_size=4096, num_heads=32):
        super().__init__()
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x):
        q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        # FlashAttention
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.o_proj(out)

class LlamaLayer(nn.Module):
    def __init__(self, hidden_size=4096, intermediate_size=11008):
        super().__init__()
        self.input_layernorm = RMSNorm(hidden_size)
        self.self_attn = LlamaAttention(hidden_size)
        self.post_attention_layernorm = RMSNorm(hidden_size)
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj   = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        h = x + self.self_attn(self.input_layernorm(x))
        # SiLU gate
        gate = F.silu(self.gate_proj(self.post_attention_layernorm(h)))
        h = h + self.down_proj(gate * self.up_proj(self.post_attention_layernorm(h)))
        return h

# 训练配置
hidden_size = 4096
num_layers = 32
batch_size = 4
seq_length = 2048

model = nn.Sequential(*[LlamaLayer(hidden_size) for _ in range(num_layers)])
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

# 训练循环
for step in range(1000):
    x = torch.randn(batch_size, seq_length, hidden_size, dtype=torch.bfloat16)
    loss = model(x).mean()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    if step % 100 == 0:
        print(f"Step {step}, loss={loss.item():.4f}")
'''


async def simulate(
    operators: list[str],
    gpu_list: list[str],
    show_roofline: bool = True,
    show_static: bool = True,
) -> dict:
    """
    模拟模式：无 GPU 情况下验证算子的数学正确性和生成质量

    包含四项检查：
    1. GPU 规格查询（MCP）
    2. 代码生成（Mock LLM）
    3. 静态代码分析
    4. Roofline 性能预测
    5. PyTorch CPU 数学验证（若安装了 PyTorch）

    示例：
        results = await simulate(
            operators=["flash_attention", "rmsnorm", "matmul"],
            gpu_list=["h100_sxm5", "mi300x"],
        )
    """
    from tools.cpu_simulator import CPUSimulator, RooflineSimulator, StaticCodeAnalyzer
    from mcp_servers.base_server import MCPClient
    from mcp_servers.gpu_spec_server import GPUSpecMCPServer
    from mcp_servers.sdk_docs_server import SDKDocsMCPServer
    from agents.spec_analyzer import OperatorSpecAgent
    from agents.code_generator import CodeGenAgent
    from tools.llm_client import create_llm_client

    print("\n" + "=" * 65)
    print("  Operator Agent - 模拟验证模式（无 GPU）")
    print("=" * 65)

    mcp = MCPClient()
    mcp.register_server(GPUSpecMCPServer())
    mcp.register_server(SDKDocsMCPServer())

    llm = create_llm_client(backend="mock")
    spec_agent  = OperatorSpecAgent(llm_client=llm)
    codegen     = CodeGenAgent(llm_client=llm)
    simulator   = CPUSimulator()
    roofline    = RooflineSimulator()
    sanalyzer   = StaticCodeAnalyzer()

    from agents.base_agent import AgentContext
    from knowledge_base.hardware_specs.gpu_database import get_gpu_spec

    all_results = {}

    for op_name in operators:
        print(f"\n─── 算子: {op_name} " + "─" * 40)

        # 1. 解析 OperatorIR
        ctx = AgentContext(operator_name=op_name)
        spec_res = await spec_agent.run(ctx, request=op_name)
        if not spec_res.success:
            print(f"  ❌ Spec analysis failed: {spec_res.error}")
            continue
        op_ir = spec_res.output
        print(f"  ✅ 算子类别: {op_ir.category.value}")

        op_results = {}
        for gpu_id in gpu_list:
            print(f"\n  GPU: {gpu_id}")
            gpu_spec = get_gpu_spec(gpu_id)
            if gpu_spec is None:
                print(f"    ⚠️  Unknown GPU, skipping")
                continue

            # 2. 代码生成
            gen_res = await codegen.run(AgentContext(), operator_ir=op_ir, gpu_spec=gpu_spec)
            if not gen_res.success:
                print(f"    ❌ CodeGen failed: {gen_res.error}")
                continue
            kernel = gen_res.output
            print(f"    ✅ 代码生成: backend={kernel.backend}, lines={kernel.source_code.count(chr(10))}")

            # 3. 静态分析
            if show_static:
                sa = sanalyzer.analyze(kernel.source_code, kernel.backend)
                icon = "✅" if sa["summary"] == "PASS" else "⚠️ "
                print(f"    {icon} 静态检查: {sa['summary']} "
                      f"({len(sa['passed_checks'])}/{len(sa['passed_checks'])+len(sa['failed_checks'])} rules)")
                if sa["warnings"]:
                    for w in sa["warnings"]:
                        print(f"      ⚠️  {w}")

            # 4. Roofline 性能预测
            if show_roofline:
                rf = roofline.predict(op_name, gpu_id,
                                     {"x_shape": [4, 512, 4096], "a_shape": [4096, 4096],
                                      "b_shape": [4096, 4096], "q_shape": [2, 8, 128, 64]})
                if "error" not in rf:
                    print(f"    📊 性能预测: {rf['bound_type']}, "
                          f"理论效率≤{rf['estimated_efficiency']:.0%}")
                    print(f"       算术密度: {rf['arithmetic_intensity']:.1f} FLOP/B "
                          f"(ridge={rf['ridge_point']:.1f})")
                    for s in rf["suggestions"][:2]:
                        print(f"       → {s}")

            # 5. PyTorch 数学验证
            test_inputs = simulator.generate_test_inputs(op_name)
            if test_inputs:
                sim_res = simulator.verify_operator(op_name, test_inputs)
                icon = "✅" if sim_res.math_correct else "❌"
                print(f"    {icon} 数学验证: {'通过' if sim_res.math_correct else '失败'} "
                      f"(tested {len(sim_res.tested_shapes)} shapes)")
                for note in sim_res.notes[:2]:
                    print(f"       {note}")

            op_results[gpu_id] = {
                "kernel_lines": kernel.source_code.count("\n"),
                "backend": kernel.backend,
                "static_pass": sa["summary"] == "PASS" if show_static else None,
                "roofline_bound": rf.get("bound_type") if show_roofline else None,
            }

        all_results[op_name] = op_results

    # 总结
    print("\n" + "=" * 65)
    print("  模拟验证完成（未使用任何 GPU）")
    print("  要在真实 GPU 上验证，请运行：python train.py --script <your_train.py> --gpus <gpu_list>")
    print("=" * 65)
    return all_results


async def run_example():
    """运行内置的 LLaMA 训练示例"""
    setup_logging("INFO")
    print("=" * 60)
    print("Operator Agent - LLaMA 训练示例（H100 + MI300X 异构集群）")
    print("=" * 60)

    results = await train(
        training_code=EXAMPLE_LLAMA_TRAINING_CODE,
        gpu_list=["h100_sxm5", "mi300x"],
        config=SystemConfig(
            llm_backend="mock",
            max_review_iterations=2,
            dry_run_training=True,
            parallel_operator_gen=False,  # Mock 模式串行更稳定
        ),
        verbose=True,
    )
    return results


if __name__ == "__main__":
    if len(sys.argv) == 1:
        # 无参数：运行内置示例
        asyncio.run(run_example())
    else:
        asyncio.run(main())
