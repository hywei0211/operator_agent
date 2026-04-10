#!/usr/bin/env python3
"""
Operator Agent CLI — 自然语言驱动的异构算子生成系统

用法:
  # 自然语言生成算子（LLM 解析意图，缺信息会追问）
  python cli.py generate "帮我生成一个 SiLU 激活函数的算子，目标是昇腾 910B"
  python cli.py generate "写个 RoPE 算子"
  python cli.py generate "Generate FlashAttention v2 for H100 and MI300X"

  # 指定参数生成（跳过意图解析）
  python cli.py generate --op silu --gpu ascend_910b --backend ascendc
  python cli.py generate --op rmsnorm --gpu rtx_4090 --backend cuda

  # NPU 测试
  python cli.py npu-test --llm qwen
  python cli.py npu-test --llm mock --ops silu gelu

  # 查看算子仓库
  python cli.py registry list
  python cli.py registry show silu ascend_910b
  python cli.py registry stats

  # 交互模式（推荐，支持多轮追问）
  python cli.py interactive
"""
import asyncio
import json
import logging
import os
import sys

import click

sys.path.insert(0, os.path.dirname(__file__))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────
# 后端映射（用于显式参数模式）
# ────────────────────────────────────────────────────────────

BACKEND_FOR_GPU = {
    "ascend_910b": "ascendc", "ascend_910c": "ascendc",
    "rtx_4090": "cuda", "h100_sxm5": "cuda", "a100_80gb": "cuda", "rtx_3090": "cuda",
    "mi300x": "hip",
}


# ────────────────────────────────────────────────────────────
# 核心生成逻辑
# ────────────────────────────────────────────────────────────

async def _do_generate(op_name, gpus, backend, llm_backend, save, output_dir, review=False):
    """执行算子生成（可选 ReviewLoop 验证）"""
    from agents.base_agent import AgentContext
    from agents.spec_analyzer import OperatorSpecAgent
    from agents.code_generator import CodeGenAgent
    from knowledge_base.hardware_specs.gpu_database import get_gpu_spec
    from tools.llm_client import create_llm_client

    llm = create_llm_client(backend=llm_backend)
    spec_agent = OperatorSpecAgent(llm_client=llm)
    codegen = CodeGenAgent(llm_client=llm)

    for gpu_id in gpus:
        gpu_spec = get_gpu_spec(gpu_id)
        if gpu_spec is None:
            click.echo(f"  ⚠ GPU {gpu_id} not found in database, skipping")
            continue

        click.echo(f"  ⏳ Generating {op_name} for {gpu_id} ({backend})...")

        ctx = AgentContext(operator_name=op_name)
        spec_res = await spec_agent.run(ctx, request=op_name)
        if not spec_res.success:
            click.echo(f"  ❌ Spec analysis failed: {spec_res.error}")
            continue

        gen_res = await codegen.run(ctx, operator_ir=spec_res.output, gpu_spec=gpu_spec)
        if not gen_res.success:
            click.echo(f"  ❌ Code generation failed: {gen_res.error}")
            continue

        kernel = gen_res.output
        click.echo(f"  ✅ Generated: {len(kernel.source_code)} chars, "
                   f"backend={kernel.backend}")

        # ReviewLoop 验证（可选）
        if review:
            click.echo(f"  🔍 Running ReviewLoop verification...")
            from agents.review_loop import ReviewLoopAgent
            from mcp_servers.base_server import MCPClient
            from mcp_servers.remote_executor_server import RemoteExecutorMCPServer

            mcp = MCPClient()
            mcp.register_server(RemoteExecutorMCPServer())

            review_agent = ReviewLoopAgent(mcp_client=mcp, llm_client=llm)

            # 设置进度回调
            stage_names = {
                "static_review": "静态审查",
                "compile": "编译检查",
                "correctness": "正确性验证",
                "performance": "性能基准",
                "meta_review": "综合评审",
            }

            def _on_stage(stage, iteration, max_iter, passed):
                name = stage_names.get(stage, stage)
                if passed is None:
                    click.echo(f"    [{iteration}/{max_iter}] ⏳ {name}...")
                elif passed:
                    click.echo(f"    [{iteration}/{max_iter}] ✅ {name}")
                else:
                    click.echo(f"    [{iteration}/{max_iter}] ❌ {name} — 修复中...")

            review_agent.set_progress_callback(_on_stage)

            review_result = await review_agent.run(
                ctx, kernel=kernel, operator_ir=spec_res.output,
                gpu_spec=gpu_spec)

            if review_result.success and review_result.output:
                summary = review_result.output
                kernel = summary.final_kernel or kernel
                vl = getattr(kernel, 'verification_level', 'unknown')
                click.echo(f"  📊 Review: passed={summary.final_passed}, "
                           f"iters={summary.total_iterations}, "
                           f"BW={summary.bandwidth_utilization:.0%}")
            else:
                click.echo(f"  ⚠ Review failed: {review_result.error}")

        # 保存代码文件
        os.makedirs(output_dir, exist_ok=True)
        ext = {"cuda": ".cu", "hip": ".hip.cpp", "ascendc": ".cpp", "triton": ".py"}.get(kernel.backend, ".txt")
        filepath = os.path.join(output_dir, f"{op_name}_{gpu_id}{ext}")
        with open(filepath, "w") as f:
            f.write(kernel.source_code)
        click.echo(f"  📄 Saved: {filepath}")

        # 保存到算子仓库
        if save:
            try:
                from operators.registry import get_registry, OperatorEntry
                reg = get_registry()
                entry = OperatorEntry(
                    operator_name=op_name,
                    gpu_model=gpu_id,
                    backend=kernel.backend,
                    source_code=kernel.source_code,
                    header_code=kernel.header_code,
                    build_flags=kernel.build_flags,
                    launch_config=kernel.launch_config,
                    tags=["cli_generated"],
                )
                reg.register(entry)
                click.echo(f"  📦 Registered in operator registry")
            except Exception as e:
                click.echo(f"  ⚠ Registry save failed: {e}")

    click.echo("\n✅ Done.")


# ────────────────────────────────────────────────────────────
# LLM 意图解析 + 追问
# ────────────────────────────────────────────────────────────

async def _parse_with_clarification(user_input: str, llm_backend: str, interactive: bool = False) -> dict | None:
    """
    用 LLM 解析用户意图，缺信息时追问。

    返回 {"operator", "gpus", "backend"} 或 None（用户放弃）
    """
    from agents.intent_parser import IntentParser
    from tools.llm_client import create_llm_client

    llm = create_llm_client(backend=llm_backend)
    parser = IntentParser(llm_client=llm)

    max_rounds = 5
    context = None

    for round_idx in range(max_rounds):
        if round_idx == 0:
            result = await parser.parse(user_input)
        else:
            result = await parser.parse(user_input, context=context)

        if result.get("status") == "ready":
            return {
                "operator": result["operator"],
                "gpus": result.get("gpus", []),
                "backend": result.get("backend"),
                "operator_description": result.get("operator_description", ""),
            }

        # 需要追问
        questions = result.get("questions", [])
        if not questions:
            # LLM 没给出问题但也没 ready，用 clarification_question 兼容
            q = result.get("clarification_question", "请补充更多信息。")
            questions = [q] if isinstance(q, str) else q

        # 显示追问
        click.echo()
        for q in questions:
            click.echo(f"  🤔 {q}")

        if not interactive:
            # 非交互模式，无法追问，直接返回 None
            click.echo("  (非交互模式，无法追问。请用 --op/--gpu 显式指定，或使用 interactive 模式)")
            return None

        # 等待用户回答
        click.echo()
        try:
            user_input = click.prompt("operator-agent", prompt_suffix="> ").strip()
        except (EOFError, KeyboardInterrupt):
            return None

        if user_input.lower() in ("quit", "exit", "q", "cancel", "取消"):
            click.echo("  已取消。")
            return None

        # 保存当前上下文供下一轮使用
        context = {
            "operator": result.get("operator"),
            "gpus": result.get("gpus", []),
            "backend": result.get("backend"),
            "operator_description": result.get("operator_description", ""),
        }

    click.echo("  ⚠ 多轮追问超过限制，请直接用 --op/--gpu 指定参数。")
    return None


# ────────────────────────────────────────────────────────────
# CLI 命令组
# ────────────────────────────────────────────────────────────

@click.group()
def cli():
    """Operator Agent — 异构 GPU 算子自动生成系统"""
    pass


@cli.command()
@click.argument("request", required=False)
@click.option("--op", help="算子名称")
@click.option("--gpu", multiple=True, help="目标 GPU (ascend_910b/rtx_4090/h100_sxm5/mi300x)")
@click.option("--backend", type=click.Choice(["cuda", "hip", "ascendc", "triton"]), help="编程后端")
@click.option("--llm", default="qwen", type=click.Choice(["qwen", "openai", "anthropic", "mock"]), help="LLM 后端")
@click.option("--save/--no-save", default=True, help="是否保存到算子仓库")
@click.option("--output", default="./output", help="输出目录")
@click.option("--review/--no-review", default=True, help="是否运行 ReviewLoop 验证（默认开启）")
def generate(request, op, gpu, backend, llm, save, output, review):
    """生成算子内核（支持自然语言输入 + LLM 意图解析）

    \b
    自然语言模式（LLM 解析，缺信息会提示）:
      python cli.py generate "帮我生成 SiLU 算子，目标昇腾 910B"
      python cli.py generate "写个 RoPE"

    \b
    显式参数模式（跳过 LLM 解析）:
      python cli.py generate --op silu --gpu ascend_910b --backend ascendc

    \b
    跳过验证（快速生成，不走 ReviewLoop）:
      python cli.py generate --op silu --gpu rtx_4090 --no-review
    """
    if op and gpu:
        gpus = list(gpu)
        backend = backend or BACKEND_FOR_GPU.get(gpus[0], "cuda")
        _print_task_info(op, gpus, backend, llm)
        asyncio.run(_do_generate(op, gpus, backend, llm, save, output, review=review))
        return

    if request:
        # 自然语言模式：LLM 解析
        click.echo(f"\n  🧠 正在用 LLM 解析你的需求...")
        parsed = asyncio.run(_parse_with_clarification(request, llm, interactive=False))
        if parsed is None:
            return

        op = parsed["operator"]
        gpus = parsed["gpus"]
        backend = parsed.get("backend") or BACKEND_FOR_GPU.get(gpus[0], "cuda") if gpus else "cuda"
        _print_task_info(op, gpus, backend, llm)
        asyncio.run(_do_generate(op, gpus, backend, llm, save, output, review=review))
        return

    # 什么都没给
    click.echo("请输入自然语言描述或使用 --op/--gpu 指定参数。")
    click.echo("示例: python cli.py generate \"写个 RoPE 算子，目标昇腾 910B\"")
    click.echo("或者: python cli.py interactive  (推荐，支持多轮追问)")


def _print_task_info(op, gpus, backend, llm):
    click.echo(f"\n{'='*50}")
    click.echo(f"  算子: {op}")
    click.echo(f"  GPU:  {', '.join(gpus)}")
    click.echo(f"  后端: {backend}")
    click.echo(f"  LLM:  {llm}")
    click.echo(f"{'='*50}\n")


@cli.command("npu-test")
@click.option("--llm", default="mock", type=click.Choice(["qwen", "openai", "anthropic", "mock"]))
@click.option("--ops", multiple=True, help="只测试指定算子")
def npu_test(llm, ops):
    """在昇腾 910B NPU 上运行算子验证测试"""
    from tests.hetero.npu_test import NPUTester
    tester = NPUTester(llm_backend=llm)
    op_filter = list(ops) if ops else None
    asyncio.run(tester.run_all(op_filter=op_filter))


# ────────────────────────────────────────────────────────────
# 算子仓库管理
# ────────────────────────────────────────────────────────────

@cli.group()
def registry():
    """算子仓库管理"""
    pass


@registry.command("list")
@click.option("--gpu", default=None, help="按 GPU 过滤")
@click.option("--backend", default=None, help="按后端过滤")
def registry_list(gpu, backend):
    """列出仓库中的所有算子"""
    from operators.registry import get_registry
    reg = get_registry()
    entries = reg.list_operators(gpu_model=gpu)
    if backend:
        entries = [e for e in entries if e.backend == backend]

    if not entries:
        click.echo("仓库为空")
        return

    click.echo(f"\n{'算子':<20} {'GPU':<16} {'后端':<10} {'验证':<6} {'版本':<6} {'代码行数':<8}")
    click.echo("─" * 70)
    for e in entries:
        verified = "✅" if e.correctness_passed else "❌"
        lines = e.source_code.count("\n") + 1
        click.echo(f"{e.operator_name:<20} {e.gpu_model:<16} {e.backend:<10} "
                   f"{verified:<6} v{e.version:<5} {lines:<8}")
    click.echo(f"\n共 {len(entries)} 条记录")


@registry.command("show")
@click.argument("operator_name")
@click.argument("gpu_model")
def registry_show(operator_name, gpu_model):
    """查看算子详情"""
    from operators.registry import get_registry
    reg = get_registry()
    entry = reg.lookup(operator_name, gpu_model)
    if entry is None:
        click.echo(f"未找到: {operator_name}::{gpu_model}")
        return

    click.echo(f"\n算子: {entry.operator_name}")
    click.echo(f"GPU:  {entry.gpu_model}")
    click.echo(f"后端: {entry.backend}")
    click.echo(f"版本: v{entry.version}")
    click.echo(f"验证: {'通过' if entry.correctness_passed else '未通过'}")
    click.echo(f"误差: {entry.max_relative_error:.2e}")
    click.echo(f"带宽: {entry.bandwidth_utilization:.1%}")
    click.echo(f"标签: {entry.tags}")
    click.echo(f"\n{'─'*50}")
    click.echo(f"源代码 ({entry.source_code.count(chr(10))+1} 行):")
    click.echo(f"{'─'*50}")
    click.echo(entry.source_code[:3000])
    if len(entry.source_code) > 3000:
        click.echo(f"\n... (truncated, total {len(entry.source_code)} chars)")


@registry.command("stats")
def registry_stats():
    """显示仓库统计"""
    from operators.registry import get_registry
    reg = get_registry()
    s = reg.stats()
    click.echo(f"\n算子仓库统计:")
    click.echo(f"  总数:       {s['total']}")
    click.echo(f"  生产就绪:   {s['production_ready']}")
    click.echo(f"  按后端:     {s['by_backend']}")
    click.echo(f"  按 GPU:     {s['by_gpu']}")


@registry.command("history")
@click.argument("operator_name")
@click.argument("gpu_model")
def registry_history(operator_name, gpu_model):
    """查看算子版本历史"""
    from operators.registry import get_registry
    reg = get_registry()
    entries = reg.get_version_history(operator_name, gpu_model)
    if not entries:
        click.echo(f"未找到: {operator_name}::{gpu_model}")
        return

    click.echo(f"\n{operator_name}::{gpu_model} 版本历史:")
    for e in entries:
        from datetime import datetime
        ts = datetime.fromtimestamp(e.created_at).strftime("%Y-%m-%d %H:%M")
        verified = "✅" if e.correctness_passed else "❌"
        click.echo(f"  v{e.version}  {ts}  {verified}  "
                   f"err={e.max_relative_error:.2e}  "
                   f"bw={e.bandwidth_utilization:.1%}  "
                   f"level={getattr(e, 'verification_level', 'n/a'):10s}  "
                   f"lines={e.source_code.count(chr(10))+1}")


@registry.command("search")
@click.option("--op", default=None, help="算子名称（模糊匹配）")
@click.option("--gpu", default=None, help="按 GPU 过滤")
@click.option("--backend", default=None, help="按后端过滤")
@click.option("--min-bw", default=None, type=float, help="最低带宽利用率")
@click.option("--verified-only", is_flag=True, help="只显示验证通过的")
def registry_search(op, gpu, backend, min_bw, verified_only):
    """搜索算子仓库（支持多条件组合过滤）

    \b
    示例:
      python cli.py registry search --gpu h100_sxm5          # H100 上有哪些算子
      python cli.py registry search --op attention            # 搜 attention 相关
      python cli.py registry search --backend cuda --min-bw 0.6  # 高性能 CUDA 算子
      python cli.py registry search --verified-only           # 只看验证通过的
    """
    from operators.registry import get_registry
    reg = get_registry()
    entries = reg.list_operators(gpu_model=gpu)

    # 过滤
    if backend:
        entries = [e for e in entries if e.backend == backend]
    if op:
        op_lower = op.lower()
        entries = [e for e in entries if op_lower in e.operator_name.lower()]
    if min_bw is not None:
        entries = [e for e in entries if e.bandwidth_utilization >= min_bw]
    if verified_only:
        entries = [e for e in entries if e.correctness_passed]

    if not entries:
        click.echo("未找到匹配的算子")
        return

    click.echo(f"\n{'算子':<20} {'GPU':<16} {'后端':<10} {'验证':<6} {'BW':<8} {'验证等级':<14}")
    click.echo("─" * 80)
    for e in entries:
        verified = "✅" if e.correctness_passed else "❌"
        vl = getattr(e, 'verification_level', 'n/a')
        click.echo(f"{e.operator_name:<20} {e.gpu_model:<16} {e.backend:<10} "
                   f"{verified:<6} {e.bandwidth_utilization:<8.1%} {vl:<14}")
    click.echo(f"\n共 {len(entries)} 条结果")


@registry.command("export")
@click.argument("output_path", default="./operator_registry_export.json")
def registry_export(output_path):
    """导出算子仓库为 JSON 文件"""
    import json
    from operators.registry import get_registry
    from dataclasses import asdict
    reg = get_registry()
    entries = reg.list_operators()
    data = []
    for e in entries:
        d = {
            "operator_name": e.operator_name,
            "gpu_model": e.gpu_model,
            "backend": e.backend,
            "correctness_passed": e.correctness_passed,
            "bandwidth_utilization": e.bandwidth_utilization,
            "version": e.version,
            "verification_level": getattr(e, 'verification_level', 'none'),
            "code_lines": e.source_code.count("\n") + 1,
        }
        data.append(d)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    click.echo(f"导出 {len(data)} 条算子到 {output_path}")


# ────────────────────────────────────────────────────────────
# 编译错误知识库管理
# ────────────────────────────────────────────────────────────

@cli.group()
def kb():
    """编译错误知识库管理"""
    pass


@kb.command("stats")
def kb_stats():
    """查看知识库统计"""
    from knowledge_base.compile_error_kb import get_compile_error_kb
    s = get_compile_error_kb().stats()
    click.echo(f"\n编译错误知识库:")
    click.echo(f"  规则总数:   {s['total_patterns']}")
    click.echo(f"  按后端:     {s['by_backend']}")
    click.echo(f"  累计触发:   {s['total_occurrences']} 次")
    click.echo(f"  存储路径:   {s['store_path']}")


@kb.command("export")
@click.argument("output_path", default="./compile_errors_export.json")
def kb_export(output_path):
    """导出知识库（可提交 git 分享给团队）"""
    from knowledge_base.compile_error_kb import get_compile_error_kb
    count = get_compile_error_kb().export_patterns(output_path)
    click.echo(f"✅ 导出 {count} 条编译错误模式到 {output_path}")


@kb.command("import")
@click.argument("input_path")
@click.option("--overwrite", is_flag=True, help="覆盖已有规则（默认合并）")
def kb_import(input_path, overwrite):
    """从 JSON 文件导入知识库"""
    from knowledge_base.compile_error_kb import get_compile_error_kb
    count = get_compile_error_kb().import_patterns(input_path, overwrite=overwrite)
    click.echo(f"✅ 导入 {count} 条新规则")


# ────────────────────────────────────────────────────────────
# LLM 缓存管理
# ────────────────────────────────────────────────────────────

@cli.group()
def cache():
    """LLM 响应缓存管理"""
    pass


@cache.command("stats")
def cache_stats():
    """查看缓存统计"""
    from tools.llm_client import get_llm_cache
    s = get_llm_cache().stats()
    click.echo(f"\nLLM 缓存:")
    click.echo(f"  缓存条数:   {s['total_cached']}")
    click.echo(f"  存储路径:   {s['db_path']}")


@cache.command("clear")
def cache_clear():
    """清空 LLM 缓存"""
    from tools.llm_client import get_llm_cache
    get_llm_cache().clear()
    click.echo("✅ LLM 缓存已清空")


# ────────────────────────────────────────────────────────────
# 交互模式（支持多轮追问）

@cli.command()
@click.option("--llm", default="qwen", type=click.Choice(["qwen", "openai", "anthropic", "mock"]))
@click.option("--review/--no-review", default=True, help="是否运行 ReviewLoop 验证（默认开启）")
def interactive(llm, review):
    """交互模式 — 用自然语言生成算子（支持多轮追问）

    \b
    进入后直接输入需求，例如:
      > 生成一个 SiLU 算子，目标昇腾 910B
      > 写个 RoPE
      > generate gelu for RTX 4090
      > list   (查看仓库)
      > quit   (退出)

    \b
    如果你的描述缺少关键信息（如目标硬件），系统会自动追问。
    使用 --no-review 跳过 ReviewLoop 验证（快速迭代）。
    """
    review_label = "✅ 开启" if review else "❌ 关闭"
    click.echo("\n" + "=" * 55)
    click.echo("  🚀 Operator Agent 交互模式")
    click.echo(f"  LLM: {llm}  |  ReviewLoop: {review_label}")
    click.echo("  输入算子需求，系统会自动解析并追问缺失信息")
    click.echo("  命令: list, stats, help, quit")
    click.echo("=" * 55 + "\n")

    while True:
        try:
            user_input = click.prompt("operator-agent", prompt_suffix="> ").strip()
        except (EOFError, KeyboardInterrupt):
            click.echo("\nBye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            click.echo("Bye!")
            break

        # 内置命令
        if user_input.lower() in ("list", "ls"):
            _cmd_list()
            continue
        if user_input.lower() == "stats":
            _cmd_stats()
            continue
        if user_input.lower() == "help":
            _cmd_help()
            continue

        # LLM 意图解析 + 多轮追问
        click.echo(f"\n  🧠 正在解析...")
        parsed = asyncio.run(_parse_with_clarification(user_input, llm, interactive=True))
        if parsed is None:
            continue

        op = parsed["operator"]
        gpus = parsed["gpus"]
        backend = parsed.get("backend") or BACKEND_FOR_GPU.get(gpus[0], "cuda") if gpus else "cuda"

        _print_task_info(op, gpus, backend, llm)
        asyncio.run(_do_generate(op, gpus, backend, llm, save=True, output_dir="./output", review=review))


def _cmd_list():
    from operators.registry import get_registry
    reg = get_registry()
    entries = reg.list_operators()
    if not entries:
        click.echo("  仓库为空")
    else:
        for e in entries:
            v = "✅" if e.correctness_passed else "❌"
            click.echo(f"  {e.operator_name:<16} {e.gpu_model:<16} {e.backend:<10} {v}")


def _cmd_stats():
    from operators.registry import get_registry
    s = get_registry().stats()
    click.echo(f"  总数={s['total']}, 生产就绪={s['production_ready']}, "
               f"后端={s['by_backend']}, GPU={s['by_gpu']}")


def _cmd_help():
    click.echo("  输入自然语言描述来生成算子，例如:")
    click.echo("    写个 RoPE 算子，目标昇腾 910B")
    click.echo("    generate gelu for RTX 4090")
    click.echo("    帮我实现一个 FusedAddRMSNorm")
    click.echo("  如果缺少信息（如目标硬件），系统会自动追问。")
    click.echo("  命令: list, stats, help, quit")


if __name__ == "__main__":
    cli()
