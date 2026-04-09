"""
Operator Agent System - 主入口
异构GPU算子生成系统
"""
import asyncio
import json
import logging
import os
import sys
from pathlib import Path

import yaml

from agents.orchestrator import OrchestratorAgent
from agents.hardware_profiler import HardwareProfilerAgent
from agents.spec_analyzer import OperatorSpecAgent
from agents.code_generator import CodeGenAgent
from agents.optimizer import OptimizerAgent
from agents.verifier import VerifierAgent
from agents.distribution import DistributionAgent
from agents.base_agent import AgentContext
from models.operator_ir import ClusterConfig
from tools.llm_client import create_llm_client


def setup_logging(level: str = "INFO", log_file: str = None):
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=handlers,
    )


def load_config(config_path: str = "config/config.yaml") -> dict:
    if os.path.exists(config_path):
        with open(config_path) as f:
            return yaml.safe_load(f)
    return {}


def build_agent_system(config: dict) -> OrchestratorAgent:
    """构建Agent系统，注册所有子Agent"""
    llm_cfg = config.get("llm", {})
    llm_backend = llm_cfg.get("backend", "mock")

    # 如有API key则使用真实LLM，否则降级为Mock
    if llm_backend == "openai" and not (llm_cfg.get("api_key") or os.environ.get("OPENAI_API_KEY")):
        logging.warning("No OpenAI API key found, using mock LLM client")
        llm_backend = "mock"
    elif llm_backend == "anthropic" and not (llm_cfg.get("api_key") or os.environ.get("ANTHROPIC_API_KEY")):
        logging.warning("No Anthropic API key found, using mock LLM client")
        llm_backend = "mock"

    llm_client = create_llm_client(
        backend=llm_backend,
        model=llm_cfg.get("model", "gpt-4o"),
    )
    logging.info(f"Using LLM backend: {llm_backend}")

    # 创建Orchestrator
    orchestrator_cfg = {
        "max_opt_iterations": config.get("optimizer", {}).get("max_iterations", 3),
        "convergence_threshold": config.get("optimizer", {}).get("convergence_threshold", 0.02),
    }
    orchestrator = OrchestratorAgent(llm_client=llm_client, config=orchestrator_cfg)

    # 注册所有子Agent
    orchestrator.register_agent(HardwareProfilerAgent(llm_client=llm_client))
    orchestrator.register_agent(OperatorSpecAgent(llm_client=llm_client))
    orchestrator.register_agent(
        CodeGenAgent(
            llm_client=llm_client,
            config=config.get("code_gen", {})
        )
    )
    orchestrator.register_agent(
        OptimizerAgent(
            llm_client=llm_client,
            config=config.get("optimizer", {})
        )
    )
    orchestrator.register_agent(
        VerifierAgent(
            llm_client=llm_client,
            config=config.get("verifier", {})
        )
    )
    orchestrator.register_agent(
        DistributionAgent(llm_client=llm_client)
    )

    return orchestrator


async def run_operator_generation(
    operator_request: str,
    target_gpus: list[str],
    cluster_config: ClusterConfig = None,
    config: dict = None,
) -> dict:
    """
    主执行函数：为指定GPU生成算子

    Args:
        operator_request: 自然语言算子描述，如 "FlashAttention v2 with causal masking"
        target_gpus: 目标GPU列表，如 ["h100_sxm5", "mi300x"]
        cluster_config: 可选的集群配置
        config: 系统配置

    Returns:
        包含生成内核和验证报告的结果字典
    """
    config = config or {}

    # 如果没有提供cluster_config，根据target_gpus构建默认配置
    if cluster_config is None and target_gpus:
        from collections import Counter
        gpu_groups = {}
        for gpu in target_gpus:
            if gpu not in gpu_groups:
                gpu_groups[gpu] = []
            gpu_groups[gpu].append(f"node0")

        cluster_config = ClusterConfig(
            cluster_name="default_cluster",
            nodes=[{"gpu_model": gpu, "num_gpus": 1} for gpu in set(target_gpus)],
            gpu_groups=gpu_groups,
        )

    # 构建Agent系统
    orchestrator = build_agent_system(config)

    # 创建执行上下文
    context = AgentContext(
        operator_name=operator_request[:50],
        target_gpus=target_gpus,
    )

    # 执行工作流
    logging.info(f"Starting operator generation: {operator_request}")
    logging.info(f"Target GPUs: {target_gpus}")

    result = await orchestrator.run(
        context=context,
        operator_request=operator_request,
        cluster_config=cluster_config,
    )

    return {
        "success": result.success,
        "operator": operator_request,
        "target_gpus": target_gpus,
        "output": result.output,
        "error": result.error,
        "elapsed_seconds": result.elapsed_seconds,
        "metrics": result.metrics,
    }


def save_results(results: dict, output_dir: str = "./output"):
    """保存生成结果到文件"""
    os.makedirs(output_dir, exist_ok=True)

    if not results["success"] or not results["output"]:
        logging.error(f"Generation failed: {results.get('error')}")
        return

    output = results["output"]
    kernels = output.get("kernels", {})
    operator_name = output.get("operator_ir").name if output.get("operator_ir") else "unknown"

    for gpu_type, kernel in kernels.items():
        ext = {"cuda": ".cu", "hip": ".hip.cpp", "triton": ".py", "sycl": ".sycl.cpp"}.get(kernel.backend, ".txt")
        filename = f"{operator_name}_{gpu_type}{ext}"
        filepath = os.path.join(output_dir, filename)

        with open(filepath, "w") as f:
            if kernel.header_code:
                f.write(kernel.header_code + "\n\n")
            f.write(kernel.source_code)

        logging.info(f"Saved kernel: {filepath}")

    # 保存报告
    report_path = os.path.join(output_dir, f"{operator_name}_report.json")
    report_data = {
        "operator": operator_name,
        "target_gpus": results["target_gpus"],
        "elapsed_seconds": results["elapsed_seconds"],
        "kernels_generated": list(kernels.keys()),
        "all_verified": output.get("all_verified", False),
        "distribution_plan": str(output.get("distribution_plan")) if output.get("distribution_plan") else None,
    }
    with open(report_path, "w") as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)

    logging.info(f"Saved report: {report_path}")


async def main():
    """命令行主函数"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Operator Agent - 为异构GPU自动生成算子内核",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 为H100和MI300X生成FlashAttention内核
  python main.py --operator "FlashAttention v2" --gpus h100_sxm5 mi300x

  # 为混合集群生成RMSNorm内核
  python main.py --operator "RMSNorm" --gpus h100_sxm5 mi300x gaudi3

  # 使用自定义配置
  python main.py --operator "GELU activation" --gpus a100_80gb --config config/config.yaml
        """
    )
    parser.add_argument("--operator", "-o", required=True, help="算子描述（自然语言）")
    parser.add_argument("--gpus", "-g", nargs="+", required=True, help="目标GPU型号列表")
    parser.add_argument("--config", "-c", default="config/config.yaml", help="配置文件路径")
    parser.add_argument("--output", default="./output", help="输出目录")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    args = parser.parse_args()

    # 加载配置
    config = load_config(args.config)
    log_cfg = config.get("logging", {})
    setup_logging(
        level=args.log_level or log_cfg.get("level", "INFO"),
        log_file=log_cfg.get("file"),
    )

    # 执行生成
    results = await run_operator_generation(
        operator_request=args.operator,
        target_gpus=args.gpus,
        config=config,
    )

    if results["success"]:
        print(f"\n✅ 算子生成成功！")
        print(f"   算子: {results['operator']}")
        print(f"   生成后端: {list(results['output']['kernels'].keys())}")
        print(f"   验证通过: {results['output']['all_verified']}")
        print(f"   耗时: {results['elapsed_seconds']:.2f}s")

        # 保存结果
        output_cfg = config.get("output", {})
        if output_cfg.get("save_kernels", True):
            save_results(results, args.output)
    else:
        print(f"\n❌ 生成失败: {results['error']}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
