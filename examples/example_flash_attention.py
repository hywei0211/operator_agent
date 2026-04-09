"""
示例：为H100 + MI300X混合集群生成FlashAttention v2算子
"""
import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import run_operator_generation, save_results, setup_logging


async def main():
    setup_logging("INFO")

    print("=" * 60)
    print("Operator Agent - FlashAttention 生成示例")
    print("=" * 60)

    # 场景：混合 NVIDIA H100 + AMD MI300X 集群
    results = await run_operator_generation(
        operator_request="FlashAttention v2 with causal masking, supports BF16 and FP16",
        target_gpus=["h100_sxm5", "mi300x"],
        config={
            "llm": {"backend": "mock"},   # 使用Mock，不需要真实API key
            "optimizer": {"max_iterations": 2, "target_efficiency": 0.7},
            "verifier": {"min_bandwidth_efficiency": 0.4},
        }
    )

    if results["success"]:
        output = results["output"]
        kernels = output.get("kernels", {})

        print(f"\n✅ 生成成功！")
        print(f"算子名称: {output['operator_ir'].name}")
        print(f"数学定义: {output['operator_ir'].math_description}")
        print(f"\n生成的内核:")
        for gpu_type, kernel in kernels.items():
            print(f"  [{gpu_type}] backend={kernel.backend}, "
                  f"代码长度={len(kernel.source_code)}字符, "
                  f"优化={kernel.optimizations_applied}")

        print(f"\n验证结果:")
        for gpu_type, verify_result in output.get("verification", {}).items():
            if verify_result.success and verify_result.output:
                report = verify_result.output
                print(f"  [{gpu_type}] {'通过' if report.overall_passed else '未通过'} - "
                      f"编译:{report.compilation_passed}, "
                      f"正确性:{report.correctness_passed}, "
                      f"性能:{report.performance_passed}")

        if output.get("distribution_plan"):
            plan = output["distribution_plan"]
            print(f"\n分布式方案: {plan.summary()}")

        print(f"\n总耗时: {results['elapsed_seconds']:.2f}s")

        # 保存结果
        save_results(results, "./output/flash_attention_example")
        print("\n内核文件已保存到 ./output/flash_attention_example/")
    else:
        print(f"\n❌ 生成失败: {results['error']}")


if __name__ == "__main__":
    asyncio.run(main())
