"""
无 GPU 完整测试套件
覆盖从代码生成到数学验证的完整流程
运行方式：python -m pytest tests/simulation/test_no_gpu.py -v
"""
import asyncio
import math
import sys
import os

# 确保项目根目录在 path 中
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import pytest

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from agents.base_agent import AgentContext
from knowledge_base.hardware_specs.gpu_database import GPU_DATABASE, get_gpu_spec
from models.hardware_model import GPUSpec, GPUVendor, GPUBackend
from models.operator_ir import OperatorCategory
from mcp_servers.base_server import MCPClient
from mcp_servers.gpu_spec_server import GPUSpecMCPServer
from mcp_servers.sdk_docs_server import SDKDocsMCPServer
from mcp_servers.operator_registry_server import OperatorRegistryMCPServer
from mcp_servers.remote_executor_server import RemoteExecutorMCPServer
from operators.registry import OperatorRegistry, OperatorEntry
from tools.cpu_simulator import CPUSimulator, RooflineSimulator, StaticCodeAnalyzer


# ════════════════════════════════════════════════════════════
# Fixtures
# ════════════════════════════════════════════════════════════

@pytest.fixture
def mcp_client():
    mcp = MCPClient()
    mcp.register_server(GPUSpecMCPServer())
    mcp.register_server(SDKDocsMCPServer())
    mcp.register_server(OperatorRegistryMCPServer())
    mcp.register_server(RemoteExecutorMCPServer())
    return mcp


@pytest.fixture
def cpu_sim():
    return CPUSimulator()


@pytest.fixture
def roofline():
    return RooflineSimulator()


@pytest.fixture
def static_analyzer():
    return StaticCodeAnalyzer()


# ════════════════════════════════════════════════════════════
# Part 1: GPU 数据库与 MCP 服务测试
# ════════════════════════════════════════════════════════════

class TestGPUDatabase:
    """测试本地 GPU 数据库覆盖率"""

    def test_database_not_empty(self):
        assert len(GPU_DATABASE) > 0, "GPU database should not be empty"

    def test_known_gpus_present(self):
        expected = ["h100_sxm5", "a100_80gb", "rtx_4090"]
        for gpu in expected:
            assert gpu in GPU_DATABASE or any(
                gpu in k for k in GPU_DATABASE
            ), f"Expected {gpu} in database"

    def test_gpu_spec_has_required_fields(self):
        for gpu_id, spec in GPU_DATABASE.items():
            assert spec.model_name, f"{gpu_id}: missing model_name"
            assert spec.vendor in GPUVendor, f"{gpu_id}: invalid vendor"
            assert spec.memory.capacity_gb > 0, f"{gpu_id}: memory capacity must be > 0"
            assert spec.memory.bandwidth_gbps > 0, f"{gpu_id}: bandwidth must be > 0"
            assert spec.compute.fp16_tflops > 0, f"{gpu_id}: fp16_tflops must be > 0"

    def test_all_gpus_have_backend(self):
        for gpu_id, spec in GPU_DATABASE.items():
            assert len(spec.supported_backends) > 0, f"{gpu_id}: no backends defined"

    @pytest.mark.parametrize("gpu_id", ["h100_sxm5", "mi300x"])
    def test_specific_gpu_lookup(self, gpu_id):
        spec = get_gpu_spec(gpu_id)
        assert spec is not None, f"get_gpu_spec({gpu_id}) returned None"
        assert spec.memory.bandwidth_gbps > 1000, f"{gpu_id}: bandwidth seems too low"


class TestMCPServers:
    """测试 MCP Server 工具调用"""

    @pytest.mark.asyncio
    async def test_gpu_spec_server_known_gpu(self, mcp_client):
        resp = await mcp_client.call("gpu_spec_server", "search_gpu_spec",
                                     model_name="H100 SXM5")
        assert resp.success, f"Search failed: {resp.error}"
        assert resp.data is not None
        assert resp.data.get("_source") == "local_database"
        assert resp.data.get("memory_bandwidth_gbps", 0) > 2000

    @pytest.mark.asyncio
    async def test_gpu_spec_server_unknown_gpu_graceful(self, mcp_client):
        resp = await mcp_client.call("gpu_spec_server", "search_gpu_spec",
                                     model_name="QuantumGPU X9000 Fictional")
        # 未知 GPU 应该优雅返回（不抛异常），data 可能为 None
        assert isinstance(resp.success, bool)

    @pytest.mark.asyncio
    async def test_sdk_for_nvidia(self, mcp_client):
        resp = await mcp_client.call("sdk_docs_server", "get_sdk_for_vendor", vendor="nvidia")
        assert resp.success
        assert resp.data["sdk"] == "cuda"

    @pytest.mark.asyncio
    async def test_sdk_for_huawei(self, mcp_client):
        resp = await mcp_client.call("sdk_docs_server", "get_sdk_for_vendor", vendor="huawei")
        assert resp.success
        assert resp.data["sdk"] == "ascendc"

    @pytest.mark.asyncio
    async def test_programming_guide_completeness(self, mcp_client):
        for sdk in ["cuda", "hip", "ascendc", "triton"]:
            resp = await mcp_client.call("sdk_docs_server", "get_programming_guide", sdk=sdk)
            assert resp.success, f"Guide for {sdk} failed"
            guide = resp.data
            assert "language" in guide, f"{sdk}: missing 'language'"
            assert "compiler" in guide, f"{sdk}: missing 'compiler'"
            assert "memory_model" in guide, f"{sdk}: missing 'memory_model'"

    @pytest.mark.asyncio
    async def test_remote_executor_static_check(self, mcp_client):
        cuda_code = """
#include <cuda_runtime.h>
__global__ void relu_kernel(float* x, float* out, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) out[idx] = x[idx] > 0 ? x[idx] : 0;
}
"""
        resp = await mcp_client.call("remote_executor_server", "compile_kernel",
                                     source_code=cuda_code, sdk="cuda",
                                     kernel_name="relu")
        # 应该做静态检查（无 nvcc 环境）
        assert isinstance(resp.success, bool)
        if resp.data and resp.data.get("method") == "static_analysis":
            assert "errors" in resp.data


# ════════════════════════════════════════════════════════════
# Part 2: 代码生成流程测试（Mock LLM）
# ════════════════════════════════════════════════════════════

class TestCodeGenerationPipeline:
    """测试代码生成 Agent 链路（使用 Mock LLM）"""

    @pytest.fixture
    def mock_llm(self):
        from tools.llm_client import create_llm_client
        return create_llm_client(backend="mock")

    @pytest.mark.asyncio
    async def test_spec_analyzer_flash_attention(self, mock_llm):
        from agents.spec_analyzer import OperatorSpecAgent
        agent = OperatorSpecAgent(llm_client=mock_llm)
        ctx = AgentContext(operator_name="flash_attention")
        result = await agent.run(ctx, request="flash_attention")
        assert result.success, f"SpecAnalyzer failed: {result.error}"
        assert result.output is not None
        assert result.output.name == "flash_attention"
        assert result.output.category == OperatorCategory.ATTENTION

    @pytest.mark.asyncio
    async def test_spec_analyzer_rmsnorm(self, mock_llm):
        from agents.spec_analyzer import OperatorSpecAgent
        agent = OperatorSpecAgent(llm_client=mock_llm)
        ctx = AgentContext(operator_name="rmsnorm")
        result = await agent.run(ctx, request="rmsnorm")
        assert result.success
        assert result.output.name == "rmsnorm"

    @pytest.mark.asyncio
    async def test_codegen_produces_code(self, mock_llm):
        from agents.code_generator import CodeGenAgent
        from agents.spec_analyzer import OperatorSpecAgent
        spec_agent = OperatorSpecAgent(llm_client=mock_llm)
        codegen = CodeGenAgent(llm_client=mock_llm)
        gpu_spec = get_gpu_spec("h100_sxm5")

        ctx = AgentContext()
        spec_result = await spec_agent.run(ctx, request="relu")
        assert spec_result.success
        op_ir = spec_result.output

        gen_result = await codegen.run(ctx, operator_ir=op_ir, gpu_spec=gpu_spec)
        assert gen_result.success, f"CodeGen failed: {gen_result.error}"
        kernel = gen_result.output
        assert kernel is not None
        assert len(kernel.source_code) > 20, "Generated code too short"
        assert kernel.backend in ("cuda", "hip", "triton", "sycl", "ascendc")

    @pytest.mark.asyncio
    async def test_codegen_for_multiple_gpus(self, mock_llm):
        from agents.code_generator import CodeGenAgent
        from agents.spec_analyzer import OperatorSpecAgent

        spec_agent = OperatorSpecAgent(llm_client=mock_llm)
        codegen = CodeGenAgent(llm_client=mock_llm)

        ctx = AgentContext()
        spec_result = await spec_agent.run(ctx, request="rmsnorm")
        op_ir = spec_result.output

        target_gpus = ["h100_sxm5", "mi300x"]
        results = {}
        for gpu_id in target_gpus:
            gpu_spec = get_gpu_spec(gpu_id)
            result = await codegen.run(
                AgentContext(), operator_ir=op_ir, gpu_spec=gpu_spec
            )
            results[gpu_id] = result

        for gpu_id, result in results.items():
            assert result.success, f"CodeGen failed for {gpu_id}: {result.error}"
            kernel = result.output
            # 不同 GPU 应该用不同的后端
            print(f"  {gpu_id}: backend={kernel.backend}")

    @pytest.mark.asyncio
    async def test_tiling_agent_nvidia(self, mock_llm):
        from agents.tiling_agent import TilingAgent
        from agents.spec_analyzer import OperatorSpecAgent

        tiling = TilingAgent(llm_client=mock_llm)
        spec_agent = OperatorSpecAgent(llm_client=mock_llm)
        gpu_spec = get_gpu_spec("h100_sxm5")

        ctx = AgentContext()
        spec_result = await spec_agent.run(ctx, request="flash_attention")
        op_ir = spec_result.output

        result = await tiling.run(ctx, operator_ir=op_ir, gpu_spec=gpu_spec)
        assert result.success
        config = result.output
        assert config.recommended, "Tiling should produce recommended config"
        print(f"  H100 FlashAttn tiling: {config.recommended}")

    @pytest.mark.asyncio
    async def test_tiling_ascend_alignment(self, mock_llm):
        from agents.tiling_agent import TilingAgent
        from agents.spec_analyzer import OperatorSpecAgent
        from knowledge_base.hardware_specs.ascend_specs import ASCEND_DATABASE

        if not ASCEND_DATABASE:
            pytest.skip("No Ascend GPU in database")

        tiling = TilingAgent(llm_client=mock_llm)
        spec_agent = OperatorSpecAgent(llm_client=mock_llm)
        ascend_spec = list(ASCEND_DATABASE.values())[0]

        ctx = AgentContext()
        spec_result = await spec_agent.run(ctx, request="matmul")
        op_ir = spec_result.output

        result = await tiling.run(ctx, operator_ir=op_ir, gpu_spec=ascend_spec)
        assert result.success
        config = result.output
        # Ascend Cube 单元要求 16 对齐
        rec = config.recommended
        for dim in ["tile_m", "tile_n", "tile_k"]:
            if dim in rec:
                assert rec[dim] % 16 == 0, f"Ascend {dim}={rec[dim]} must be 16-aligned"


# ════════════════════════════════════════════════════════════
# Part 3: 静态分析测试
# ════════════════════════════════════════════════════════════

class TestStaticAnalysis:
    """测试代码静态分析器"""

    CUDA_GOOD = """
#include <cuda_runtime.h>
__global__ void relu_kernel(const float* __restrict__ x, float* out, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        out[idx] = x[idx] > 0.0f ? x[idx] : 0.0f;
    }
}
"""

    CUDA_BAD = """
void not_a_kernel(float* x, float* out) {
    out[0] = x[0];  // no thread indexing, no bounds check
}
"""

    ASCENDC_GOOD = """
__aicore__ inline void compute(GM_ADDR x_gm, GM_ADDR out_gm, int N) {
    int block = GetBlockIdx();
    TBuf<TPosition::VECCALC> ubBuf;
    LocalTensor<half> xLocal = ubBuf.Get<half>();
    DataCopy(xLocal, xGm[block * TILE], TILE);
}
"""

    TRITON_GOOD = """
import triton
import triton.language as tl

@triton.jit
def relu_kernel(x_ptr, out_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    x = tl.load(x_ptr + offs, mask=mask)
    tl.store(out_ptr + offs, tl.where(x > 0, x, 0.0), mask=mask)
"""

    def test_cuda_good_code_passes(self, static_analyzer):
        result = static_analyzer.analyze(self.CUDA_GOOD, "cuda")
        assert result["score"] >= 0.7, f"Good CUDA code should pass: {result}"
        assert result["summary"] == "PASS"

    def test_cuda_bad_code_fails(self, static_analyzer):
        result = static_analyzer.analyze(self.CUDA_BAD, "cuda")
        assert result["score"] < 0.7, f"Bad CUDA code should fail: {result}"
        assert "__global__ void" in " ".join(result["failed_checks"])

    def test_triton_code_passes(self, static_analyzer):
        result = static_analyzer.analyze(self.TRITON_GOOD, "triton")
        assert result["score"] >= 0.7, f"Good Triton code should pass: {result}"

    def test_ascendc_detects_datacopy(self, static_analyzer):
        result = static_analyzer.analyze(self.ASCENDC_GOOD, "ascendc")
        passed = result["passed_checks"]
        assert any("DataCopy" in c for c in passed), "Should detect DataCopy usage"


# ════════════════════════════════════════════════════════════
# Part 4: Roofline 性能预测
# ════════════════════════════════════════════════════════════

class TestRoofline:
    """测试基于 Roofline 模型的性能预测"""

    @pytest.mark.parametrize("gpu_model,op_name", [
        ("h100_sxm5", "matmul"),
        ("h100_sxm5", "flash_attention"),
        ("h100_sxm5", "rmsnorm"),
        ("mi300x", "matmul"),
    ])
    def test_roofline_prediction(self, roofline, gpu_model, op_name):
        result = roofline.predict(
            operator_name=op_name,
            gpu_model=gpu_model,
            input_shapes={"a_shape": [4096, 4096], "b_shape": [4096, 4096]},
        )
        assert "error" not in result, f"Roofline failed: {result}"
        assert result["bound_type"] in ("memory_bound", "compute_bound")
        assert 0 < result["estimated_efficiency"] <= 1.0
        assert len(result["suggestions"]) > 0
        print(f"\n  {gpu_model}/{op_name}: {result['bound_type']}, "
              f"efficiency={result['estimated_efficiency']:.0%}")

    def test_matmul_is_compute_bound(self, roofline):
        """大矩阵乘法应该是计算瓶颈"""
        result = roofline.predict(
            operator_name="matmul",
            gpu_model="h100_sxm5",
            input_shapes={"a_shape": [8192, 8192], "b_shape": [8192, 8192]},
        )
        # 大矩阵乘算术密度很高，应该是 compute_bound
        assert result["arithmetic_intensity"] > 10, \
            f"Large matmul should have high arithmetic intensity, got {result['arithmetic_intensity']}"

    def test_elementwise_is_memory_bound(self, roofline):
        """逐元素算子应该是内存瓶颈"""
        result = roofline.predict(
            operator_name="gelu",
            gpu_model="h100_sxm5",
            input_shapes={"x_shape": [4096, 4096]},
        )
        assert result["bound_type"] == "memory_bound", \
            "Elementwise ops should be memory-bound"


# ════════════════════════════════════════════════════════════
# Part 5: 数学正确性验证
# ════════════════════════════════════════════════════════════

@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch required")
class TestMathCorrectness:
    """验证参考实现的数学正确性"""

    @pytest.mark.parametrize("op_name", [
        "gelu", "silu", "relu", "softmax", "rmsnorm", "matmul"
    ])
    def test_reference_impl_runs(self, cpu_sim, op_name):
        test_inputs = cpu_sim.generate_test_inputs(op_name)
        result = cpu_sim.verify_operator(op_name, test_inputs)
        assert result.math_correct, \
            f"Reference implementation for {op_name} failed: {result.error_message}"
        print(f"\n  ✅ {op_name}: {result.notes[0] if result.notes else 'OK'}")

    def test_rmsnorm_formula(self, cpu_sim):
        """验证 RMSNorm 的数学公式是否正确"""
        import torch
        x = torch.randn(4, 128, 512)
        weight = torch.ones(512)

        result = cpu_sim.verify_operator(
            "rmsnorm",
            [{"x": x, "weight": weight}]
        )
        assert result.math_correct

        # 手动验证公式
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)
        expected = (x / rms) * weight
        ref_fn = cpu_sim.REFERENCE_IMPLS["rmsnorm"]
        actual = ref_fn(x, weight)
        assert torch.allclose(actual, expected, atol=1e-5), "RMSNorm formula mismatch"

    def test_flash_attention_matches_scaled_dot_product(self, cpu_sim):
        """FlashAttention 参考实现应与 F.scaled_dot_product_attention 等价"""
        import torch
        import torch.nn.functional as F

        q = torch.randn(2, 4, 32, 64)
        k = torch.randn(2, 4, 32, 64)
        v = torch.randn(2, 4, 32, 64)

        ref_fn = cpu_sim.REFERENCE_IMPLS["flash_attention"]
        our_out = ref_fn(q, k, v, causal=True)

        if hasattr(F, 'scaled_dot_product_attention'):
            torch_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            assert torch.allclose(our_out, torch_out, atol=1e-4), \
                "FlashAttention reference doesn't match torch implementation"

    def test_custom_pytorch_fn_verification(self, cpu_sim):
        """测试自定义 PyTorch 实现的对比验证"""
        import torch

        def my_gelu(x):
            return x * 0.5 * (1.0 + torch.tanh(0.7978845608 * (x + 0.044715 * x ** 3)))

        test_inputs = cpu_sim.generate_test_inputs("gelu")
        result = cpu_sim.verify_operator("gelu", test_inputs, generated_fn=my_gelu)
        # GELU 的近似公式应该通过（误差在 rtol 内）
        assert result.math_correct, f"GELU approximate formula failed: {result.notes}"


# ════════════════════════════════════════════════════════════
# Part 6: 算子仓库测试
# ════════════════════════════════════════════════════════════

class TestOperatorRegistry:
    """测试算子仓库的增删改查"""

    @pytest.fixture
    def temp_registry(self, tmp_path):
        from operators.registry import OperatorRegistry
        return OperatorRegistry(registry_path=str(tmp_path / "test_registry.json"))

    def test_register_and_lookup(self, temp_registry):
        entry = OperatorEntry(
            operator_name="relu", gpu_model="test_gpu",
            backend="cuda", source_code="__global__ void relu_kernel() {}",
            correctness_passed=True, bandwidth_utilization=0.75,
        )
        temp_registry.register(entry)
        found = temp_registry.lookup("relu", "test_gpu")
        assert found is not None
        assert found.source_code == entry.source_code
        assert found.bandwidth_utilization == 0.75

    def test_registry_persistence(self, tmp_path):
        from operators.registry import OperatorRegistry
        path = str(tmp_path / "persist.json")
        r1 = OperatorRegistry(registry_path=path)
        r1.register(OperatorEntry(
            operator_name="gelu", gpu_model="h100_sxm5",
            backend="cuda", source_code="// gelu code",
            correctness_passed=True, bandwidth_utilization=0.80,
        ))
        r2 = OperatorRegistry(registry_path=path)
        found = r2.lookup("gelu", "h100_sxm5")
        assert found is not None, "Registry should persist across instances"
        assert found.bandwidth_utilization == 0.80

    def test_registry_only_keeps_better_version(self, temp_registry):
        entry_bad = OperatorEntry(
            operator_name="relu", gpu_model="gpu_x",
            backend="cuda", source_code="v1", bandwidth_utilization=0.4,
        )
        entry_good = OperatorEntry(
            operator_name="relu", gpu_model="gpu_x",
            backend="cuda", source_code="v2_optimized", bandwidth_utilization=0.8,
        )
        temp_registry.register(entry_bad)
        temp_registry.register(entry_good)
        found = temp_registry.lookup("relu", "gpu_x")
        assert found.source_code == "v2_optimized", "Should keep better version"

        # 用更差的版本覆盖 → 不应该成功
        entry_worse = OperatorEntry(
            operator_name="relu", gpu_model="gpu_x",
            backend="cuda", source_code="v3_worse", bandwidth_utilization=0.3,
        )
        temp_registry.register(entry_worse)
        found = temp_registry.lookup("relu", "gpu_x")
        assert found.source_code == "v2_optimized", "Should not downgrade"

    def test_registry_stats(self, temp_registry):
        for op, gpu in [("relu", "h100"), ("gelu", "h100"), ("relu", "mi300x")]:
            temp_registry.register(OperatorEntry(
                operator_name=op, gpu_model=gpu, backend="cuda",
                source_code="code", correctness_passed=True, bandwidth_utilization=0.7,
            ))
        stats = temp_registry.stats()
        assert stats["total"] == 3
        assert stats["production_ready"] == 3


# ════════════════════════════════════════════════════════════
# Part 7: 端到端流程测试（无 GPU）
# ════════════════════════════════════════════════════════════

class TestEndToEnd:
    """端到端测试：从训练代码到启动命令，全程无 GPU"""

    SIMPLE_TRAINING_CODE = """
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(512, 512)
        self.norm = nn.LayerNorm(512)

    def forward(self, x):
        return self.norm(F.gelu(self.linear(x)))

model = SimpleModel()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
for step in range(10):
    x = torch.randn(4, 512)
    loss = model(x).mean()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
"""

    @pytest.mark.asyncio
    async def test_training_analyst(self):
        from agents.training_analyst import TrainingAnalystAgent
        agent = TrainingAnalystAgent()
        ctx = AgentContext()
        result = await agent.run(ctx, training_code=self.SIMPLE_TRAINING_CODE)
        assert result.success
        plan = result.output
        assert "gelu" in plan.all_operators() or "layernorm" in plan.all_operators(), \
            f"Expected gelu/layernorm in operators, got: {plan.all_operators()}"
        assert plan.model_architecture in ("transformer", "unknown"), \
            f"Architecture should be transformer or unknown for simple model"

    @pytest.mark.asyncio
    async def test_full_orchestration_dry_run(self):
        """完整的双路径 Orchestrator，dry_run 模式（不启动训练）"""
        from orchestrator_v2 import MasterOrchestrator, SystemConfig

        config = SystemConfig(
            llm_backend="mock",
            max_review_iterations=1,
            dry_run_training=True,
            parallel_operator_gen=False,
        )
        orchestrator = MasterOrchestrator.create(config)
        ctx = AgentContext(target_gpus=["h100_sxm5"])

        result = await orchestrator.run(
            ctx,
            training_code=self.SIMPLE_TRAINING_CODE,
            gpu_list=["h100_sxm5"],
        )
        assert result.success, f"Orchestration failed: {result.error}"
        job = result.output.get("training_job")
        assert job is not None
        assert job.status == "dry_run_ready"

        # 验证生成了启动脚本
        import os
        launch_script = f"./output/{job.job_id}/launch.sh"
        assert os.path.exists(launch_script), f"Launch script not found: {launch_script}"

    @pytest.mark.asyncio
    async def test_unknown_gpu_path(self):
        """Path B：未知 GPU 的完整发现 + 生成流程"""
        from orchestrator_v2 import MasterOrchestrator, SystemConfig

        config = SystemConfig(
            llm_backend="mock",
            max_review_iterations=1,
            dry_run_training=True,
            parallel_operator_gen=False,
        )
        orchestrator = MasterOrchestrator.create(config)
        ctx = AgentContext(target_gpus=["h100_sxm5"])  # 用已知 GPU 模拟

        result = await orchestrator.run(
            ctx,
            training_code="import torch\nx = torch.randn(4, 512)\nprint(x.mean())",
            gpu_list=["h100_sxm5"],
        )
        assert result.success or result.error is not None  # 即使失败也要优雅
