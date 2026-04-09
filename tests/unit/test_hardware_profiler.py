"""
硬件分析Agent单元测试
"""
import asyncio
import pytest

from agents.hardware_profiler import HardwareProfilerAgent
from agents.base_agent import AgentContext
from models.operator_ir import ClusterConfig
from knowledge_base.hardware_specs.gpu_database import GPU_DATABASE, get_gpu_spec


class TestHardwareProfiler:

    @pytest.fixture
    def profiler(self):
        return HardwareProfilerAgent()

    @pytest.fixture
    def context(self):
        return AgentContext(target_gpus=["h100_sxm5", "mi300x"])

    def test_gpu_database_loaded(self):
        assert len(GPU_DATABASE) > 0
        assert "h100_sxm5" in GPU_DATABASE
        assert "mi300x" in GPU_DATABASE
        assert "a100_80gb" in GPU_DATABASE

    def test_get_gpu_spec(self):
        spec = get_gpu_spec("h100_sxm5")
        assert spec is not None
        assert spec.model_name == "H100 SXM5"
        assert spec.compute.fp16_tflops > 0
        assert spec.memory.bandwidth_gbps > 0

    def test_roofline_analysis(self):
        spec = get_gpu_spec("h100_sxm5")
        # 算术强度高的操作（GEMM）应该是计算受限
        assert not spec.is_memory_bound(flops=1e15, bytes_accessed=1e9)
        # 算术强度低的操作（element-wise）应该是内存受限
        assert spec.is_memory_bound(flops=1e9, bytes_accessed=1e9)

    @pytest.mark.asyncio
    async def test_profile_cluster(self, profiler, context):
        cluster_config = ClusterConfig(
            cluster_name="test_cluster",
            nodes=[
                {"gpu_model": "h100_sxm5", "num_gpus": 8},
                {"gpu_model": "mi300x", "num_gpus": 4},
            ],
            gpu_groups={
                "h100_sxm5": ["node0"],
                "mi300x": ["node1"],
            },
        )
        result = await profiler.run(context, cluster_config=cluster_config)
        assert result.success
        assert "h100_sxm5" in result.output
        assert "mi300x" in result.output

    @pytest.mark.asyncio
    async def test_keyword_matching(self, profiler, context):
        context.target_gpus = []
        result = await profiler.run(
            context,
            hardware_description="We have H100 and MI300X GPUs in our cluster"
        )
        assert result.success
        assert len(result.output) > 0

    def test_cluster_analysis_heterogeneous(self, profiler):
        from knowledge_base.hardware_specs.gpu_database import H100_SXM5, MI300X
        profiles = {"h100_sxm5": H100_SXM5, "mi300x": MI300X}
        analysis = profiler._analyze_cluster(profiles)
        assert analysis["is_heterogeneous"] is True
        assert len(analysis["vendors"]) == 2

    def test_cluster_analysis_homogeneous(self, profiler):
        from knowledge_base.hardware_specs.gpu_database import H100_SXM5
        profiles = {"h100_sxm5": H100_SXM5}
        analysis = profiler._analyze_cluster(profiles)
        assert len(analysis["vendors"]) == 1
