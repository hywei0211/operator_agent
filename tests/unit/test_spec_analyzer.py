"""
算子规格解析Agent单元测试
"""
import pytest

from agents.spec_analyzer import OperatorSpecAgent, OPERATOR_TEMPLATES
from agents.base_agent import AgentContext
from models.operator_ir import OperatorCategory


class TestSpecAnalyzer:

    @pytest.fixture
    def analyzer(self):
        return OperatorSpecAgent()

    @pytest.fixture
    def context(self):
        return AgentContext()

    def test_templates_loaded(self):
        assert "flash_attention" in OPERATOR_TEMPLATES
        assert "rmsnorm" in OPERATOR_TEMPLATES
        assert "gelu" in OPERATOR_TEMPLATES

    def test_keyword_match_flash_attention(self, analyzer):
        assert analyzer._match_template("FlashAttention v2") == "flash_attention"
        assert analyzer._match_template("multi-head attention") == "flash_attention"
        assert analyzer._match_template("FLASH_ATTENTION") == "flash_attention"

    def test_keyword_match_rmsnorm(self, analyzer):
        assert analyzer._match_template("RMSNorm layer") == "rmsnorm"
        assert analyzer._match_template("rms norm normalization") == "rmsnorm"

    def test_keyword_match_gelu(self, analyzer):
        assert analyzer._match_template("GELU activation function") == "gelu"

    def test_keyword_match_moe(self, analyzer):
        assert analyzer._match_template("Mixture of Experts routing") == "fused_moe"
        assert analyzer._match_template("MoE layer") == "fused_moe"

    def test_build_from_template(self, analyzer):
        ir = analyzer._build_from_template("flash_attention", "FlashAttention v2")
        assert ir.name == "flash_attention"
        assert ir.category == OperatorCategory.ATTENTION
        assert len(ir.inputs) == 3   # Q, K, V
        assert len(ir.outputs) == 1  # O
        assert ir.math_description != ""
        assert ir.flops_formula != ""

    def test_ir_validation_complete(self, analyzer):
        ir = analyzer._build_from_template("rmsnorm", "RMSNorm")
        issues = analyzer._validate_ir(ir)
        assert len(issues) == 0

    @pytest.mark.asyncio
    async def test_run_with_template_match(self, analyzer, context):
        result = await analyzer.run(context, request="FlashAttention v2 with causal masking")
        assert result.success
        assert result.output.name == "flash_attention"
        assert result.output.category == OperatorCategory.ATTENTION

    @pytest.mark.asyncio
    async def test_run_no_request(self, analyzer, context):
        result = await analyzer.run(context, request="")
        assert not result.success
