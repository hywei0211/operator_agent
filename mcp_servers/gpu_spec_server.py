"""
GPU 规格发现 MCP Server
工具链：本地数据库 → TechPowerUp 爬虫 → LLM 推断
"""
import json
import logging
import re
from typing import Optional

from mcp_servers.base_server import BaseMCPServer, MCPTool

logger = logging.getLogger(__name__)

# TechPowerUp 的 GPU 名称映射（用于构建 URL）
TECHPOWERUP_SLUG_MAP = {
    "h100 sxm5": "nvidia-h100-sxm5.d21021",
    "h100 pcie": "nvidia-h100-pcie.d21015",
    "a100 80gb": "nvidia-a100-sxm4-80-gb.d19902",
    "mi300x": "amd-instinct-mi300x.d25652",
    "mi250x": "amd-instinct-mi250x.d23851",
    "gaudi3": "intel-gaudi-3.d27015",
    "rtx 4090": "nvidia-geforce-rtx-4090.d23467",
    "rtx 4080": "nvidia-geforce-rtx-4080.d23575",
}

# 关键规格字段的提取规则
SPEC_FIELD_PATTERNS = {
    "compute_units":        r"(?:Shader Processors|Stream Processors|AI Cores?)\s*[:\s]+([0-9,]+)",
    "memory_gb":            r"Memory Size\s*[:\s]+([0-9.]+)\s*GB",
    "memory_bandwidth":     r"Memory Bandwidth\s*[:\s]+([0-9.]+)\s*GB/s",
    "fp16_tflops":          r"FP16\s*(?:\(Half\))?\s*[:\s]+([0-9.]+)\s*TFLOPS",
    "bf16_tflops":          r"BF16\s*[:\s]+([0-9.]+)\s*TFLOPS",
    "tdp":                  r"TDP\s*[:\s]+([0-9]+)\s*W",
    "base_clock":           r"Base Clock\s*[:\s]+([0-9]+)\s*MHz",
    "boost_clock":          r"Boost Clock\s*[:\s]+([0-9]+)\s*MHz",
}


class GPUSpecMCPServer(BaseMCPServer):
    """
    GPU 规格查询 MCP Server

    工具：
    - search_gpu_spec: 根据型号名搜索 GPU 规格
    - get_arch_details: 获取架构详情（内存层次、计算单元）
    - normalize_gpu_name: 标准化 GPU 名称
    """

    def __init__(self, llm_client=None):
        super().__init__("gpu_spec_server")
        self.llm_client = llm_client
        self._web_cache: dict[str, dict] = {}

    def setup(self):
        self.register_tool(MCPTool(
            name="search_gpu_spec",
            description="根据 GPU 型号名称搜索完整硬件规格",
            parameters={
                "model_name": {"type": "string", "description": "GPU 型号，如 'H100 SXM5' 或 'Ascend 910B'"}
            },
            handler=self._search_gpu_spec,
        ))
        self.register_tool(MCPTool(
            name="get_arch_details",
            description="获取 GPU 架构的片上内存层次和计算单元详情",
            parameters={
                "model_name": {"type": "string"},
                "vendor": {"type": "string", "description": "nvidia/amd/intel/huawei"}
            },
            handler=self._get_arch_details,
        ))
        self.register_tool(MCPTool(
            name="normalize_gpu_name",
            description="将用户输入的 GPU 名称标准化为系统内部 ID",
            parameters={"raw_name": {"type": "string"}},
            handler=self._normalize_name,
        ))

    async def _search_gpu_spec(self, model_name: str) -> Optional[dict]:
        """三层查询：本地数据库 → 网络爬虫 → LLM 推断"""
        normalized = self._normalize_name(model_name)

        # 1. 查本地数据库
        spec = self._query_local_db(normalized)
        if spec:
            spec["_source"] = "local_database"
            spec["_confidence"] = 1.0
            return spec

        # 2. 网络爬虫（TechPowerUp）
        spec = await self._scrape_techpowerup(normalized)
        if spec:
            spec["_source"] = "techpowerup_scrape"
            spec["_confidence"] = 0.9
            self._web_cache[normalized] = spec
            return spec

        # 3. LLM 推断
        if self.llm_client:
            spec = await self._llm_infer_spec(model_name)
            if spec:
                spec["_source"] = "llm_inference"
                spec["_confidence"] = 0.6
                return spec

        return None

    def _query_local_db(self, model_name: str) -> Optional[dict]:
        """查询本地 GPU 数据库"""
        from knowledge_base.hardware_specs.gpu_database import GPU_DATABASE
        name_lower = model_name.lower().replace(" ", "_").replace("-", "_")

        # 精确匹配
        for db_id, spec in GPU_DATABASE.items():
            if db_id == name_lower or spec.model_name.lower().replace(" ", "_") == name_lower:
                return {
                    "model_name": spec.model_name,
                    "vendor": spec.vendor.value,
                    "architecture": spec.architecture,
                    "compute_units": spec.compute_units,
                    "fp16_tflops": spec.compute.fp16_tflops,
                    "bf16_tflops": spec.compute.bf16_tflops,
                    "memory_gb": spec.memory.capacity_gb,
                    "memory_bandwidth_gbps": spec.memory.bandwidth_gbps,
                    "memory_type": spec.memory.memory_type,
                    "has_tensor_core": spec.compute.has_tensor_core,
                    "has_matrix_core": spec.compute.has_matrix_core,
                    "supported_backends": [b.value for b in spec.supported_backends],
                    "compute_capability": spec.compute_capability,
                }

        # 模糊匹配（关键词）
        for db_id, spec in GPU_DATABASE.items():
            if any(kw in name_lower for kw in db_id.split("_")):
                return self._query_local_db(db_id)
        return None

    async def _scrape_techpowerup(self, model_name: str) -> Optional[dict]:
        """
        从 TechPowerUp 爬取 GPU 规格
        真实实现：requests + BeautifulSoup 解析
        沙箱中：使用缓存数据模拟
        """
        try:
            import aiohttp
            from bs4 import BeautifulSoup

            # 构建 URL
            slug = self._find_slug(model_name)
            if not slug:
                return None

            url = f"https://www.techpowerup.com/gpu-specs/{slug}"
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    if resp.status != 200:
                        return None
                    html = await resp.text()

            soup = BeautifulSoup(html, "html.parser")
            return self._parse_techpowerup_html(soup, model_name)

        except ImportError:
            logger.debug("aiohttp/bs4 not installed, skipping web scrape")
            return None
        except Exception as e:
            logger.debug(f"Scraping failed for {model_name}: {e}")
            return None

    def _find_slug(self, model_name: str) -> Optional[str]:
        name_lower = model_name.lower()
        for key, slug in TECHPOWERUP_SLUG_MAP.items():
            if key in name_lower or name_lower in key:
                return slug
        return None

    def _parse_techpowerup_html(self, soup, model_name: str) -> dict:
        """解析 TechPowerUp 页面，提取关键规格"""
        result = {"model_name": model_name}
        text = soup.get_text()

        for field_name, pattern in SPEC_FIELD_PATTERNS.items():
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                val = match.group(1).replace(",", "")
                try:
                    result[field_name] = float(val) if "." in val else int(val)
                except ValueError:
                    result[field_name] = val

        # 推断厂商
        if "nvidia" in text.lower() or "geforce" in text.lower():
            result["vendor"] = "nvidia"
        elif "amd" in text.lower() or "radeon" in text.lower():
            result["vendor"] = "amd"
        elif "intel" in text.lower():
            result["vendor"] = "intel"

        return result

    async def _llm_infer_spec(self, model_name: str) -> Optional[dict]:
        """通过 LLM 推断未知 GPU 的规格"""
        prompt = f"""请提供以下 GPU 的硬件规格，以 JSON 格式返回：
GPU 型号: {model_name}

需要的字段（如不确定请估算并标注 estimated=true）：
- vendor (nvidia/amd/intel/huawei/other)
- architecture (架构名称)
- compute_units (计算单元数量，如SM/CU/AI Core)
- fp16_tflops (FP16算力)
- memory_gb (显存容量)
- memory_bandwidth_gbps (显存带宽)
- sram_per_unit_kb (每个计算单元的片上SRAM大小)
- supported_backends (cuda/hip/sycl/ascendc/triton 等)
- programming_model_notes (编程模型说明)

只返回 JSON，不要其他内容。"""

        try:
            response = await self.llm_client.chat(
                system="你是GPU硬件专家，对各厂商GPU规格有深入了解。",
                user=prompt,
                temperature=0.0,
            )
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                data["_llm_inferred"] = True
                return data
        except Exception as e:
            logger.warning(f"LLM spec inference failed: {e}")
        return None

    def _normalize_name(self, raw_name: str) -> str:
        """标准化 GPU 名称"""
        name = raw_name.lower().strip()
        replacements = {
            "nvidia ": "", "amd ": "", "intel ": "",
            "geforce ": "", "radeon ": "", "instinct ": "",
            "tesla ": "", " gpu": "", "-": " ",
        }
        for old, new in replacements.items():
            name = name.replace(old, new)
        return name.strip()

    async def _get_arch_details(self, model_name: str, vendor: str = "") -> dict:
        """获取架构级别的片上内存详情（Tiling 必需）"""
        spec = await self._search_gpu_spec(model_name)
        if not spec:
            return {}

        v = (vendor or spec.get("vendor", "")).lower()

        # 根据厂商提供典型的片上内存参数
        arch_defaults = {
            "nvidia": {
                "shared_memory_per_sm_kb": 164,
                "l1_cache_per_sm_kb": 32,
                "register_file_per_sm_kb": 256,
                "warp_size": 32,
                "max_threads_per_block": 1024,
                "memory_management": "automatic",  # GPU自动管理缓存
            },
            "amd": {
                "lds_per_cu_kb": 64,
                "l1_cache_per_cu_kb": 32,
                "wavefront_size": 64,
                "max_threads_per_block": 1024,
                "memory_management": "automatic",
            },
            "intel": {
                "slm_per_subslice_kb": 64,
                "l1_cache_kb": 128,
                "eu_simd_width": 8,
                "memory_management": "automatic",
            },
            "huawei": {
                "ub_size_kb": 256,
                "l1_buffer_kb": 1024,
                "l0a_kb": 64, "l0b_kb": 64, "l0c_kb": 256,
                "cube_tile_m": 16, "cube_tile_n": 16, "cube_tile_k": 16,
                "vector_width_fp16": 128,
                "memory_management": "manual",  # 必须手动 DataCopy！
            },
        }
        details = arch_defaults.get(v, {"memory_management": "unknown"})
        details.update(spec)
        return details
