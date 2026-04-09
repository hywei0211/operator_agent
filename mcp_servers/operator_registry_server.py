"""
算子仓库 MCP Server
封装 OperatorRegistry，让 Agent 通过工具调用访问算子仓库
"""
from mcp_servers.base_server import BaseMCPServer, MCPTool
from operators.registry import get_registry, OperatorEntry


class OperatorRegistryMCPServer(BaseMCPServer):

    def __init__(self):
        super().__init__("operator_registry_server")
        self.registry = get_registry()

    def setup(self):
        self.register_tool(MCPTool(
            name="lookup_operator",
            description="查找已验证的算子实现",
            parameters={
                "operator_name": {"type": "string"},
                "gpu_model": {"type": "string"},
            },
            handler=self._lookup,
        ))
        self.register_tool(MCPTool(
            name="find_similar_operator",
            description="在相近GPU上查找同类算子实现（用于加速迁移）",
            parameters={
                "operator_name": {"type": "string"},
                "gpu_model": {"type": "string"},
            },
            handler=self._find_similar,
        ))
        self.register_tool(MCPTool(
            name="register_operator",
            description="将验证通过的算子写入仓库",
            parameters={"entry": {"type": "object"}},
            handler=self._register,
        ))
        self.register_tool(MCPTool(
            name="registry_stats",
            description="获取算子仓库统计信息",
            parameters={},
            handler=lambda: self.registry.stats(),
        ))

    def _lookup(self, operator_name: str, gpu_model: str):
        entry = self.registry.lookup(operator_name, gpu_model)
        return entry.__dict__ if entry else None

    def _find_similar(self, operator_name: str, gpu_model: str):
        entry = self.registry.find_similar(operator_name, gpu_model)
        return entry.__dict__ if entry else None

    def _register(self, entry: dict):
        self.registry.register(OperatorEntry(**entry))
        return {"status": "registered", "key": f"{entry['operator_name']}::{entry['gpu_model']}"}
