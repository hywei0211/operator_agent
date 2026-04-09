"""
MCP Server 基类
实现轻量级的工具调用协议，兼容 MCP 标准接口
每个 Server 对外暴露若干 Tool，Agent 通过 MCPClient 调用
"""
import asyncio
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger(__name__)


@dataclass
class MCPTool:
    """MCP 工具定义"""
    name: str
    description: str
    parameters: dict            # JSON Schema 格式的参数描述
    handler: Callable           # 实际执行函数


@dataclass
class MCPResponse:
    """MCP 工具调用响应"""
    success: bool
    data: Any = None
    error: str = ""
    source: str = ""            # 数据来源标注（数据库/网络/LLM）
    confidence: float = 1.0     # 数据可信度 0~1


class BaseMCPServer(ABC):
    """MCP Server 基类"""

    def __init__(self, name: str):
        self.name = name
        self._tools: dict[str, MCPTool] = {}

    def register_tool(self, tool: MCPTool):
        self._tools[tool.name] = tool
        logger.debug(f"[{self.name}] Registered tool: {tool.name}")

    async def call(self, tool_name: str, **kwargs) -> MCPResponse:
        tool = self._tools.get(tool_name)
        if tool is None:
            return MCPResponse(success=False, error=f"Tool '{tool_name}' not found in {self.name}")
        try:
            if asyncio.iscoroutinefunction(tool.handler):
                result = await tool.handler(**kwargs)
            else:
                result = tool.handler(**kwargs)
            return MCPResponse(success=True, data=result, source=self.name)
        except Exception as e:
            logger.error(f"[{self.name}] Tool {tool_name} failed: {e}")
            return MCPResponse(success=False, error=str(e))

    def list_tools(self) -> list[dict]:
        return [{"name": t.name, "description": t.description} for t in self._tools.values()]

    @abstractmethod
    def setup(self):
        """注册所有工具"""
        raise NotImplementedError


class MCPClient:
    """
    MCP 客户端 - Agent 使用此类调用各 Server 的工具
    支持同时管理多个 Server
    """

    def __init__(self):
        self._servers: dict[str, BaseMCPServer] = {}

    def register_server(self, server: BaseMCPServer):
        server.setup()
        self._servers[server.name] = server
        logger.info(f"[MCPClient] Registered server: {server.name} "
                    f"with tools: {[t['name'] for t in server.list_tools()]}")

    async def call(self, server_name: str, tool_name: str, **kwargs) -> MCPResponse:
        server = self._servers.get(server_name)
        if server is None:
            return MCPResponse(success=False, error=f"Server '{server_name}' not registered")
        return await server.call(tool_name, **kwargs)

    async def call_with_fallback(
        self,
        calls: list[tuple[str, str, dict]],  # [(server, tool, kwargs), ...]
    ) -> MCPResponse:
        """按优先级依次尝试调用，返回第一个成功的结果"""
        last_error = ""
        for server_name, tool_name, kwargs in calls:
            resp = await self.call(server_name, tool_name, **kwargs)
            if resp.success and resp.data:
                return resp
            last_error = resp.error
        return MCPResponse(success=False, error=f"All fallbacks failed. Last: {last_error}")
