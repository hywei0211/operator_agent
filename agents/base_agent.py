"""
Agent基类 - 所有专业Agent的抽象基类
定义统一的接口、状态管理和通信机制
"""
import asyncio
import logging
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


logger = logging.getLogger(__name__)


class AgentStatus(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    WAITING = "waiting"      # 等待子Agent返回
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class MessageType(str, Enum):
    REQUEST = "request"
    RESPONSE = "response"
    FEEDBACK = "feedback"
    ERROR = "error"
    STATUS = "status"


@dataclass
class AgentMessage:
    """Agent间通信的消息结构"""
    msg_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    msg_type: MessageType = MessageType.REQUEST
    sender: str = ""
    receiver: str = ""
    content: Any = None
    metadata: dict = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    parent_msg_id: Optional[str] = None     # 用于追踪请求-响应链


@dataclass
class AgentContext:
    """Agent执行上下文 - 在整个工作流中传递的共享状态"""
    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    operator_name: str = ""
    target_gpus: list[str] = field(default_factory=list)
    conversation_history: list[dict] = field(default_factory=list)
    shared_artifacts: dict[str, Any] = field(default_factory=dict)  # 共享产物（IR、代码等）
    iteration_count: int = 0
    max_iterations: int = 5
    metadata: dict = field(default_factory=dict)

    def add_artifact(self, key: str, value: Any):
        self.shared_artifacts[key] = value
        logger.debug(f"[Context] Added artifact: {key}")

    def get_artifact(self, key: str, default=None) -> Any:
        return self.shared_artifacts.get(key, default)


@dataclass
class AgentResult:
    """Agent执行结果的统一封装"""
    success: bool
    agent_name: str
    output: Any = None
    error: Optional[str] = None
    metrics: dict = field(default_factory=dict)     # 执行指标
    next_action: Optional[str] = None              # 建议的下一步动作
    elapsed_seconds: float = 0.0


class BaseAgent(ABC):
    """
    所有Agent的抽象基类

    设计原则：
    - 每个Agent专注于单一职责
    - 通过AgentContext共享状态
    - 通过AgentMessage异步通信
    - 支持LLM调用和工具调用
    """

    def __init__(self, name: str, llm_client=None, config: dict = None):
        self.name = name
        self.llm_client = llm_client
        self.config = config or {}
        self.status = AgentStatus.IDLE
        self.message_queue: list[AgentMessage] = []
        self._start_time: Optional[float] = None
        logger.info(f"[{self.name}] Agent initialized")

    @abstractmethod
    async def run(self, context: AgentContext, **kwargs) -> AgentResult:
        """
        Agent的主执行逻辑

        Args:
            context: 共享执行上下文
            **kwargs: Agent特定的额外参数

        Returns:
            AgentResult: 执行结果
        """
        raise NotImplementedError

    @abstractmethod
    def get_system_prompt(self) -> str:
        """返回该Agent的系统提示词，定义其角色和能力"""
        raise NotImplementedError

    async def call_llm(
        self,
        user_message: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: int = 4096,
    ) -> str:
        """统一的LLM调用接口，带指数退避重试"""
        if self.llm_client is None:
            raise RuntimeError(f"[{self.name}] LLM client not configured")

        prompt = system_prompt or self.get_system_prompt()
        max_retries = 2
        if self.config and isinstance(self.config, dict):
            max_retries = self.config.get("max_retry_on_failure", 2)
        base_delay = 2.0

        logger.debug(f"[{self.name}] Calling LLM, message length: {len(user_message)}")

        last_error = None
        for attempt in range(max_retries + 1):
            start = time.time()
            try:
                response = await self.llm_client.chat(
                    system=prompt,
                    user=user_message,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                elapsed = time.time() - start
                logger.debug(f"[{self.name}] LLM response in {elapsed:.2f}s (attempt {attempt+1})")
                return response
            except Exception as e:
                last_error = e
                elapsed = time.time() - start
                if attempt < max_retries:
                    delay = base_delay * (2 ** attempt)
                    logger.warning(
                        f"[{self.name}] LLM call failed (attempt {attempt+1}/{max_retries+1}, "
                        f"{elapsed:.1f}s): {e}. Retrying in {delay:.0f}s..."
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(
                        f"[{self.name}] LLM call failed after {max_retries+1} attempts: {e}"
                    )

        raise last_error

    def send_message(self, receiver: str, content: Any, msg_type: MessageType = MessageType.REQUEST) -> AgentMessage:
        msg = AgentMessage(
            msg_type=msg_type,
            sender=self.name,
            receiver=receiver,
            content=content,
        )
        logger.debug(f"[{self.name}] -> [{receiver}]: {msg_type.value}")
        return msg

    def set_status(self, status: AgentStatus):
        old_status = self.status
        self.status = status
        logger.info(f"[{self.name}] Status: {old_status.value} -> {status.value}")

    def _start_timer(self):
        self._start_time = time.time()

    def _elapsed(self) -> float:
        if self._start_time is None:
            return 0.0
        return time.time() - self._start_time

    def success_result(self, output: Any, metrics: dict = None) -> AgentResult:
        return AgentResult(
            success=True,
            agent_name=self.name,
            output=output,
            metrics=metrics or {},
            elapsed_seconds=self._elapsed(),
        )

    def failure_result(self, error: str, output: Any = None) -> AgentResult:
        logger.error(f"[{self.name}] Failed: {error}")
        return AgentResult(
            success=False,
            agent_name=self.name,
            output=output,
            error=error,
            elapsed_seconds=self._elapsed(),
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name}, status={self.status.value})"
