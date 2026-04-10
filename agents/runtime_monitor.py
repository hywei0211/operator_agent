"""
Runtime Monitor Agent - 训练运行时监控
实时监控 GPU 利用率、loss 曲线、通信状态，检测异常并给出建议
"""
import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

from agents.base_agent import BaseAgent, AgentContext, AgentResult, AgentStatus
from agents.training_executor import TrainingJob

logger = logging.getLogger(__name__)


@dataclass
class GPUMetrics:
    """单个 GPU 的实时指标"""
    gpu_id: str
    gpu_model: str
    utilization_pct: float = 0.0
    memory_used_gb: float = 0.0
    memory_total_gb: float = 0.0
    temperature_c: float = 0.0
    power_w: float = 0.0
    timestamp: float = field(default_factory=time.time)


@dataclass
class TrainingMetrics:
    """训练整体指标快照"""
    step: int = 0
    loss: float = float('inf')
    throughput_samples_per_sec: float = 0.0
    gpu_metrics: list = field(default_factory=list)
    comm_overhead_pct: float = 0.0
    load_imbalance_pct: float = 0.0
    timestamp: float = field(default_factory=time.time)


@dataclass
class MonitorReport:
    """监控报告"""
    job_id: str
    status: str                     # healthy / warning / critical
    alerts: list = field(default_factory=list)
    recommendations: list = field(default_factory=list)
    latest_metrics: Optional[TrainingMetrics] = None
    metrics_history: list = field(default_factory=list)


class RuntimeMonitorAgent(BaseAgent):
    """
    运行时监控 Agent

    监控内容：
    1. GPU 利用率（是否持续低于 70%）
    2. 显存使用（是否接近上限）
    3. 通信耗时占比（是否超过 30%）
    4. 负载不均衡（最慢 GPU 拖累整体）
    5. Loss 曲线（是否发散或停止下降）
    6. 训练吞吐量

    检测到异常后触发：
    - 警告日志
    - 调整建议（给 Orchestrator 反馈）
    - 严重问题时触发 FaultRecoveryAgent
    """

    def __init__(self, llm_client=None, config: dict = None):
        super().__init__("RuntimeMonitorAgent", llm_client, config)
        cfg = config or {}
        self.poll_interval_sec = cfg.get("poll_interval_sec", 30)
        self.min_gpu_utilization = cfg.get("min_gpu_utilization", 0.70)
        self.max_comm_overhead = cfg.get("max_comm_overhead", 0.30)
        self.max_loss_stagnation_steps = cfg.get("max_loss_stagnation_steps", 100)
        self._loss_history: list[float] = []

    def get_system_prompt(self) -> str:
        return "你是分布式训练监控专家，能够诊断训练异常并给出优化建议。"

    async def run(self, context: AgentContext, **kwargs) -> AgentResult:
        self._start_timer()
        self.set_status(AgentStatus.RUNNING)

        job: Optional[TrainingJob] = (
            kwargs.get("job") or context.get_artifact("training_job")
        )
        monitor_once: bool = kwargs.get("monitor_once", True)

        if job is None:
            return self.failure_result("No training job to monitor")

        report = MonitorReport(job_id=job.job_id, status="healthy")

        try:
            if monitor_once:
                metrics = await self._collect_metrics(job)
                report.latest_metrics = metrics
                alerts = self._analyze_metrics(metrics, job)
                report.alerts = alerts
                report.recommendations = self._generate_recommendations(alerts, job)
                report.status = "critical" if any(a["level"] == "critical" for a in alerts) else \
                               "warning" if alerts else "healthy"
            else:
                # 持续监控（异步后台任务）
                await self._continuous_monitor(job, report)

            context.add_artifact("monitor_report", report)
            logger.info(f"[Monitor] Job {job.job_id}: status={report.status}, "
                       f"alerts={len(report.alerts)}")

            return self.success_result(
                output=report,
                metrics={"status": report.status, "alerts": len(report.alerts)}
            )
        except Exception as e:
            self.set_status(AgentStatus.FAILED)
            return self.failure_result(str(e))

    async def _collect_metrics(self, job: TrainingJob) -> TrainingMetrics:
        """收集当前训练指标（真实环境调用 nvidia-smi 等工具）"""
        gpu_metrics = []
        for gpu_group, nodes in job.cluster_config.gpu_groups.items():
            for node in nodes:
                # 真实实现：ssh到节点执行 nvidia-smi / rocm-smi / npu-smi
                gpu_metrics.append(GPUMetrics(
                    gpu_id=f"{node}_{gpu_group}",
                    gpu_model=gpu_group,
                    utilization_pct=self._query_gpu_utilization(gpu_group),
                    memory_used_gb=self._query_memory_used(gpu_group),
                    memory_total_gb=self._query_memory_total(gpu_group),
                ))

        return TrainingMetrics(
            step=0,
            gpu_metrics=gpu_metrics,
            comm_overhead_pct=0.15,  # 真实实现：解析 NCCL 日志
        )

    def _query_gpu_utilization(self, gpu_model: str) -> float:
        """查询 GPU 利用率（支持 nvidia-smi / rocm-smi / npu-smi）"""
        import subprocess

        # 尝试 NVIDIA GPU
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                return float(result.stdout.strip().split("\n")[0]) / 100
        except (FileNotFoundError, Exception):
            pass

        # 尝试 AMD GPU
        try:
            result = subprocess.run(
                ["rocm-smi", "--showuse", "--csv"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                lines = result.stdout.strip().split("\n")
                if len(lines) > 1:
                    return float(lines[1].split(",")[1]) / 100
        except (FileNotFoundError, Exception):
            pass

        # 尝试华为昇腾 NPU
        try:
            result = subprocess.run(
                ["npu-smi", "info", "-t", "usages"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0 and "AI Core" in result.stdout:
                import re
                m = re.search(r'AI Core.*?(\d+)%', result.stdout)
                if m:
                    return float(m.group(1)) / 100
        except (FileNotFoundError, Exception):
            pass

        logger.warning(f"[RuntimeMonitor] No GPU query tool available for {gpu_model}, utilization unknown")
        return -1.0  # 无法查询时返回 -1 表示未知

    def _query_memory_used(self, gpu_model: str) -> float:
        """查询 GPU 显存使用量 (GB)"""
        import subprocess
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                return float(result.stdout.strip().split("\n")[0]) / 1024  # MiB -> GiB
        except (FileNotFoundError, Exception):
            pass
        return 0.0

    def _query_memory_total(self, gpu_model: str) -> float:
        from knowledge_base.hardware_specs.gpu_database import get_gpu_spec
        spec = get_gpu_spec(gpu_model)
        return spec.memory.capacity_gb if spec else 80.0

    def _analyze_metrics(self, metrics: TrainingMetrics, job: TrainingJob) -> list[dict]:
        """分析指标，生成告警列表"""
        alerts = []

        # 检查 GPU 利用率
        for gm in metrics.gpu_metrics:
            if gm.utilization_pct < 0:
                # 利用率未知（无查询工具），报告 unknown 而不是误报
                alerts.append({
                    "level": "info",
                    "type": "utilization_unknown",
                    "message": f"GPU {gm.gpu_id} utilization unknown (no query tool available)",
                    "gpu": gm.gpu_id,
                })
            elif gm.utilization_pct < self.min_gpu_utilization:
                alerts.append({
                    "level": "warning",
                    "type": "low_utilization",
                    "message": f"GPU {gm.gpu_id} utilization {gm.utilization_pct:.0%} < {self.min_gpu_utilization:.0%}",
                    "gpu": gm.gpu_id,
                })
            mem_ratio = gm.memory_used_gb / max(gm.memory_total_gb, 1)
            if mem_ratio > 0.95:
                alerts.append({
                    "level": "critical",
                    "type": "oom_risk",
                    "message": f"GPU {gm.gpu_id} memory {mem_ratio:.0%} used, OOM risk!",
                    "gpu": gm.gpu_id,
                })

        # 检查通信开销
        if metrics.comm_overhead_pct > self.max_comm_overhead:
            alerts.append({
                "level": "warning",
                "type": "high_comm_overhead",
                "message": f"Communication overhead {metrics.comm_overhead_pct:.0%} > {self.max_comm_overhead:.0%}",
            })

        # 检查负载均衡（异构 GPU）
        if len(metrics.gpu_metrics) > 1:
            utils = [g.utilization_pct for g in metrics.gpu_metrics]
            imbalance = max(utils) - min(utils)
            if imbalance > 0.3:
                alerts.append({
                    "level": "warning",
                    "type": "load_imbalance",
                    "message": f"Load imbalance {imbalance:.0%} between GPUs",
                })

        # 检查 Loss 停滞
        if metrics.loss < float('inf'):
            self._loss_history.append(metrics.loss)
            if len(self._loss_history) >= self.max_loss_stagnation_steps:
                recent = self._loss_history[-self.max_loss_stagnation_steps:]
                if max(recent) - min(recent) < 1e-4:
                    alerts.append({
                        "level": "warning",
                        "type": "loss_stagnation",
                        "message": f"Loss has not improved in {self.max_loss_stagnation_steps} steps",
                    })

        return alerts

    def _generate_recommendations(self, alerts: list[dict], job: TrainingJob) -> list[str]:
        """根据告警生成优化建议"""
        recs = []
        alert_types = {a["type"] for a in alerts}

        if "low_utilization" in alert_types:
            recs.append("Increase batch size or reduce data loading bottleneck")
            recs.append("Check if custom operators have correctness issues causing slow execution")

        if "high_comm_overhead" in alert_types:
            recs.append("Enable gradient compression (FP16 communication)")
            recs.append("Consider gradient accumulation to reduce AllReduce frequency")
            recs.append("For heterogeneous cluster: switch to pipeline parallel to reduce comm")

        if "load_imbalance" in alert_types:
            recs.append("Rebalance batch distribution: assign more work to faster GPUs")
            recs.append("Consider pipeline parallel: put lighter layers on slower GPU")

        if "oom_risk" in alert_types:
            recs.append("Enable gradient checkpointing to reduce activation memory")
            recs.append("Reduce batch size or sequence length")

        return recs

    async def _continuous_monitor(self, job: TrainingJob, report: MonitorReport):
        """持续监控（每 N 秒采集一次，直到训练结束）"""
        logger.info(f"[Monitor] Starting continuous monitoring for job {job.job_id}")
        while job.status == "running":
            metrics = await self._collect_metrics(job)
            report.metrics_history.append(metrics)
            report.latest_metrics = metrics
            alerts = self._analyze_metrics(metrics, job)
            if alerts:
                report.alerts.extend(alerts)
                logger.warning(f"[Monitor] Alerts: {[a['message'] for a in alerts]}")
            await asyncio.sleep(self.poll_interval_sec)
