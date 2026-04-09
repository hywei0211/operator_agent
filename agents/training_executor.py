"""
Training Executor Agent - 训练执行器
负责为异构 GPU 集群启动分布式训练，注入生成的自定义算子
"""
import asyncio
import logging
import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from agents.base_agent import BaseAgent, AgentContext, AgentResult, AgentStatus
from agents.training_analyst import TrainingPlan
from models.operator_ir import ClusterConfig, GeneratedKernel
from operators.registry import get_registry

logger = logging.getLogger(__name__)


@dataclass
class TrainingJob:
    """训练任务描述"""
    job_id: str
    training_code: str
    cluster_config: ClusterConfig
    training_plan: TrainingPlan
    injected_kernels: dict[str, str] = field(default_factory=dict)  # op_name → kernel_path
    output_dir: str = "./output/training"
    pid: Optional[int] = None
    status: str = "pending"     # pending/running/completed/failed


class TrainingExecutorAgent(BaseAgent):
    """
    训练执行 Agent

    职责：
    1. 从 OperatorRegistry 加载已验证的算子
    2. 生成 PyTorch 自定义算子注册代码（torch.library）
    3. 修改训练脚本，注入生成的算子
    4. 根据分布式方案生成启动命令（torchrun / deepspeed）
    5. 启动训练进程
    6. 监控进程状态

    异构 GPU 的关键：
    - 每种 GPU 加载自己对应的算子（CUDA/HIP/AscendC）
    - 用 RANK 环境变量区分不同 GPU 的进程
    - 通信通过 UCX/MPI 统一管理
    """

    def __init__(self, llm_client=None, config: dict = None):
        super().__init__("TrainingExecutorAgent", llm_client, config)
        self.registry = get_registry()

    def get_system_prompt(self) -> str:
        return "你是分布式训练专家，熟悉 PyTorch DDP、DeepSpeed 和异构GPU训练。"

    async def run(self, context: AgentContext, **kwargs) -> AgentResult:
        self._start_timer()
        self.set_status(AgentStatus.RUNNING)

        training_code: str = kwargs.get("training_code", "")
        cluster_config: Optional[ClusterConfig] = kwargs.get("cluster_config")
        training_plan: Optional[TrainingPlan] = (
            kwargs.get("training_plan") or context.get_artifact("training_plan")
        )
        kernels: dict = context.get_artifact("optimized_kernels") or {}

        if not training_code:
            return self.failure_result("No training code provided")

        try:
            job_id = f"job_{int(self._elapsed() * 1000)}"
            job = TrainingJob(
                job_id=job_id,
                training_code=training_code,
                cluster_config=cluster_config or ClusterConfig("default", [], {}),
                training_plan=training_plan or TrainingPlan(),
            )

            # 1. 保存算子代码到临时目录
            kernel_dir = self._save_kernels(kernels, job_id)
            job.injected_kernels = kernel_dir

            # 2. 生成算子注册代码
            registration_code = self._generate_op_registration(kernels, training_plan)

            # 3. 生成修改后的训练脚本（注入算子）
            patched_code = self._patch_training_code(training_code, registration_code, kernels)

            # 4. 生成启动命令
            launch_cmd, env = self._generate_launch_command(job, cluster_config)

            # 5. 保存并启动
            output_dir = Path(f"./output/{job_id}")
            output_dir.mkdir(parents=True, exist_ok=True)

            script_path = output_dir / "train_patched.py"
            script_path.write_text(patched_code)

            launch_script = output_dir / "launch.sh"
            launch_script.write_text(f"#!/bin/bash\n{launch_cmd}\n")
            launch_script.chmod(0o755)

            logger.info(f"[TrainingExecutor] Job {job_id} ready to launch")
            logger.info(f"[TrainingExecutor] Script: {script_path}")
            logger.info(f"[TrainingExecutor] Launch: {launch_cmd[:100]}...")

            # 实际启动（如果配置了 dry_run=False）
            if not (self.config or {}).get("dry_run", True):
                proc = await self._launch_process(str(launch_script), env)
                job.pid = proc.pid
                job.status = "running"
            else:
                job.status = "dry_run_ready"
                logger.info(f"[TrainingExecutor] dry_run=True, not actually launching")

            context.add_artifact("training_job", job)
            return self.success_result(
                output=job,
                metrics={
                    "job_id": job_id,
                    "status": job.status,
                    "script_path": str(script_path),
                    "launch_cmd": launch_cmd,
                    "kernels_injected": len(kernels),
                }
            )
        except Exception as e:
            self.set_status(AgentStatus.FAILED)
            return self.failure_result(str(e))

    def _save_kernels(self, kernels: dict, job_id: str) -> dict[str, str]:
        """将生成的内核文件保存到磁盘"""
        kernel_dir = Path(f"./output/{job_id}/kernels")
        kernel_dir.mkdir(parents=True, exist_ok=True)
        saved = {}
        for gpu_id, kernel in kernels.items():
            if not isinstance(kernel, GeneratedKernel):
                continue
            ext = {"cuda": ".cu", "hip": ".hip.cpp", "triton": ".py",
                   "sycl": ".sycl.cpp", "ascendc": ".cpp"}.get(kernel.backend, ".cpp")
            fname = kernel_dir / f"{kernel.operator_name}_{gpu_id}{ext}"
            fname.write_text(
                (kernel.header_code + "\n\n" if kernel.header_code else "") + kernel.source_code
            )
            saved[f"{kernel.operator_name}_{gpu_id}"] = str(fname)

        # 尝试编译 CUDA/HIP kernel 为 .so
        self._compile_kernels(saved, kernels, kernel_dir)
        return saved

    def _compile_kernels(self, saved: dict, kernels: dict, kernel_dir: Path):
        """编译保存的 kernel 源文件为共享库 (.so)"""
        import subprocess
        for key, src_path in saved.items():
            kernel = None
            for gpu_id, k in kernels.items():
                if isinstance(k, GeneratedKernel) and f"{k.operator_name}_{gpu_id}" == key:
                    kernel = k
                    break
            if kernel is None:
                continue

            so_path = str(Path(src_path).with_suffix(".so"))
            try:
                if kernel.backend == "cuda":
                    cmd = ["nvcc", "--shared", "-o", so_path, src_path,
                           "-Xcompiler", "-fPIC"] + kernel.build_flags
                elif kernel.backend == "hip":
                    cmd = ["hipcc", "--shared", "-o", so_path, src_path, "-fPIC"]
                elif kernel.backend == "triton":
                    continue  # Triton is Python, no compilation needed
                else:
                    continue

                result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
                if result.returncode == 0:
                    logger.info(f"[TrainingExecutor] Compiled {key} -> {so_path}")
                else:
                    logger.warning(f"[TrainingExecutor] Compile failed for {key}: {result.stderr[:200]}")
            except FileNotFoundError:
                logger.debug(f"[TrainingExecutor] Compiler not found for {kernel.backend}, skipping {key}")
            except Exception as e:
                logger.warning(f"[TrainingExecutor] Compile error for {key}: {e}")
        return saved

    def _generate_op_registration(
        self, kernels: dict, plan: Optional[TrainingPlan]
    ) -> str:
        """生成 PyTorch 自定义算子注册代码"""
        lines = [
            "# ── Auto-generated operator registration ──────────────────",
            "import torch",
            "import os",
            "import ctypes",
            "",
            "_DEVICE_TYPE = None",
            "",
            "def _get_device_type():",
            "    global _DEVICE_TYPE",
            "    if _DEVICE_TYPE is None:",
            "        if torch.cuda.is_available():",
            "            _DEVICE_TYPE = 'cuda'",
            "        elif hasattr(torch, 'hip') and torch.hip.is_available():",
            "            _DEVICE_TYPE = 'hip'",
            "        else:",
            "            _DEVICE_TYPE = 'cpu'",
            "    return _DEVICE_TYPE",
            "",
        ]

        if kernels:
            lines += [
                "# Load compiled kernels based on device type",
                "def _load_kernels():",
                "    device = _get_device_type()",
                "    kernel_dir = os.path.join(os.path.dirname(__file__), 'kernels')",
                "    # Kernels are pre-compiled, load the .so file",
                "    so_path = os.path.join(kernel_dir, f'ops_{device}.so')",
                "    if os.path.exists(so_path):",
                "        torch.ops.load_library(so_path)",
                "        print(f'[OperatorAgent] Loaded custom kernels: {so_path}')",
                "    else:",
                "        print(f'[OperatorAgent] No compiled kernels found at {so_path}, using PyTorch defaults')",
                "",
                "_load_kernels()",
            ]

        lines += ["", "# ─────────────────────────────────────────────────────────"]
        return "\n".join(lines)

    def _patch_training_code(
        self, code: str, registration_code: str, kernels: dict
    ) -> str:
        """在训练代码开头注入算子注册代码"""
        header = f"""#!/usr/bin/env python3
# Auto-patched by Operator Agent - custom kernels injected

{registration_code}

# ── Original training code ──────────────────────────────────
"""
        return header + code

    def _generate_launch_command(
        self, job: TrainingJob, cluster_config: Optional[ClusterConfig]
    ) -> tuple[str, dict]:
        """生成训练启动命令（支持单机多卡和多机多卡）"""
        env = os.environ.copy()
        script = f"./output/{job.job_id}/train_patched.py"

        if cluster_config is None or cluster_config.total_gpus() <= 1:
            return f"python {script}", env

        total_gpus = cluster_config.total_gpus()
        num_nodes = len(cluster_config.nodes)
        comm_backend = getattr(cluster_config, 'communication_backend', 'nccl')

        # 设置通信后端
        if comm_backend == "ucc":
            env["TORCH_DISTRIBUTED_BACKEND"] = "ucc"
        elif comm_backend in ("nccl", "rccl"):
            env["NCCL_DEBUG"] = "INFO"

        if num_nodes == 1:
            # 单机多卡：torchrun
            cmd = (
                f"torchrun --nproc_per_node={total_gpus} "
                f"--master_port=29500 "
                f"{script}"
            )
        else:
            # 多机：torchrun + rendezvous
            cmd = (
                f"torchrun "
                f"--nnodes={num_nodes} "
                f"--nproc_per_node={total_gpus // num_nodes} "
                f"--rdzv_backend=c10d "
                f"--rdzv_endpoint=master:29500 "
                f"{script}"
            )

        return cmd, env

    async def _launch_process(self, launch_script: str, env: dict) -> asyncio.subprocess.Process:
        proc = await asyncio.create_subprocess_exec(
            "bash", launch_script,
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        logger.info(f"[TrainingExecutor] Launched PID={proc.pid}")
        return proc
