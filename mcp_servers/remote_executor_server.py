"""
远程执行器 MCP Server
负责在目标 GPU 环境中编译和运行生成的算子
支持：本地执行 / Docker 容器 / 远程 SSH 机器
"""
import asyncio
import hashlib
import json
import logging
import os
import subprocess
import tempfile
from dataclasses import dataclass
from typing import Optional

from mcp_servers.base_server import BaseMCPServer, MCPTool

logger = logging.getLogger(__name__)


@dataclass
class ExecutionResult:
    success: bool
    stdout: str = ""
    stderr: str = ""
    exit_code: int = -1
    elapsed_ms: float = 0.0


class RemoteExecutorMCPServer(BaseMCPServer):
    """
    远程执行器 MCP Server

    执行策略：
    1. 本地执行（有 GPU 且 SDK 已安装）
    2. Docker 容器（SDK 打包在镜像中）
    3. 远程 SSH（目标机器有 GPU）
    4. 模拟执行（无硬件时做静态分析）
    """

    # SDK → Docker 镜像映射
    SDK_DOCKER_IMAGES = {
        "cuda":     "nvcr.io/nvidia/cuda:12.4-devel-ubuntu22.04",
        "hip":      "rocm/dev-ubuntu-22.04:6.0",
        "ascendc":  "ascendhub.huawei.com/public-ascendhub/ascend-toolkit:latest",
        "sycl":     "intel/oneapi-hpckit:latest",
        "triton":   "nvcr.io/nvidia/cuda:12.4-devel-ubuntu22.04",
    }

    def __init__(self, ssh_hosts: dict = None, use_docker: bool = True):
        super().__init__("remote_executor_server")
        self.ssh_hosts = ssh_hosts or {}     # {"cuda": "user@host", "hip": "user@amd-host"}
        self.use_docker = use_docker
        self._compile_cache: dict[str, bool] = {}

    def setup(self):
        self.register_tool(MCPTool(
            name="compile_kernel",
            description="编译生成的 GPU 内核代码",
            parameters={
                "source_code": {"type": "string"},
                "sdk": {"type": "string"},
                "build_flags": {"type": "array"},
                "kernel_name": {"type": "string"},
            },
            handler=self._compile_kernel,
        ))
        self.register_tool(MCPTool(
            name="run_correctness_test",
            description="运行算子正确性测试，与参考实现对比",
            parameters={
                "kernel_name": {"type": "string"},
                "sdk": {"type": "string"},
                "reference_impl": {"type": "string"},
                "test_shapes": {"type": "array"},
            },
            handler=self._run_correctness_test,
        ))
        self.register_tool(MCPTool(
            name="run_benchmark",
            description="运行性能基准测试",
            parameters={
                "kernel_name": {"type": "string"},
                "sdk": {"type": "string"},
                "warmup_iters": {"type": "integer"},
                "benchmark_iters": {"type": "integer"},
            },
            handler=self._run_benchmark,
        ))
        self.register_tool(MCPTool(
            name="check_environment",
            description="检查目标执行环境是否可用",
            parameters={"sdk": {"type": "string"}},
            handler=self._check_environment,
        ))

    async def _compile_kernel(
        self,
        source_code: str,
        sdk: str,
        build_flags: list = None,
        kernel_name: str = "kernel",
    ) -> dict:
        # 检查缓存
        code_hash = hashlib.md5(source_code.encode()).hexdigest()[:8]
        cache_key = f"{sdk}:{code_hash}"
        if cache_key in self._compile_cache:
            return {"success": self._compile_cache[cache_key], "cached": True}

        env_available = await self._check_environment(sdk)
        if not env_available.get("available"):
            # 降级：做静态语法检查
            result = self._static_syntax_check(source_code, sdk)
            return result

        # 根据环境选择编译方式
        if self.ssh_hosts.get(sdk):
            result = await self._compile_via_ssh(source_code, sdk, build_flags or [], kernel_name)
        elif self.use_docker:
            result = await self._compile_via_docker(source_code, sdk, build_flags or [], kernel_name)
        else:
            result = await self._compile_local(source_code, sdk, build_flags or [], kernel_name)

        self._compile_cache[cache_key] = result.get("success", False)
        return result

    async def _compile_via_docker(
        self, source_code: str, sdk: str, build_flags: list, kernel_name: str
    ) -> dict:
        image = self.SDK_DOCKER_IMAGES.get(sdk)
        if not image:
            return {"success": False, "error": f"No Docker image for SDK: {sdk}"}

        ext = {"cuda": ".cu", "hip": ".cpp", "triton": ".py", "sycl": ".cpp", "ascendc": ".cpp"}.get(sdk, ".cpp")
        compiler = {"cuda": "nvcc", "hip": "hipcc", "sycl": "icpx", "ascendc": "atc"}.get(sdk, "g++")

        with tempfile.TemporaryDirectory() as tmpdir:
            src_path = os.path.join(tmpdir, f"{kernel_name}{ext}")
            out_path = os.path.join(tmpdir, f"{kernel_name}.o")
            with open(src_path, "w") as f:
                f.write(source_code)

            flags_str = " ".join(build_flags) if build_flags else "-O2"
            if sdk == "cuda":
                cmd = f"{compiler} {flags_str} -c {src_path} -o {out_path}"
            elif sdk == "triton":
                cmd = f"python -c 'import triton; exec(open(\"{src_path}\").read())'"
            else:
                cmd = f"{compiler} {flags_str} -c {src_path} -o {out_path}"

            docker_cmd = [
                "docker", "run", "--rm",
                "-v", f"{tmpdir}:/workspace",
                "-w", "/workspace",
                image, "bash", "-c", cmd
            ]
            try:
                proc = await asyncio.create_subprocess_exec(
                    *docker_cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=120)
                success = proc.returncode == 0
                return {
                    "success": success,
                    "stdout": stdout.decode(),
                    "stderr": stderr.decode(),
                    "method": "docker",
                }
            except asyncio.TimeoutError:
                return {"success": False, "error": "Compilation timeout (120s)"}
            except FileNotFoundError:
                return {"success": False, "error": "Docker not available, falling back to static check",
                        "fallback": self._static_syntax_check(source_code, sdk)}

    async def _compile_via_ssh(
        self, source_code: str, sdk: str, build_flags: list, kernel_name: str
    ) -> dict:
        host = self.ssh_hosts[sdk]
        ext = {"cuda": ".cu", "hip": ".cpp"}.get(sdk, ".cpp")
        remote_path = f"/tmp/{kernel_name}_{id(source_code)}{ext}"

        try:
            # 上传源文件
            echo_cmd = f'echo {repr(source_code)} > {remote_path}'
            proc = await asyncio.create_subprocess_exec(
                "ssh", host, echo_cmd,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            await proc.communicate()

            # 编译
            compiler = {"cuda": "nvcc", "hip": "hipcc"}.get(sdk, "g++")
            compile_cmd = f"{compiler} -c {remote_path} -o {remote_path}.o {' '.join(build_flags)}"
            proc2 = await asyncio.create_subprocess_exec(
                "ssh", host, compile_cmd,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await asyncio.wait_for(proc2.communicate(), timeout=60)
            return {
                "success": proc2.returncode == 0,
                "stdout": stdout.decode(),
                "stderr": stderr.decode(),
                "method": "ssh",
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    async def _compile_local(
        self, source_code: str, sdk: str, build_flags: list, kernel_name: str
    ) -> dict:
        ext = {"cuda": ".cu", "hip": ".cpp", "triton": ".py"}.get(sdk, ".cpp")
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False, mode="w") as f:
            f.write(source_code)
            tmp_path = f.name
        try:
            compiler = {"cuda": "nvcc", "hip": "hipcc", "sycl": "icpx"}.get(sdk)
            if not compiler:
                os.unlink(tmp_path)
                return self._static_syntax_check(source_code, sdk)
            cmd = [compiler] + build_flags + ["-c", tmp_path, "-o", tmp_path + ".o"]
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            return {"success": proc.returncode == 0, "stdout": proc.stdout, "stderr": proc.stderr, "method": "local"}
        except Exception as e:
            return {"success": False, "error": str(e)}
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def _static_syntax_check(self, source_code: str, sdk: str) -> dict:
        """无硬件时的静态语法检查"""
        errors = []
        code = source_code

        if sdk in ("cuda", "hip", "ascendc"):
            if code.count("{") != code.count("}"):
                errors.append("Mismatched braces { }")
            if sdk == "cuda" and "__global__" not in code:
                errors.append("Missing __global__ kernel function")
            if sdk == "hip" and "__global__" not in code:
                errors.append("Missing __global__ kernel function")
            if sdk == "ascendc" and "__aicore__" not in code:
                errors.append("Missing __aicore__ kernel function")
        elif sdk == "triton":
            if "@triton.jit" not in code:
                errors.append("Missing @triton.jit decorator")
            if "import triton" not in code:
                errors.append("Missing triton import")

        return {
            "success": len(errors) == 0,
            "errors": errors,
            "method": "static_analysis",
            "note": "No hardware available, static check only",
        }

    async def _run_correctness_test(
        self,
        kernel_name: str,
        sdk: str,
        reference_impl: str = "",
        test_shapes: list = None,
    ) -> dict:
        env = await self._check_environment(sdk)
        if not env.get("available"):
            return {
                "status": "skipped",
                "reason": f"No {sdk} hardware available",
                "suggestion": "Run on target hardware to verify correctness",
            }
        return {
            "status": "pending",
            "message": "Correctness test requires actual hardware execution",
            "test_shapes": test_shapes or [[1, 128, 4096], [4, 512, 4096]],
        }

    async def _run_benchmark(
        self,
        kernel_name: str,
        sdk: str,
        warmup_iters: int = 20,
        benchmark_iters: int = 100,
    ) -> dict:
        env = await self._check_environment(sdk)
        if not env.get("available"):
            return {"status": "skipped", "reason": f"No {sdk} hardware available"}
        return {
            "status": "pending",
            "message": "Benchmark requires actual hardware",
            "config": {"warmup": warmup_iters, "iters": benchmark_iters},
        }

    async def _check_environment(self, sdk: str) -> dict:
        """检查本地是否有对应 SDK 环境"""
        checks = {
            "cuda": ["nvcc", "--version"],
            "hip": ["hipcc", "--version"],
            "sycl": ["icpx", "--version"],
            "triton": ["python", "-c", "import triton"],
            "ascendc": ["atc", "--version"],
        }
        cmd = checks.get(sdk.lower())
        if not cmd:
            return {"available": False, "reason": f"Unknown SDK: {sdk}"}
        try:
            proc = subprocess.run(cmd, capture_output=True, timeout=5)
            return {"available": proc.returncode == 0, "sdk": sdk}
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return {"available": False, "sdk": sdk, "reason": f"{cmd[0]} not found"}
