"""
LLM 客户端封装
支持：OpenAI GPT-4 / Anthropic Claude / 阿里云 Qwen (DashScope) / Mock
内置 LLM 响应缓存（content-addressed，避免重复调用浪费时间和金钱）
"""
import asyncio
import hashlib
import json as _json
import logging
import os
import re
import sqlite3
import time
from abc import ABC, abstractmethod
from typing import Optional

logger = logging.getLogger(__name__)

_LLM_CACHE_PATH = os.path.join(os.path.dirname(__file__), "../.llm_cache.db")


class LLMCache:
    """
    LLM 响应缓存（SQLite）
    key = hash(model + system + user + temperature)
    自动过期：默认 7 天
    """

    def __init__(self, db_path: str = _LLM_CACHE_PATH, ttl_seconds: int = 7 * 86400):
        self.db_path = db_path
        self.ttl = ttl_seconds
        self._conn: Optional[sqlite3.Connection] = None
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path)
            self._conn.execute("PRAGMA journal_mode=WAL")
        return self._conn

    def _init_db(self):
        conn = self._get_conn()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS llm_cache (
                cache_key TEXT PRIMARY KEY,
                response TEXT NOT NULL,
                model TEXT,
                created_at REAL NOT NULL
            )
        """)
        conn.commit()

    @staticmethod
    def _make_key(model: str, system: str, user: str, temperature: float) -> str:
        raw = f"{model}||{system}||{user}||{temperature}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def get(self, model: str, system: str, user: str, temperature: float) -> Optional[str]:
        key = self._make_key(model, system, user, temperature)
        conn = self._get_conn()
        row = conn.execute(
            "SELECT response, created_at FROM llm_cache WHERE cache_key=?", (key,)
        ).fetchone()
        if row is None:
            return None
        if time.time() - row[1] > self.ttl:
            conn.execute("DELETE FROM llm_cache WHERE cache_key=?", (key,))
            conn.commit()
            return None
        logger.debug(f"[LLMCache] Hit: key={key[:8]}...")
        return row[0]

    def put(self, model: str, system: str, user: str, temperature: float, response: str):
        key = self._make_key(model, system, user, temperature)
        conn = self._get_conn()
        conn.execute(
            "INSERT OR REPLACE INTO llm_cache (cache_key, response, model, created_at) VALUES (?, ?, ?, ?)",
            (key, response, model, time.time())
        )
        conn.commit()

    def stats(self) -> dict:
        conn = self._get_conn()
        total = conn.execute("SELECT COUNT(*) FROM llm_cache").fetchone()[0]
        return {"total_cached": total, "db_path": self.db_path}

    def clear(self):
        conn = self._get_conn()
        conn.execute("DELETE FROM llm_cache")
        conn.commit()
        logger.info("[LLMCache] Cache cleared")


# 全局缓存实例
_llm_cache: Optional[LLMCache] = None


def get_llm_cache() -> LLMCache:
    global _llm_cache
    if _llm_cache is None:
        _llm_cache = LLMCache()
    return _llm_cache


def _load_env():
    """加载 .env 文件（不依赖 python-dotenv）"""
    env_path = os.path.join(os.path.dirname(__file__), "../.env")
    if not os.path.exists(env_path):
        return
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, _, val = line.partition("=")
                os.environ.setdefault(key.strip(), val.strip())


_load_env()


class BaseLLMClient(ABC):
    """LLM 客户端抽象基类（内置缓存）"""

    # 子类需要设置 model name，用于缓存 key
    _model_name: str = "unknown"
    _use_cache: bool = True

    @abstractmethod
    async def _raw_chat(
        self,
        system: str,
        user: str,
        temperature: float = 0.1,
        max_tokens: int = 4096,
    ) -> str:
        raise NotImplementedError

    async def chat(
        self,
        system: str,
        user: str,
        temperature: float = 0.1,
        max_tokens: int = 4096,
    ) -> str:
        """带缓存的 chat 接口"""
        if self._use_cache:
            cache = get_llm_cache()
            cached = cache.get(self._model_name, system, user, temperature)
            if cached is not None:
                return cached

        response = await self._raw_chat(system, user, temperature, max_tokens)

        if self._use_cache:
            cache = get_llm_cache()
            cache.put(self._model_name, system, user, temperature, response)

        return response


class OpenAIClient(BaseLLMClient):
    """OpenAI GPT-4 客户端"""

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o",
                 base_url: Optional[str] = None):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.model = model
        self._model_name = model
        self.base_url = base_url
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                from openai import AsyncOpenAI
                kwargs = {"api_key": self.api_key}
                if self.base_url:
                    kwargs["base_url"] = self.base_url
                self._client = AsyncOpenAI(**kwargs)
            except ImportError:
                raise ImportError("openai package required: pip install openai")
        return self._client

    async def _raw_chat(self, system: str, user: str,
                        temperature: float = 0.1, max_tokens: int = 4096) -> str:
        client = self._get_client()
        response = await client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content


class QwenClient(BaseLLMClient):
    """
    阿里云 Qwen 客户端（DashScope OpenAI 兼容模式）

    支持：
    - qwen3-235b-a22b（MoE 大模型，代码能力强）
    - qwen2.5-coder-32b-instruct（代码专用）
    - qwen-plus / qwen-turbo

    Qwen3 系列输出带 <think>...</think> 推理链，会自动过滤
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = None,
        base_url: Optional[str] = None,
        enable_thinking: bool = False,   # True = 保留推理链（更准确但更慢）
    ):
        self.api_key  = api_key  or os.environ.get("QWEN_API_KEY")
        self.model    = model    or os.environ.get("QWEN_MODEL", "qwen3-235b-a22b")
        self.base_url = base_url or os.environ.get(
            "QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        self.enable_thinking = enable_thinking
        self._model_name = self.model
        self._client = None

        if not self.api_key:
            raise ValueError(
                "Qwen API Key 未配置。请设置环境变量 QWEN_API_KEY 或在 .env 文件中添加。"
            )

    async def _raw_chat(self, system: str, user: str,
                        temperature: float = 0.1, max_tokens: int = 8192) -> str:
        """
        直接用 httpx 调用 DashScope OpenAI 兼容接口
        避免 openai 包的版本兼容性问题
        """
        import httpx, json as _json

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            "temperature": temperature,
            "max_tokens":  max_tokens,
        }
        # Qwen3 系列支持 enable_thinking 控制推理链
        if "qwen3" in self.model.lower():
            payload["enable_thinking"] = self.enable_thinking

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type":  "application/json",
        }

        url = self.base_url.rstrip("/") + "/chat/completions"

        try:
            async with httpx.AsyncClient(timeout=300.0) as client:
                resp = await client.post(url, json=payload, headers=headers)
                resp.raise_for_status()
                data = resp.json()

            content = data["choices"][0]["message"]["content"] or ""
            content = self._strip_thinking(content)
            return content.strip()

        except httpx.HTTPStatusError as e:
            body = e.response.text[:300]
            logger.error(f"[QwenClient] HTTP {e.response.status_code}: {body}")
            raise RuntimeError(f"Qwen API HTTP {e.response.status_code}: {body}") from e
        except Exception as e:
            logger.error(f"[QwenClient] API call failed: {e}")
            raise

    def _strip_thinking(self, text: str) -> str:
        """去除 Qwen3 的 <think>...</think> 推理块"""
        # 去除完整的 think 标签块
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        # 去除未闭合的 think 块（截断情况）
        text = re.sub(r"<think>.*$", "", text, flags=re.DOTALL)
        return text.strip()

    def with_thinking(self) -> "QwenClient":
        """返回启用思考模式的副本（用于复杂推理任务）"""
        return QwenClient(
            api_key=self.api_key,
            model=self.model,
            base_url=self.base_url,
            enable_thinking=True,
        )


class AnthropicClient(BaseLLMClient):
    """Anthropic Claude 客户端"""

    def __init__(self, api_key: Optional[str] = None, model: str = "claude-opus-4-5"):
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.model = model
        self._model_name = model
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                import anthropic
                self._client = anthropic.AsyncAnthropic(api_key=self.api_key)
            except ImportError:
                raise ImportError("anthropic package required: pip install anthropic")
        return self._client

    async def _raw_chat(self, system: str, user: str,
                        temperature: float = 0.1, max_tokens: int = 4096) -> str:
        client = self._get_client()
        response = await client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return response.content[0].text


class MockLLMClient(BaseLLMClient):
    """测试用 Mock 客户端，按算子关键词返回预设骨架代码"""

    # 各后端的算子模板
    _TEMPLATES = {
        "flash_attention": {
            "cuda": '''```cuda
#include <cuda_runtime.h>
#include <math.h>
__global__ void flash_attention_kernel(
    const half* __restrict__ Q, const half* __restrict__ K,
    const half* __restrict__ V, half* __restrict__ Out,
    int B, int H, int S, int D
) {
    extern __shared__ half smem[];
    int b = blockIdx.z, h = blockIdx.y;
    int tid = threadIdx.x;
    float scale = 1.0f / sqrtf((float)D);
    float m = -INFINITY, l = 0.0f;
    // tiled flash attention loop
    for (int j = 0; j < S; j += blockDim.x) {
        int jj = j + tid;
        if (jj >= S) continue;
        float qk = 0.0f;
        for (int d = 0; d < D; d++)
            qk += (float)Q[b*H*S*D+h*S*D+tid*D+d] * (float)K[b*H*S*D+h*S*D+jj*D+d];
        qk *= scale;
        float m_new = fmaxf(m, qk);
        l = l * expf(m - m_new) + expf(qk - m_new);
        m = m_new;
    }
    // write output
    if (tid < S)
        for (int d = 0; d < D; d++)
            Out[b*H*S*D+h*S*D+tid*D+d] = __float2half(1.0f / l);
}
```''',
            "ascendc": '''```cpp
#include "kernel_operator.h"
using namespace AscendC;
constexpr int TILE = 64;
class FlashAttentionKernel {
public:
    __aicore__ inline void Init(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR out, int S, int D) {
        qGm.SetGlobalBuffer((__gm__ half*)q, S*D);
        kGm.SetGlobalBuffer((__gm__ half*)k, S*D);
        outGm.SetGlobalBuffer((__gm__ half*)out, S*D);
        pipe.InitBuffer(inQ, 2, TILE*D*sizeof(half));
    }
    __aicore__ inline void Process() {
        int blockId = GetBlockIdx();
        LocalTensor<half> qLocal = inQ.AllocTensor<half>();
        DataCopy(qLocal, qGm[blockId*TILE*1], TILE);
        inQ.EnQue(qLocal);
        auto qi = inQ.DeQue<half>();
        inQ.FreeTensor(qi);
    }
private:
    GlobalTensor<half> qGm, kGm, outGm;
    TPipe pipe;
    TQue<QuePosition::VECIN, 2> inQ;
};
extern "C" __global__ __aicore__ void flash_attention_kernel(
    GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR out
) {
    FlashAttentionKernel op;
    op.Init(q, k, v, out, 512, 64);
    op.Process();
}
```''',
        },
        "rmsnorm": {
            "cuda": '''```cuda
#include <cuda_runtime.h>
__global__ void rmsnorm_kernel(
    const half* __restrict__ x, const half* __restrict__ weight,
    half* __restrict__ out, int N, int D, float eps
) {
    int row = blockIdx.x;
    extern __shared__ float smem[];
    float sum = 0.0f;
    for (int i = threadIdx.x; i < D; i += blockDim.x)
        sum += (float)x[row*D+i] * (float)x[row*D+i];
    smem[threadIdx.x] = sum;
    __syncthreads();
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (threadIdx.x < s) smem[threadIdx.x] += smem[threadIdx.x+s];
        __syncthreads();
    }
    float rms = rsqrtf(smem[0] / D + eps);
    for (int i = threadIdx.x; i < D; i += blockDim.x)
        out[row*D+i] = __float2half(__half2float(x[row*D+i]) * rms * __half2float(weight[i]));
}
```''',
            "ascendc": '''```cpp
#include "kernel_operator.h"
using namespace AscendC;
constexpr int TILE_LEN = 256;
class RMSNormKernel {
public:
    __aicore__ inline void Init(GM_ADDR xGmAddr, GM_ADDR wGmAddr, GM_ADDR outGmAddr,
                                 int rows, int cols) {
        xGm.SetGlobalBuffer((__gm__ half*)xGmAddr, rows*cols);
        wGm.SetGlobalBuffer((__gm__ half*)wGmAddr, cols);
        outGm.SetGlobalBuffer((__gm__ half*)outGmAddr, rows*cols);
        pipe.InitBuffer(inQ, 2, TILE_LEN * sizeof(half));
        pipe.InitBuffer(outQ, 2, TILE_LEN * sizeof(half));
        this->cols = cols;
    }
    __aicore__ inline void Process() {
        int blockId = GetBlockIdx();
        auto xLocal = inQ.AllocTensor<half>();
        DataCopy(xLocal, xGm[blockId * cols], cols);
        inQ.EnQue(xLocal);
        auto xi = inQ.DeQue<half>();
        auto yo = outQ.AllocTensor<half>();
        // RMS = sqrt(mean(x^2) + eps)
        Mul(yo, xi, xi, cols);          // x^2
        float rms_val = 0.0f;           // accumulate on scalar unit
        Muls(yo, yo, (half)(1.0f/cols), cols);
        outQ.EnQue(yo);
        auto y = outQ.DeQue<half>();
        DataCopy(outGm[blockId * cols], y, cols);
        outQ.FreeTensor(y);
        inQ.FreeTensor(xi);
    }
private:
    GlobalTensor<half> xGm, wGm, outGm;
    TPipe pipe;
    TQue<QuePosition::VECIN,  2> inQ;
    TQue<QuePosition::VECOUT, 2> outQ;
    int cols;
};
extern "C" __global__ __aicore__ void rmsnorm_kernel(
    GM_ADDR x, GM_ADDR w, GM_ADDR out
) {
    RMSNormKernel op;
    op.Init(x, w, out, 512, 4096);
    op.Process();
}
```''',
        },
        "gelu": {
            "cuda": '''```cuda
#include <cuda_runtime.h>
#include <math.h>
__global__ void gelu_kernel(const half* __restrict__ x, half* __restrict__ out, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float xi = __half2float(x[idx]);
        float v = 0.5f * xi * (1.0f + tanhf(0.7978845608f * (xi + 0.044715f * xi*xi*xi)));
        out[idx] = __float2half(v);
    }
}
```''',
        },
        "matmul": {
            "cuda": '''```cuda
#include <cuda_runtime.h>
#define TILE 128
__global__ void matmul_kernel(
    const half* __restrict__ A, const half* __restrict__ B,
    half* __restrict__ C, int M, int N, int K
) {
    __shared__ half As[TILE][TILE], Bs[TILE][TILE];
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    float acc = 0.0f;
    for (int k = 0; k < K; k += TILE) {
        As[ty][tx] = A[(by*TILE+ty)*K + k+tx];
        Bs[ty][tx] = B[(k+ty)*N + bx*TILE+tx];
        __syncthreads();
        for (int i = 0; i < TILE; i++) acc += __half2float(As[ty][i]) * __half2float(Bs[i][tx]);
        __syncthreads();
    }
    if (by*TILE+ty < M && bx*TILE+tx < N)
        C[(by*TILE+ty)*N + bx*TILE+tx] = __float2half(acc);
}
```''',
        },
    }

    def __init__(self, responses: dict[str, str] = None):
        self.responses = responses or {}
        self.call_count = 0
        self._model_name = "mock"
        self._use_cache = False  # Mock 不需要缓存

    async def _raw_chat(self, system: str, user: str,
                   temperature: float = 0.1, max_tokens: int = 4096) -> str:
        self.call_count += 1

        user_lower = user.lower()

        # 精确匹配用户自定义响应
        for keyword, response in self.responses.items():
            if keyword.lower() in user_lower:
                return response

        # 按算子 + 后端匹配内置模板
        # 注意：先判断 backward，避免 silu/gelu backward prompt 被误判为 forward 模板
        is_rmsnorm = any(k in user_lower for k in ("rmsnorm", "rms_norm", "rms norm"))
        is_backward = "backward" in user_lower or "grad_output" in user_lower

        # RMSNorm forward/backward（_TEMPLATES 里的 rmsnorm 没有 launcher，走专用逻辑）
        if is_rmsnorm and is_backward:
            return self._wrap_response(self._rmsnorm_backward_template(), "rmsnorm_backward", "cuda")
        elif is_rmsnorm:
            return self._wrap_response(self._rmsnorm_forward_template(), "rmsnorm_forward", "cuda")

        # SiLU/GeLU backward
        if is_backward:
            op = "silu" if "silu" in user_lower else ("gelu" if "gelu" in user_lower else "elementwise")
            return self._wrap_response(self._elementwise_backward_template(op), f"{op}_backward", "cuda")

        # 其余 forward：走 _TEMPLATES（silu、gelu、flash_attention 等）
        for op_name, backends in self._TEMPLATES.items():
            if op_name in user_lower:
                # rmsnorm 已在上面处理，这里不重复
                if op_name == "rmsnorm":
                    continue
                for backend, code in backends.items():
                    if backend in user_lower or backend == "cuda":
                        return self._wrap_response(code, op_name, backend)

        # 通用 forward fallback
        backend = "ascendc" if "ascendc" in user_lower else "cuda"
        op = "silu" if "silu" in user_lower else ("gelu" if "gelu" in user_lower else "custom_operator")
        return self._wrap_response(self._elementwise_forward_template(op), op, backend)

    def _elementwise_forward_template(self, op_name: str) -> str:
        if "silu" in op_name:
            impl = "float v = xi / (1.0f + expf(-xi));"
        elif "gelu" in op_name:
            impl = "float v = 0.5f * xi * (1.0f + tanhf(0.7978845608f * (xi + 0.044715f * xi*xi*xi)));"
        else:
            impl = "float v = xi;"
        return f'''```cuda
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void {op_name}_forward_kernel(const half* __restrict__ x, half* __restrict__ out, int N) {{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {{
        float xi = __half2float(x[idx]);
        {impl}
        out[idx] = __float2half(v);
    }}
}}

extern "C" void launch_kernel(void* x, void* out, int N) {{
    int block = 256;
    int grid = (N + block - 1) / block;
    {op_name}_forward_kernel<<<grid, block>>>(
        reinterpret_cast<const half*>(x),
        reinterpret_cast<half*>(out),
        N);
}}
```'''

    def _elementwise_backward_template(self, op_name: str) -> str:
        if "silu" in op_name:
            grad_impl = ("float sig = 1.0f / (1.0f + expf(-xi));\n"
                        "        float gx = go * sig * (1.0f + xi * (1.0f - sig));")
        elif "gelu" in op_name:
            grad_impl = ("float inner = 0.7978845608f * (xi + 0.044715f * xi*xi*xi);\n"
                        "        float cdf = 0.5f * (1.0f + tanhf(inner));\n"
                        "        float pdf = 0.7978845608f * (1.0f + 3.0f*0.044715f*xi*xi) / (coshf(inner)*coshf(inner));\n"
                        "        float gx = go * (cdf + xi * pdf);")
        else:
            grad_impl = "float gx = go;"
        return f'''```cuda
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math.h>

// grad_in 输出为 float32，避免 fp16 overflow 导致训练 NaN
__global__ void {op_name}_backward_kernel(
    const half* __restrict__ grad_out,
    const half* __restrict__ x,
    float* __restrict__ grad_in,   // float* 而非 half*
    int N
) {{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {{
        float go = __half2float(grad_out[idx]);
        float xi = __half2float(x[idx]);
        {grad_impl}
        grad_in[idx] = gx;   // 直接写 float，不做 float->half 截断
    }}
}}

// launcher：grad_in_fp32 指向 float32 buffer
extern "C" void launch_kernel(void* grad_out, void* x, void* grad_in_fp32, int N) {{
    int block = 256;
    int grid = (N + block - 1) / block;
    {op_name}_backward_kernel<<<grid, block>>>(
        reinterpret_cast<const half*>(grad_out),
        reinterpret_cast<const half*>(x),
        reinterpret_cast<float*>(grad_in_fp32),   // float* 输出
        N);
}}
```'''

    def _rmsnorm_forward_template(self) -> str:
        return '''```cuda
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math.h>

__global__ void rmsnorm_forward_kernel(
    const half* __restrict__ x,
    const half* __restrict__ weight,
    half* __restrict__ out,
    int H, float eps
) {
    extern __shared__ float smem[];
    int row = blockIdx.x;
    int tid = threadIdx.x;

    float sum_sq = 0.0f;
    for (int i = tid; i < H; i += blockDim.x) {
        float v = __half2float(x[row * H + i]);
        smem[i] = v;
        sum_sq += v * v;
    }
    __syncthreads();
    // Sequential reduction by thread 0
    if (tid == 0) {
        float total = 0.0f;
        for (int i = 0; i < H; i++) total += smem[i] * smem[i];
        smem[H] = rsqrtf(total / H + eps);
    }
    __syncthreads();
    float rms_inv = smem[H];
    for (int i = tid; i < H; i += blockDim.x) {
        out[row * H + i] = __float2half(smem[i] * rms_inv * __half2float(weight[i]));
    }
}

extern "C" void launch_kernel(void* x, void* weight, void* out, int N, int H, float eps) {
    int block = min(H, 256);
    int smem = (H + 1) * sizeof(float);
    rmsnorm_forward_kernel<<<N, block, smem>>>(
        reinterpret_cast<const half*>(x),
        reinterpret_cast<const half*>(weight),
        reinterpret_cast<half*>(out),
        H, eps);
}
```'''

    def _rmsnorm_backward_template(self) -> str:
        return '''```cuda
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math.h>

// grad_x 输出为 float32，避免 fp16 overflow 导致训练 NaN
__global__ void rmsnorm_backward_kernel(
    const half* __restrict__ grad_out,
    const half* __restrict__ x,
    const half* __restrict__ weight,
    float* __restrict__ grad_x,       // float* 而非 half*
    float* __restrict__ grad_weight,
    int H, float eps
) {
    extern __shared__ float smem[];
    float* sx = smem;
    float* sgo = smem + H;
    int row = blockIdx.x;
    int tid = threadIdx.x;

    for (int i = tid; i < H; i += blockDim.x) {
        sx[i] = __half2float(x[row * H + i]);
        sgo[i] = __half2float(grad_out[row * H + i]);
    }
    __syncthreads();

    if (tid == 0) {
        float ss = 0.0f;
        for (int i = 0; i < H; i++) ss += sx[i] * sx[i];
        float rms_inv = rsqrtf(ss / H + eps);
        smem[2*H] = rms_inv;
        // dot = mean(go * x * w)，注意 x 是原始输入（非 x_norm）
        float dot = 0.0f;
        for (int i = 0; i < H; i++)
            dot += sgo[i] * sx[i] * __half2float(weight[i]);  // 不含 rms_inv
        smem[2*H+1] = dot / H;
    }
    __syncthreads();

    float rms_inv = smem[2*H];
    float dot_mean = smem[2*H+1];
    for (int i = tid; i < H; i += blockDim.x) {
        float wi = __half2float(weight[i]);
        float xn = sx[i] * rms_inv;
        // 正确公式: (go*w - x_norm * mean(go*x*w)) * rms_inv
        float gx = (wi * sgo[i] - xn * dot_mean) * rms_inv;
        grad_x[row * H + i] = gx;   // 直接写 float，不做 float->half 截断
        atomicAdd(&grad_weight[i], sgo[i] * xn);
    }
}

// launcher：grad_x_fp32 指向 float32 buffer
extern "C" void launch_kernel(void* grad_out, void* x, void* weight,
                               void* grad_x_fp32, void* grad_weight,
                               int N, int H, float eps) {
    int block = min(H, 256);
    int smem = (2 * H + 2) * sizeof(float);
    rmsnorm_backward_kernel<<<N, block, smem>>>(
        reinterpret_cast<const half*>(grad_out),
        reinterpret_cast<const half*>(x),
        reinterpret_cast<const half*>(weight),
        reinterpret_cast<float*>(grad_x_fp32),   // float* 输出
        reinterpret_cast<float*>(grad_weight),
        H, eps);
}
```'''

    def _wrap_response(self, code: str, op_name: str, backend: str) -> str:
        return (
            f'Generated {backend} kernel for {op_name}:\n\n'
            f'{code}\n\n'
            f'Build flags: ["-O3", "-arch=native"]\n'
            f'Estimated bandwidth efficiency: 0.72\n'
            f'Optimizations: tiling, vectorized_load, shared_memory'
        )

    def _generic_template(self, backend: str) -> str:
        if backend == "ascendc":
            return '''```cpp
#include "kernel_operator.h"
using namespace AscendC;
class CustomKernel {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR out, int N) {
        xGm.SetGlobalBuffer((__gm__ half*)x, N);
        outGm.SetGlobalBuffer((__gm__ half*)out, N);
        pipe.InitBuffer(inQ, 2, 256 * sizeof(half));
        pipe.InitBuffer(outQ, 2, 256 * sizeof(half));
    }
    __aicore__ inline void Process() {
        int blockId = GetBlockIdx();
        auto xLocal = inQ.AllocTensor<half>();
        DataCopy(xLocal, xGm[blockId * 256], 256);
        inQ.EnQue(xLocal);
        auto xi = inQ.DeQue<half>();
        auto yo = outQ.AllocTensor<half>();
        Muls(yo, xi, (half)2.0f, 256);
        outQ.EnQue(yo);
        auto y = outQ.DeQue<half>();
        DataCopy(outGm[blockId * 256], y, 256);
        outQ.FreeTensor(y);
        inQ.FreeTensor(xi);
    }
private:
    GlobalTensor<half> xGm, outGm;
    TPipe pipe;
    TQue<QuePosition::VECIN, 2> inQ;
    TQue<QuePosition::VECOUT, 2> outQ;
};
extern "C" __global__ __aicore__ void custom_kernel(GM_ADDR x, GM_ADDR out) {
    CustomKernel op;
    op.Init(x, out, 1024);
    op.Process();
}
```'''
        return '''```cuda
#include <cuda_runtime.h>
__global__ void custom_kernel(const half* __restrict__ x, half* __restrict__ out, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float v = __half2float(x[idx]);
        out[idx] = __float2half(v * 2.0f);
    }
}
```'''


def create_llm_client(backend: str = "qwen", **kwargs) -> BaseLLMClient:
    """
    工厂函数：创建 LLM 客户端

    backend 可选：
      qwen      → 阿里云 Qwen（DashScope）
      openai    → OpenAI GPT-4o
      anthropic → Anthropic Claude
      mock      → 本地 Mock（测试用）
    """
    if backend == "qwen":
        allowed = {"api_key", "model", "base_url", "enable_thinking"}
        qwen_kwargs = {k: v for k, v in kwargs.items() if k in allowed}
        return QwenClient(**qwen_kwargs)
    elif backend == "openai":
        allowed = {"api_key", "model", "base_url"}
        return OpenAIClient(**{k: v for k, v in kwargs.items() if k in allowed})
    elif backend == "anthropic":
        allowed = {"api_key", "model"}
        return AnthropicClient(**{k: v for k, v in kwargs.items() if k in allowed})
    elif backend == "mock":
        allowed = {"responses"}
        return MockLLMClient(**{k: v for k, v in kwargs.items() if k in allowed})
    else:
        raise ValueError(
            f"Unknown LLM backend: '{backend}'. "
            f"Choose from: qwen, openai, anthropic, mock"
        )
