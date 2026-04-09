"""
编译错误知识库
============
自动积累 LLM 生成代码的编译错误模式，提供：
1. auto_fix(source, backend) — 自动修复已知错误（替代手工 _patch_cuda_source）
2. generate_prompt_fragment(backend) — 生成注入 CodeGen prompt 的禁用列表
3. record_error(backend, stderr, source) — 编译失败时自动学习新模式
"""
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field, asdict
from typing import Optional

logger = logging.getLogger(__name__)

STORE_PATH = os.path.join(os.path.dirname(__file__), "..", ".compile_errors.json")


@dataclass
class ErrorPattern:
    """一条编译错误修复规则"""
    pattern_id: str
    backend: str  # "cuda" | "hip" | "ascendc"
    description: str  # 人类可读的描述
    # 自动修复
    fix_type: str = "replace"  # "replace" | "regex" | "inject_header" | "remove_block" | "conditional"
    match_pattern: str = ""  # 匹配源码的正则/字符串
    replacement: str = ""  # 替换内容
    condition: str = ""  # 条件表达式（用于 conditional 类型）
    # prompt 注入
    bad_example: str = ""
    good_example: str = ""
    prompt_hint: str = ""  # 简短的禁用说明
    # 统计
    occurrence_count: int = 0
    last_seen: float = 0.0
    created_at: float = field(default_factory=time.time)


class CompileErrorKB:
    """编译错误知识库 — 从失败中学习，注入 prompt 防止重复错误"""

    def __init__(self, store_path: str = STORE_PATH):
        self.store_path = store_path
        self._patterns: dict[str, ErrorPattern] = {}
        self._load()
        if not self._patterns:
            self._seed_cuda_rules()

    # ── 公开 API ──────────────────────────────────────────

    def auto_fix(self, source: str, backend: str) -> str:
        """对源码应用所有已知的自动修复规则"""
        for p in sorted(self._patterns.values(), key=lambda x: x.pattern_id):
            if p.backend != backend:
                continue
            try:
                source = self._apply_fix(source, p)
            except Exception as e:
                logger.debug(f"[CompileErrorKB] Fix {p.pattern_id} failed: {e}")

        # 附加修复：处理嵌套括号的 half2 = make_float2(...) 模式
        if backend == "cuda":
            source = self._fix_half2_make_float2(source)
        return source

    def _fix_half2_make_float2(self, src: str) -> str:
        """修复 half2 var = make_float2(nested_func(...), ...) 的类型不匹配"""
        # 找所有 half2 varname = make_float2( 开始的位置，手动匹配平衡括号
        result = []
        i = 0
        pattern = re.compile(r'(half2\s+\w+\s*=\s*)make_float2\(')
        while i < len(src):
            m = pattern.search(src, i)
            if not m:
                result.append(src[i:])
                break
            result.append(src[i:m.start()])
            # 找到 make_float2( 之后，匹配平衡括号
            paren_start = m.end() - 1  # 指向 (
            depth = 1
            j = paren_start + 1
            while j < len(src) and depth > 0:
                if src[j] == '(':
                    depth += 1
                elif src[j] == ')':
                    depth -= 1
                j += 1
            # src[paren_start+1:j-1] 是 make_float2 的参数
            args = src[paren_start + 1:j - 1]
            result.append(m.group(1) + f"__floats2half2_rn({args})")
            i = j
        return "".join(result)

    def generate_prompt_fragment(self, backend: str, max_chars: int = 2500) -> str:
        """生成注入 CodeGen prompt 的已知陷阱列表"""
        relevant = [p for p in self._patterns.values()
                    if p.backend == backend and p.prompt_hint]
        # 按出现频率降序
        relevant.sort(key=lambda x: x.occurrence_count, reverse=True)

        lines = ["严格禁止使用以下不存在的API或写法（会导致编译失败）："]
        total = 0
        for p in relevant:
            hint = f"- {p.prompt_hint}"
            if p.bad_example and p.good_example:
                hint += f"（错误：{p.bad_example}，正确：{p.good_example}）"
            if total + len(hint) > max_chars:
                break
            lines.append(hint)
            total += len(hint)

        return "\n".join(lines)

    def record_error(self, backend: str, stderr: str, source_code: str) -> Optional[ErrorPattern]:
        """编译失败时调用，尝试归类到已有模式或学习新模式"""
        if not stderr:
            return None

        # 检查是否匹配已有模式
        for p in self._patterns.values():
            if p.backend == backend and self._matches_stderr(p, stderr):
                p.occurrence_count += 1
                p.last_seen = time.time()
                self._save()
                logger.info(f"[CompileErrorKB] Known pattern matched: {p.pattern_id} (count={p.occurrence_count})")
                return p

        # 尝试自动提取新模式（从 stderr 中提取 identifier/type 错误）
        new_pattern = self._try_extract_pattern(backend, stderr, source_code)
        if new_pattern:
            self._patterns[new_pattern.pattern_id] = new_pattern
            self._save()
            logger.info(f"[CompileErrorKB] Learned new pattern: {new_pattern.pattern_id}")
            return new_pattern

        return None

    def get_pattern(self, pattern_id: str) -> Optional[ErrorPattern]:
        return self._patterns.get(pattern_id)

    def count(self, backend: str = None) -> int:
        if backend:
            return sum(1 for p in self._patterns.values() if p.backend == backend)
        return len(self._patterns)

    # ── 修复引擎 ──────────────────────────────────────────

    def _apply_fix(self, src: str, p: ErrorPattern) -> str:
        if p.fix_type == "replace":
            if p.match_pattern and p.match_pattern in src:
                src = src.replace(p.match_pattern, p.replacement)
        elif p.fix_type == "regex":
            if p.match_pattern:
                src = re.sub(p.match_pattern, p.replacement, src)
        elif p.fix_type == "inject_header":
            if p.condition:
                # condition 格式: "trigger_keyword && !already_present"
                parts = p.condition.split(" && ")
                trigger = parts[0].strip()
                guard = parts[1].strip().lstrip("!") if len(parts) > 1 else ""
                if trigger in src and (not guard or guard not in src):
                    src = p.replacement + "\n" + src
        elif p.fix_type == "remove_block":
            if p.match_pattern:
                src = re.sub(p.match_pattern, p.replacement or "", src)
        elif p.fix_type == "conditional":
            # 自定义条件逻辑，由 _apply_conditional_fix 处理
            src = self._apply_conditional_fix(src, p)
        return src

    def _apply_conditional_fix(self, src: str, p: ErrorPattern) -> str:
        """处理复杂的条件修复逻辑"""
        cid = p.pattern_id

        if cid == "cuda_wmma_namespace":
            has_wmma = any(kw in src for kw in [
                "wmma::", "nvcuda::wmma", "load_matrix_sync",
                "store_matrix_sync", "fill_fragment", "mma_sync"
            ])
            if has_wmma:
                if "#include <mma.h>" not in src:
                    src = "#include <mma.h>\n" + src
                src = re.sub(r'using\s+namespace\s+nvcuda\s*;\s*\n?', '', src)
                if "wmma::" in src and "nvcuda::wmma::" not in src and "using namespace nvcuda::wmma" not in src:
                    src = re.sub(
                        r'(#include\s*<mma\.h>\s*\n)',
                        r'\1using namespace nvcuda::wmma;\n',
                        src, count=1
                    )
                if "fragment<" in src and "wmma::" not in src and "nvcuda::wmma" not in src:
                    src = re.sub(
                        r'(#include\s*<mma\.h>\s*\n)',
                        r'\1using namespace nvcuda::wmma;\n',
                        src, count=1
                    )

        elif cid == "cuda_half4_typedef":
            if "half4" in src:
                src = "typedef float4 half4;  // patched: half4 does not exist\n" + src

        elif cid == "cuda_wmma_fragment_size":
            src = re.sub(
                r'fragment<(wmma::|nvcuda::wmma::)(matrix_a|matrix_b|accumulator),\s*8,\s*8,\s*8',
                r'fragment<\1\2, 16, 16, 16',
                src
            )

        elif cid == "cuda_undefined_constants":
            consts = []
            if "sqrt_2_over_pi" in src and "constexpr" not in src.split("sqrt_2_over_pi")[0][-50:]:
                consts.append("constexpr float sqrt_2_over_pi = 0.7978845608028654f;")
            if "M_SQRT2" in src and "#define M_SQRT2" not in src:
                consts.append("#define M_SQRT2 1.41421356237f")
            if "M_PI" in src and "#define M_PI" not in src and "<math.h>" not in src and "<cmath>" not in src:
                consts.append("#define M_PI 3.14159265358979323846f")
            if consts:
                last_inc = max(src.rfind("#include"), 0)
                pos = src.find('\n', last_inc) + 1 if last_inc >= 0 else 0
                src = src[:pos] + "\n".join(consts) + "\n" + src[pos:]

        elif cid == "cuda_remove_duplicate_h2_funcs":
            for fn in ('h2exp', 'h2sin', 'h2cos', 'h2sqrt', 'h2log', 'h2neg', 'h2abs'):
                src = re.sub(
                    rf'__device__\s+(?:__forceinline__\s+)?half2\s+{fn}\s*\([^)]*\)\s*\{{[^}}]*\}}\s*\n?',
                    f'// patched: removed duplicate {fn}\n',
                    src
                )

        elif cid == "cuda_vla_fix":
            src = re.sub(
                r'((?:float|half|__half|int)\s+\w+)\[(\s*head_dim\s*)\]',
                r'\1[64 /* patched: head_dim must be constexpr */]',
                src
            )
            src = re.sub(
                r'((?:float|half|__half|int)\s+\w+)\[(\s*seq_len\s*)\]',
                r'\1[512 /* patched: seq_len must be constexpr */]',
                src
            )

        elif cid == "cuda_remove_torch_wrapper":
            src = re.sub(
                r'(?:^|\n)(?:torch::Tensor|at::Tensor|void)\s+\w+_cuda\s*\(.*?\n\}',
                '', src, flags=re.DOTALL
            )
            src = re.sub(r'#include\s*<(?:torch|ATen|c10)/[^>]*>\s*\n?', '', src)

        return src

    # ── 错误模式匹配 ─────────────────────────────────────

    def _matches_stderr(self, p: ErrorPattern, stderr: str) -> bool:
        """检查 stderr 是否匹配某个已知模式"""
        keywords = {
            "cuda_float22half2": "__float22half2_rn",
            "cuda_h2neg": "__h2neg",
            "cuda_h2recip": "h2recip",
            "cuda_h2div": "__h2div",
            "cuda_half4_typedef": "half4",
            "cuda_flt_max_header": "FLT_MAX",
            "cuda_sqrtf_host": "__sqrtf",
            "cuda_reduce_sum": "__reduce_sum",
            "cuda_half2float4": "__half2float4",
            "cuda_keyword_varname": 'half2 in',
            "cuda_reinterpret_cast_assign": "reinterpret_cast",
            "cuda_wmma_namespace": "using namespace nvcuda;",
        }
        kw = keywords.get(p.pattern_id, "")
        return bool(kw and kw in stderr)

    def _try_extract_pattern(self, backend: str, stderr: str, source: str) -> Optional[ErrorPattern]:
        """尝试从 stderr 自动提取新的错误模式"""
        # 提取 "identifier XXX is undefined"
        m = re.search(r'identifier\s+"(\w+)"\s+is undefined', stderr)
        if m:
            ident = m.group(1)
            pid = f"{backend}_undefined_{ident}"
            if pid not in self._patterns:
                return ErrorPattern(
                    pattern_id=pid,
                    backend=backend,
                    description=f"Identifier '{ident}' is undefined",
                    prompt_hint=f"不要使用 {ident}（不存在）",
                    occurrence_count=1,
                    last_seen=time.time(),
                )

        # 提取 "name must be a namespace name"
        if "name must be a namespace name" in stderr:
            m2 = re.search(r'using\s+namespace\s+(\w+)', stderr)
            if m2:
                ns = m2.group(1)
                pid = f"{backend}_bad_namespace_{ns}"
                if pid not in self._patterns:
                    return ErrorPattern(
                        pattern_id=pid,
                        backend=backend,
                        description=f"'{ns}' is not a valid namespace",
                        prompt_hint=f"不要使用 using namespace {ns};（不是合法命名空间）",
                        occurrence_count=1,
                        last_seen=time.time(),
                    )

        return None

    # ── 种子数据：从现有 16 条 patch 规则初始化 ────────────

    def _seed_cuda_rules(self):
        """将 _patch_cuda_source 的 16 条规则转为结构化 ErrorPattern"""
        rules = [
            # 1. 注入头文件
            ErrorPattern(
                pattern_id="cuda_fp16_header",
                backend="cuda",
                description="使用 half/half2 时需要 #include <cuda_fp16.h>",
                fix_type="inject_header",
                condition="half && !#include <cuda_fp16.h>",
                replacement="#include <cuda_fp16.h>",
                prompt_hint="使用 half/half2 时必须 #include <cuda_fp16.h>",
            ),
            ErrorPattern(
                pattern_id="cuda_stdio_header",
                backend="cuda",
                description="fprintf/stderr 需要 #include <stdio.h>",
                fix_type="inject_header",
                condition="fprintf && !#include <stdio.h>",
                replacement="#include <stdio.h>",
                prompt_hint="不要在 kernel 代码里使用 fprintf/printf/stderr",
            ),
            ErrorPattern(
                pattern_id="cuda_flt_max_header",
                backend="cuda",
                description="FLT_MAX/FLT_MIN 需要 #include <cfloat>",
                fix_type="inject_header",
                condition="FLT_MAX && !#include <cfloat>",
                replacement="#include <cfloat>",
                prompt_hint="使用 FLT_MAX 时必须 #include <cfloat>",
            ),

            # 1c. __sqrtf → sqrtf
            ErrorPattern(
                pattern_id="cuda_sqrtf_host",
                backend="cuda",
                description="__sqrtf 是 host 函数，device 代码用 sqrtf",
                fix_type="replace",
                match_pattern="__sqrtf(",
                replacement="sqrtf(",
                prompt_hint="不要使用 __sqrtf/__logf（是 host 函数），用 sqrtf/logf",
                bad_example="__sqrtf(x)",
                good_example="sqrtf(x)",
            ),
            ErrorPattern(
                pattern_id="cuda_logf_host",
                backend="cuda",
                description="__logf 是 host 版本",
                fix_type="replace",
                match_pattern="__logf(",
                replacement="logf(",
            ),
            ErrorPattern(
                pattern_id="cuda_expf_host",
                backend="cuda",
                description="__expf_host 不存在",
                fix_type="replace",
                match_pattern="__expf_host(",
                replacement="expf(",
            ),

            # 1d. 删除 torch:: wrapper
            ErrorPattern(
                pattern_id="cuda_remove_torch_wrapper",
                backend="cuda",
                description="纯 CUDA 编译不需要 torch:: 包装函数",
                fix_type="conditional",
                prompt_hint="不要生成 torch::Tensor / at::Tensor 的 host wrapper 函数",
            ),

            # 1e. VLA 修复
            ErrorPattern(
                pattern_id="cuda_vla_fix",
                backend="cuda",
                description="CUDA device 代码不允许 VLA (Variable Length Array)",
                fix_type="conditional",
                prompt_hint="不要在 device 代码中使用变长数组如 float arr[head_dim]，必须用编译期常量",
                bad_example="float arr[head_dim]",
                good_example="float arr[64]",
            ),

            # 2. reinterpret_cast 缺少 *
            ErrorPattern(
                pattern_id="cuda_reinterpret_cast_assign",
                backend="cuda",
                description="reinterpret_cast 直接赋值缺少解引用 *",
                fix_type="regex",
                match_pattern=r'(?<!\*)(reinterpret_cast<[^>]+\*[^>]*>\s*\([^)]+\))\s*=\s*(?!=)',
                replacement=r'*\1 = ',
                prompt_hint="不要对 reinterpret_cast 结果直接赋值，必须加 *：*reinterpret_cast<half2*>(p) = v;",
                bad_example="reinterpret_cast<half2*>(p) = v",
                good_example="*reinterpret_cast<half2*>(p) = v",
            ),

            # 3. __float22half2_rn → __floats2half2_rn
            ErrorPattern(
                pattern_id="cuda_float22half2",
                backend="cuda",
                description="__float22half2_rn 签名错误",
                fix_type="replace",
                match_pattern="__float22half2_rn",
                replacement="__floats2half2_rn",
                prompt_hint="__float22half2_rn(float,float) 不存在！正确是 __floats2half2_rn(float a, float b)",
                bad_example="__float22half2_rn(a, b)",
                good_example="__floats2half2_rn(a, b)",
            ),

            # 4. half4 不存在
            ErrorPattern(
                pattern_id="cuda_half4_typedef",
                backend="cuda",
                description="half4 类型不存在",
                fix_type="conditional",
                prompt_hint="half4 / half8 不存在，用两个 half2 或 float4 代替",
            ),

            # 5. 不存在的 float2 intrinsics
            ErrorPattern(
                pattern_id="cuda_float2float2_rn",
                backend="cuda",
                description="__float2float2_rn 不存在",
                fix_type="regex",
                match_pattern=r'__float2float2_rn\(([^)]+)\)',
                replacement=r'make_float2(\1, \1)',
                prompt_hint="__float2float2_rn / __fmul2_rn / __fadd2_rn / __tanh2 均不存在",
            ),
            ErrorPattern(
                pattern_id="cuda_fmul2_rn",
                backend="cuda",
                description="__fmul2_rn 不存在",
                fix_type="regex",
                match_pattern=r'__fmul2_rn\(([^,]+),\s*([^)]+)\)',
                replacement=r'make_float2(\1.x * \2.x, \1.y * \2.y)',
            ),
            ErrorPattern(
                pattern_id="cuda_fadd2_rn",
                backend="cuda",
                description="__fadd2_rn 不存在",
                fix_type="regex",
                match_pattern=r'__fadd2_rn\(([^,]+),\s*([^)]+)\)',
                replacement=r'make_float2(\1.x + \2.x, \1.y + \2.y)',
            ),
            ErrorPattern(
                pattern_id="cuda_tanh2",
                backend="cuda",
                description="__tanh2 不存在",
                fix_type="regex",
                match_pattern=r'__tanh2\(([^)]+)\)',
                replacement=r'make_float2(tanhf(\1.x), tanhf(\1.y))',
            ),

            # 6. 多余括号
            ErrorPattern(
                pattern_id="cuda_extra_bracket",
                backend="cuda",
                description="LLM 有时生成多余的 ] 括号",
                fix_type="regex",
                match_pattern=r'\]\s*\]\s*\)',
                replacement=r'])',
            ),

            # 7. wmma 命名空间
            ErrorPattern(
                pattern_id="cuda_wmma_namespace",
                backend="cuda",
                description="wmma 命名空间处理",
                fix_type="conditional",
                prompt_hint='不要使用 "using namespace nvcuda;"（nvcuda 不是 C++ namespace），正确写法是 "using namespace nvcuda::wmma;"',
                bad_example="using namespace nvcuda;",
                good_example="using namespace nvcuda::wmma;",
            ),

            # 8. wmma fragment 尺寸
            ErrorPattern(
                pattern_id="cuda_wmma_fragment_size",
                backend="cuda",
                description="RTX 4090 (sm_89) 支持 16x16x16 不支持 8x8x8",
                fix_type="conditional",
                prompt_hint="wmma fragment 尺寸用 16x16x16（sm_89 不支持 8x8x8）",
            ),

            # 9. wmma store_matrix_sync half* → float*
            ErrorPattern(
                pattern_id="cuda_wmma_store_half",
                backend="cuda",
                description="store_matrix_sync 不接受 half* 目标指针",
                fix_type="regex",
                match_pattern=r'store_matrix_sync\s*\(\s*\((?:__)?half\s*\*\)',
                replacement=r'store_matrix_sync((float*)',
                prompt_hint="wmma store_matrix_sync 目标指针必须是 float*，不能是 half*",
            ),

            # 10. float2 ↔ half2 类型不匹配
            ErrorPattern(
                pattern_id="cuda_float2_to_half2_assign",
                backend="cuda",
                description="float2 var = __floats2half2_rn(...) 类型不匹配",
                fix_type="regex",
                match_pattern=r'float2\s+(\w+)\s*=\s*(__floats2half2_rn\s*\([^)]+\))',
                replacement=r'half2 \1 = \2',
                prompt_hint="__floats2half2_rn 返回 half2，不能赋给 float2 变量",
            ),
            ErrorPattern(
                pattern_id="cuda_float2_to_halves2half2",
                backend="cuda",
                description="float2 var = __halves2half2(...) 类型不匹配",
                fix_type="regex",
                match_pattern=r'float2\s+(\w+)\s*=\s*(__halves2half2\s*\([^)]+\))',
                replacement=r'half2 \1 = \2',
            ),

            # 11. 未声明常量
            ErrorPattern(
                pattern_id="cuda_undefined_constants",
                backend="cuda",
                description="LLM 使用未声明的常量名",
                fix_type="conditional",
                prompt_hint="不要使用未定义的常量名如 sqrt_2_over_pi / M_SQRT2 等，直接写数值",
            ),

            # 12. 重复定义 h2exp 等
            ErrorPattern(
                pattern_id="cuda_remove_duplicate_h2_funcs",
                backend="cuda",
                description="重新定义 h2exp/h2sin 等 CUDA 内置函数",
                fix_type="conditional",
                prompt_hint="不要重新定义 h2exp / h2sin / h2cos / h2sqrt 等 CUDA 内置 half2 函数",
            ),

            # 13. __h2neg → __hneg2
            ErrorPattern(
                pattern_id="cuda_h2neg",
                backend="cuda",
                description="__h2neg/__h2neg2 不存在",
                fix_type="replace",
                match_pattern="__h2neg2(",
                replacement="__hneg2(",
                prompt_hint="__h2neg / __h2neg2 不存在，用 __hneg2 代替",
            ),
            ErrorPattern(
                pattern_id="cuda_h2neg_v2",
                backend="cuda",
                description="__h2neg 不存在",
                fix_type="replace",
                match_pattern="__h2neg(",
                replacement="__hneg2(",
            ),
            ErrorPattern(
                pattern_id="cuda_h2recip",
                backend="cuda",
                description="h2recip 不存在",
                fix_type="regex",
                match_pattern=r'\bh2recip\s*\(\s*([^)]+)\s*\)',
                replacement=r'__halves2half2(hrcp(__low2half(\1)), hrcp(__high2half(\1)))',
                prompt_hint="h2recip 不存在",
            ),
            ErrorPattern(
                pattern_id="cuda_h2div",
                backend="cuda",
                description="__h2div/__h2div2 不存在",
                fix_type="regex",
                match_pattern=r'__h2div2?\s*\(\s*([^,]+),\s*([^)]+)\)',
                replacement=r'__halves2half2(__hdiv(__low2half(\1), __low2half(\2)), __hdiv(__high2half(\1), __high2half(\2)))',
                prompt_hint="__h2div / __h2div2 不存在",
            ),
            ErrorPattern(
                pattern_id="cuda_h2exp2",
                backend="cuda",
                description="__h2exp2 不存在",
                fix_type="replace",
                match_pattern="__h2exp2(",
                replacement="h2exp(",
            ),
            ErrorPattern(
                pattern_id="cuda_h2exp_prefix",
                backend="cuda",
                description="__h2exp 不存在",
                fix_type="replace",
                match_pattern="__h2exp(",
                replacement="h2exp(",
            ),

            # 14. __half2float4 不存在
            ErrorPattern(
                pattern_id="cuda_half2float4",
                backend="cuda",
                description="__half2float4 不存在",
                fix_type="regex",
                match_pattern=r'__half2float4\s*\(\s*([^)]+)\s*\)',
                replacement=r'make_float4(__half2float(\1[0]), __half2float(\1[1]), __half2float(\1[2]), __half2float(\1[3]))',
                prompt_hint="__half2float4 / __float2half4 不存在",
            ),
            ErrorPattern(
                pattern_id="cuda_float2half4",
                backend="cuda",
                description="__float2half4 不存在",
                fix_type="regex",
                match_pattern=r'__float2half4\s*\(\s*([^)]+)\s*\)',
                replacement=r'// patched: __float2half4 does not exist',
            ),

            # 15. __reduce_sum 不存在
            ErrorPattern(
                pattern_id="cuda_reduce_sum",
                backend="cuda",
                description="__reduce_sum 不存在",
                fix_type="regex",
                match_pattern=r'__reduce_sum\s*\(\s*([^)]+)\s*\)',
                replacement=r'[&](){auto _v=\1; for(int _i=16;_i>0;_i>>=1) _v+=__shfl_down_sync(0xffffffff,_v,_i); return _v;}()',
                prompt_hint="__reduce_sum / __reduce_add 不存在，用 __shfl_down_sync 手动实现",
            ),
            ErrorPattern(
                pattern_id="cuda_reduce_add",
                backend="cuda",
                description="__reduce_add 不存在",
                fix_type="regex",
                match_pattern=r'__reduce_add\s*\(',
                replacement=r'__reduce_sum(',
            ),

            # 16. C++ 关键字变量名
            ErrorPattern(
                pattern_id="cuda_keyword_varname",
                backend="cuda",
                description="'in' 是 C++ 关键字不能做变量名",
                fix_type="regex",
                match_pattern=r'\bhalf2\s+in\b',
                replacement='half2 val_in',
                prompt_hint="不要用 C++ 关键字 (in, register) 做变量名",
            ),
            ErrorPattern(
                pattern_id="cuda_keyword_varname_half",
                backend="cuda",
                description="'in' 是 C++ 关键字不能做变量名 (half)",
                fix_type="regex",
                match_pattern=r'\bhalf\s+in\b',
                replacement='half val_in',
            ),
            ErrorPattern(
                pattern_id="cuda_keyword_varname_float",
                backend="cuda",
                description="'in' 是 C++ 关键字不能做变量名 (float)",
                fix_type="regex",
                match_pattern=r'\bfloat\s+in\b',
                replacement='float val_in',
            ),
        ]

        for r in rules:
            self._patterns[r.pattern_id] = r

        self._save()
        logger.info(f"[CompileErrorKB] Seeded {len(rules)} CUDA rules")

    # ── 持久化 ────────────────────────────────────────────

    def _load(self):
        if not os.path.exists(self.store_path):
            return
        try:
            with open(self.store_path) as f:
                data = json.load(f)
            for pid, d in data.items():
                self._patterns[pid] = ErrorPattern(**d)
            logger.info(f"[CompileErrorKB] Loaded {len(self._patterns)} patterns")
        except Exception as e:
            logger.warning(f"[CompileErrorKB] Failed to load: {e}")

    def _save(self):
        try:
            with open(self.store_path, "w") as f:
                json.dump({pid: asdict(p) for pid, p in self._patterns.items()}, f, indent=2)
        except Exception as e:
            logger.warning(f"[CompileErrorKB] Failed to save: {e}")


# ── 全局单例 ──────────────────────────────────────────────

_instance: Optional[CompileErrorKB] = None


def get_compile_error_kb(store_path: str = STORE_PATH) -> CompileErrorKB:
    global _instance
    if _instance is None:
        _instance = CompileErrorKB(store_path)
    return _instance
