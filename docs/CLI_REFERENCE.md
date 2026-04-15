# Operator Agent 2 — CLI 系统完整探索

**项目路径：** `/remote-home1/hywei/operator_agent_2/`

**生成日期：** 2026-04-15

---

## 一、入口脚本和功能

### 1.1 主入口文件

#### **main.py**（异构GPU算子生成系统主入口）
- **位置：** `/remote-home1/hywei/operator_agent_2/main.py`
- **功能：** 核心系统初始化和编排
- **关键函数：**
  - `setup_logging()` - 配置日志记录
  - `load_config()` - 加载 config/config.yaml
  - `build_agent_system()` - 构建Agent系统，注册所有子Agent
- **依赖Agent：**
  - `HardwareProfilerAgent` - 硬件分析
  - `OperatorSpecAgent` - 算子规格解析
  - `CodeGenAgent` - 代码生成
  - `OptimizerAgent` - 优化算子
  - `VerifierAgent` - 验证算子
  - `DistributionAgent` - 分布式优化

#### **cli.py**（主CLI命令行接口）
- **位置：** `/remote-home1/hywei/operator_agent_2/cli.py`
- **工具框架：** Click（Python CLI）
- **主要功能：** 自然语言驱动的异构算子生成系统

---

## 二、CLI 命令系统

### 2.1 顶级命令组

```bash
python cli.py [COMMAND] [OPTIONS]
```

#### **generate** - 生成算子内核
**用法模式：**
1. **自然语言模式**（LLM意图解析，缺信息会追问）：
   ```bash
   python cli.py generate "帮我生成一个 SiLU 激活函数的算子，目标是昇腾 910B"
   python cli.py generate "写个 RoPE 算子"
   python cli.py generate "Generate FlashAttention v2 for H100 and MI300X"
   ```

2. **显式参数模式**（跳过意图解析）：
   ```bash
   python cli.py generate --op silu --gpu ascend_910b --backend ascendc
   python cli.py generate --op rmsnorm --gpu rtx_4090 --backend cuda
   python cli.py generate --op gelu --gpu h100_sxm5 mi300x --backend hip
   ```

**参数列表：**
| 参数 | 类型 | 默认值 | 说明 |
|------|------|-------|------|
| `request` | 文本 | - | 自然语言需求描述 |
| `--op` | 字符串 | - | 算子名称（显式指定时必需） |
| `--gpu` | 多值 | - | 目标GPU（可指定多个）：`ascend_910b`, `ascend_910c`, `rtx_4090`, `h100_sxm5`, `a100_80gb`, `rtx_3090`, `mi300x` |
| `--backend` | 选择 | - | 编程后端：`cuda` \| `hip` \| `ascendc` \| `triton` |
| `--llm` | 选择 | `qwen` | LLM后端：`qwen` \| `openai` \| `anthropic` \| `mock` |
| `--save/--no-save` | 布尔 | True | 是否保存到算子仓库 |
| `--output` | 路径 | `./output` | 输出代码文件目录 |
| `--review/--no-review` | 布尔 | True | 是否运行 ReviewLoop 验证 |

**环境变量映射：**
- LLM后端与API密钥：
  - `openai` → `OPENAI_API_KEY`
  - `anthropic` → `ANTHROPIC_API_KEY`
  - `qwen` → `QWEN_API_KEY`

---

#### **npu-test** - NPU测试（昇腾910B）
```bash
python cli.py npu-test --llm qwen
python cli.py npu-test --llm mock --ops silu gelu
```

**参数：**
- `--llm` (默认: `mock`) - LLM后端
- `--ops` (多值) - 指定测试的算子列表

---

#### **registry** - 算子仓库管理（子命令组）

##### `registry list` - 列出所有算子
```bash
python cli.py registry list
python cli.py registry list --gpu h100_sxm5
python cli.py registry list --backend cuda
```
**参数：**
- `--gpu` - 按GPU型号过滤
- `--backend` - 按后端过滤

##### `registry show` - 查看算子详情
```bash
python cli.py registry show silu ascend_910b
python cli.py registry show rmsnorm rtx_4090
```
**参数：**
- 位置参数1: `operator_name` - 算子名称
- 位置参数2: `gpu_model` - GPU型号

**输出信息：** 算子版本、验证状态、相对误差、带宽利用率、源代码预览

##### `registry stats` - 仓库统计
```bash
python cli.py registry stats
```
**输出：** 总数、生产就绪数、按后端/GPU的分布

##### `registry history` - 版本历史
```bash
python cli.py registry history silu rtx_4090
```

##### `registry search` - 多条件搜索
```bash
python cli.py registry search --gpu h100_sxm5
python cli.py registry search --op attention
python cli.py registry search --backend cuda --min-bw 0.6
python cli.py registry search --verified-only
```

**参数：**
- `--op` - 算子名称（模糊匹配）
- `--gpu` - GPU型号
- `--backend` - 编程后端
- `--min-bw` - 最低带宽利用率（浮点数）
- `--verified-only` - 仅显示验证通过的

##### `registry export` - 导出为JSON
```bash
python cli.py registry export ./operators.json
```

---

#### **kb** - 编译错误知识库管理（子命令组）

##### `kb stats` - 知识库统计
```bash
python cli.py kb stats
```

##### `kb export` - 导出知识库
```bash
python cli.py kb export ./compile_errors.json
```

##### `kb import` - 导入知识库
```bash
python cli.py kb import ./compile_errors.json --overwrite
```

---

#### **cache** - LLM缓存管理（子命令组）

##### `cache stats` - 缓存统计
```bash
python cli.py cache stats
```

##### `cache clear` - 清空缓存
```bash
python cli.py cache clear
```

---

#### **interactive** - 交互模式（推荐）
```bash
python cli.py interactive
python cli.py interactive --llm qwen --review
python cli.py interactive --llm mock --no-review
```

**参数：**
- `--llm` (默认: `qwen`) - LLM后端
- `--review/--no-review` (默认: 开启) - ReviewLoop验证

**交互命令：**
在提示符下输入：
- 自然语言需求（自动解析和追问）
- `list` - 查看仓库
- `stats` - 统计信息
- `help` - 帮助
- `quit`/`exit` - 退出

---

## 三、示例脚本 (examples/)

### 3.1 全Agent LoRA训练

**文件：** `/remote-home1/hywei/operator_agent_2/examples/full_agent_lora_train.py`

**功能：** 全部训练算子由Agent系统生成，包括SiLU、RMSNorm等

**参数：**
```bash
python examples/full_agent_lora_train.py \
  --mode {custom|baseline|no_finetune|all} \
  --llm {qwen|openai|anthropic|mock} \
  --model /path/to/model \
  --steps <int> \
  --lr <float> \
  --output-dir <path> \
  [--skip-compile]
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|-------|------|
| `--mode` | 选择 | `all` | `custom` (Agent算子) \| `baseline` (PyTorch) \| `no_finetune` (无训练) \| `all` (三种全跑) |
| `--llm` | 选择 | `mock` | LLM后端 |
| `--model` | 路径 | `/remote-home1/share/models/Qwen/Qwen3-0.6B` | 模型路径 |
| `--steps` | 整数 | 100 | 训练步数 |
| `--lr` | 浮点 | 5e-5 | 学习率 |
| `--output-dir` | 路径 | `./output/full_agent` | 输出目录 |
| `--skip-compile` | 标志 | False | 跳过nvcc编译 |

**用法示例：**
```bash
# Mock LLM快速测试
python examples/full_agent_lora_train.py --llm mock --mode custom --steps 5

# Qwen LLM生成真实kernel
python examples/full_agent_lora_train.py --llm qwen --mode custom --steps 20

# 三模式对比
python examples/full_agent_lora_train.py --mode all --llm qwen --steps 20
```

---

### 3.2 Qwen LoRA训练

**文件：** `/remote-home1/hywei/operator_agent_2/examples/qwen_lora_train.py`

**功能：** 使用Agent生成的SiLU kernel进行LoRA训练

**参数：**
```bash
python examples/qwen_lora_train.py \
  --llm {qwen|openai|anthropic|mock} \
  --model /path/to/model \
  --steps <int> \
  --lr <float> \
  --output-dir <path> \
  [--baseline] \
  [--skip-compile]
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|-------|------|
| `--llm` | 选择 | `mock` | LLM后端 |
| `--model` | 路径 | `/remote-home1/share/models/Qwen/Qwen3-0.6B` | 模型路径 |
| `--steps` | 整数 | 10 | 训练步数 |
| `--lr` | 浮点 | 2e-4 | 学习率 |
| `--baseline` | 标志 | False | 使用PyTorch原生算子 |
| `--skip-compile` | 标志 | False | 跳过编译 |
| `--output-dir` | 路径 | `./output/qwen_lora` | 输出目录 |

**用法示例：**
```bash
# Qwen LLM生成kernel
python examples/qwen_lora_train.py --llm qwen --steps 20

# Mock LLM（快速测试）
python examples/qwen_lora_train.py --llm mock --steps 5

# Baseline（PyTorch原生）
python examples/qwen_lora_train.py --baseline --steps 20
```

---

### 3.3 FlashAttention示例

**文件：** `/remote-home1/hywei/operator_agent_2/examples/example_flash_attention.py`

**功能：** 为混合集群（H100 + MI300X）生成FlashAttention v2算子

**用法：**
```bash
python examples/example_flash_attention.py
```

---

## 四、Shell脚本和SLURM任务

### 4.1 主SLURM脚本

**文件：** `/remote-home1/hywei/operator_agent_2/run_lora_slurm.sh`

**SLURM配置：**
```bash
#SBATCH --job-name=operator_lora
#SBATCH --partition=fnlp-4090
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
```

**支持的环境变量：**
| 环境变量 | 默认值 | 说明 |
|---------|-------|------|
| `MODE` | `mock` | 运行模式：`mock` \| `qwen` \| `baseline` \| `full_agent` \| `full_agent_custom_only` |
| `STEPS` | 344 | 训练步数 |
| `LLM` | `mock` | LLM后端 |
| `MODEL` | `/remote-home1/share/models/Qwen/Qwen3-0.6B` | 模型路径 |
| `https_proxy` | `http://10.176.52.116:7890` | 代理设置 |
| `http_proxy` | `http://10.176.52.116:7890` | 代理设置 |

**模式路由：**
```bash
MODE=full_agent          → 全Agent模式（SiLU+RMSNorm由Agent生成）
MODE=full_agent_custom_only → Agent Custom模式（仅自定义算子）
MODE=baseline            → Baseline模式（PyTorch原生）
MODE=qwen               → Qwen LLM模式
MODE=mock               → 默认模式（快速验证）
```

**提交方式：**
```bash
# 默认Mock模式
sbatch run_lora_slurm.sh

# 指定参数
MODE=full_agent LLM=qwen STEPS=300 MODEL=/path/to/model sbatch run_lora_slurm.sh

# 环境变量覆盖
export MODE=qwen
export STEPS=500
sbatch run_lora_slurm.sh
```

---

### 4.2 其他SLURM脚本

| 脚本 | 功能 | 关键参数 |
|------|------|---------|
| `test_qwen_slurm.sh` | 测试Qwen集成 | 无 |
| `tests/hetero/run_phase1_slurm.sh` | Phase 1异构测试 | `PROJECT_DIR`, `PHASE`, `GPU_ID`, `BACKEND`, `LLM_BACKEND` |

**Phase1脚本参数：**
```bash
PROJECT_DIR          - 项目目录
PHASE                - 阶段号（1, 2, 3等）
GPU_ID              - GPU型号（rtx_4090等）
BACKEND             - 后端（cuda等）
LLM_BACKEND         - LLM后端（qwen等）
CONDA_ENV_NAME      - Conda环境名（可选）
```

---

### 4.3 本地测试脚本

| 脚本 | 功能 |
|------|------|
| `test_rmsnorm_scale.sh` | RMSNorm反向传播梯度缩放测试 |
| `test_rmsnorm_grad.sh` | RMSNorm梯度验证 |
| `test_silu_bwd.sh` | SiLU反向传播验证 |
| `test_gradcheck.sh` | PyTorch梯度检查 |
| `test_anomaly.sh` | 梯度异常检测 |
| `test_detect.sh` | 编译错误检测 |
| `test_verify_kernel.sh` | 核心验证 |

---

## 五、配置文件

### 5.1 config.yaml

系统启动时加载 `config/config.yaml`，支持的配置项：

```yaml
llm:
  backend: qwen|openai|anthropic|mock
  model: model_name
  base_url: https://...
  api_key: sk-...
  enable_thinking: true|false

optimizer:
  max_iterations: 3
  convergence_threshold: 0.02
  target_efficiency: 0.7

code_gen:
  # 代码生成配置

verifier:
  min_bandwidth_efficiency: 0.4
```

---

## 六、文档

### 6.1 docs/ 目录

| 文件 | 内容 |
|------|------|
| `ARCHITECTURE.md` | 系统架构详解、各子系统职责、数据流 |
| `EXAMPLES.md` | 四个具体例子展示系统运作全过程 |
| `OPERATORS.md` | 算子定义、规格、验证方法 |
| `TRAINING.md` | 端到端训练流程详解、性能数据 |
| `WORKFLOW.md` | 工作流和最佳实践 |

---

## 七、支持的硬件和后端

### 7.1 GPU 列表
- `ascend_910b`, `ascend_910c` → 后端：`ascendc`
- `rtx_4090`, `h100_sxm5`, `a100_80gb`, `rtx_3090` → 后端：`cuda`
- `mi300x` → 后端：`hip`

### 7.2 LLM 后端
- `qwen` - 阿里千问（DashScope API）
- `openai` - OpenAI GPT
- `anthropic` - Anthropic Claude
- `mock` - 本地模拟（用于测试）

### 7.3 编程后端
- `cuda` - NVIDIA CUDA C++
- `hip` - AMD HIP
- `ascendc` - 昇腾AscendC
- `triton` - OpenAI Triton（实验性）

---

## 八、常见使用场景

### 8.1 快速冒烟（验证系统正确性）
```bash
# 1. 单个算子生成+验证
python cli.py generate --op silu --gpu rtx_4090 --llm mock

# 2. 交互模式测试
python cli.py interactive --llm mock

# 3. LoRA端到端快速验证
python examples/full_agent_lora_train.py --llm mock --mode custom --steps 5
```

### 8.2 真实模型训练
```bash
# 提交SLURM任务（1小时内完成）
MODE=full_agent LLM=qwen STEPS=300 sbatch run_lora_slurm.sh

# 或本地运行
python examples/full_agent_lora_train.py \
  --llm qwen \
  --mode custom \
  --steps 100 \
  --model /path/to/Qwen3-8B
```

### 8.3 查看现有算子
```bash
# 列出所有已生成算子
python cli.py registry list

# 高性能算子搜索
python cli.py registry search --verified-only --min-bw 0.6

# 导出为JSON（分享给团队）
python cli.py registry export
```

### 8.4 自动化集成测试
```bash
# NPU测试
python cli.py npu-test --llm qwen --ops silu gelu rmsnorm

# 知识库管理
python cli.py kb stats
python cli.py kb export compile_errors.json
```

---

## 九、关键环境变量

| 变量 | 说明 | 示例 |
|------|------|------|
| `QWEN_API_KEY` | 阿里千问API密钥 | `sk-xxxxxxxx` |
| `QWEN_BASE_URL` | 千问服务地址 | `https://dashscope.aliyuncs.com/compatible-mode/v1` |
| `OPENAI_API_KEY` | OpenAI密钥 | - |
| `ANTHROPIC_API_KEY` | Anthropic密钥 | - |
| `CUDA_HOME` | CUDA安装路径 | `/usr/local/cuda` |
| `LD_LIBRARY_PATH` | 库路径 | `/usr/local/cuda/lib64` |
| `MODE` | SLURM脚本模式 | `mock`, `qwen`, `full_agent` |
| `STEPS` | 训练步数 | 100, 300, 500 |
| `LLM` | LLM后端 | `mock`, `qwen` |

---

## 十、完整的命令参考

### 最常用的5条命令

```bash
# 1. 交互生成算子（推荐新手）
python cli.py interactive --llm mock

# 2. 显式参数生成
python cli.py generate --op silu --gpu rtx_4090 --llm qwen

# 3. 本地快速端到端测试
python examples/full_agent_lora_train.py --llm mock --mode custom --steps 5

# 4. 查看已有算子
python cli.py registry list

# 5. 提交SLURM任务
sbatch run_lora_slurm.sh
```

---

**文档生成完毕 ✅**

