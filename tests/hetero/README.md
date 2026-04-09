# 异构 GPU 算子验证方案

## 三阶段测试流程

```
本地Mac（现在就能跑）          AutoDL 租 NVIDIA        华为云/Vast.ai
─────────────────────       ──────────────────      ──────────────────
Phase 0：代码生成质量         Phase 1：真实编译         Phase 2：跨架构验证
  ✅ 无需任何GPU              ✅ CUDA 编译通过           ✅ HIP/AscendC
  ✅ 静态分析                 ✅ 数值正确性              ✅ 异构对比报告
  ✅ Roofline 预测            ✅ 性能基准                预算：¥20–50
  预算：¥0                   预算：¥10–20
```

---

## 第一步：本地运行 Phase 0（现在就能做）

```bash
cd /path/to/operator_agent

# 测试所有目标 GPU 的代码生成质量
python tests/hetero/hetero_test.py --phase 0

# 只测试特定 GPU
python tests/hetero/hetero_test.py --phase 0 --target-gpus rtx_4090 ascend_910b

# 用真实 LLM（代码质量更好，需要 API Key）
export OPENAI_API_KEY=sk-...
python tests/hetero/hetero_test.py --phase 0 --llm openai
```

**输出示例：**
```
Phase 0 通过: 8/10
  flash_attention: 2/4 GPUs pass
    ✅ rtx_4090         backend=cuda       lines=52  static=85%  bound=compute_bound
    ✅ h100_sxm5        backend=cuda       lines=56  static=90%  bound=compute_bound
    ⚠️  mi300x           backend=hip        lines=48  static=65%  bound=compute_bound
    ✅ ascend_910b       backend=ascendc    lines=71  static=80%  bound=memory_bound
```

---

## 第二步：AutoDL 上运行 Phase 1（NVIDIA 真实编译）

### 租机器
1. 登录 autodl.com
2. 选择 **RTX 4090** 或 **vGPU-32GB**（有货的最便宜的）
3. 镜像选择 **PyTorch 2.x + CUDA 12.x**

### 上传代码
```bash
# 方法1：打包上传
zip -r operator_agent.zip . -x "*.pyc" -x "__pycache__/*" -x "output/*"
# 在 AutoDL 控制台上传

# 方法2：git（如果机器能访问 GitHub）
git clone https://github.com/yourname/operator_agent.git
```

### 执行测试
```bash
# SSH 进入 AutoDL 机器后
bash tests/hetero/setup_autodl.sh    # 一键配置环境

# 运行 Phase 1
python tests/hetero/hetero_test.py --phase 1 --gpu rtx_4090 --llm mock
```

---

## 第三步：华为云运行 Phase 2a（Ascend 910B）

### 获取免费额度
1. 注册 [华为云](https://www.huaweicloud.com)
2. 进入 **ModelArts** → 开发环境 → Notebook
3. 新用户有 **300元免费额度**，选择 `Ascend 910B` 规格

### 执行测试
```bash
# 在华为云 Notebook Terminal 中
bash tests/hetero/setup_huaweicloud.sh    # 配置 CANN 环境

python tests/hetero/hetero_test.py \
    --phase 2 --gpu ascend_910b --backend ascendc --llm mock
```

---

## 第四步：Vast.ai 运行 Phase 2b（AMD ROCm）

### 租机器
1. 注册 [vast.ai](https://vast.ai)，充值 $10
2. 搜索过滤：GPU = `RX 7900 XTX` 或 `MI250`，Image = `ROCm`
3. 价格约 $0.3–0.8/h，1–2小时够了

### 执行测试
```bash
bash tests/hetero/setup_vastai_amd.sh    # 配置 ROCm 环境

python tests/hetero/hetero_test.py \
    --phase 2 --gpu mi300x --backend hip --llm mock
```

---

## 生成汇总报告

完成各 Phase 后，在本地执行：

```bash
# 所有结果已保存在 output/hetero_results/
python tests/hetero/hetero_test.py --report
```

生成 `output/hetero_results/report.md`，包含：
- 各算子 × GPU 通过率矩阵
- 跨架构代码特征对比
- 性能预测 vs 实测对比

---

## 测试的算子优先级

| 优先级 | 算子 | 原因 |
|-------|------|------|
| P1 ⭐ | `rmsnorm` | Qwen/LLaMA 必用，逻辑简单，首选验证 |
| P1 ⭐ | `gelu` / `silu` | 纯逐元素，最容易通过，快速冒烟测试 |
| P2 ⭐⭐ | `flash_attention` | 最复杂，最有价值，是系统核心能力体现 |
| P2 ⭐⭐ | `matmul` | 所有模型骨干，性能最关键 |
| P3 | `softmax` | 规约算子，有代表性 |

---

## 预期结果与验收标准

| 指标 | 最低通过线 | 目标 |
|------|----------|------|
| Phase 0 静态分析通过率 | ≥ 70% | ≥ 85% |
| Phase 1 CUDA 编译通过率 | ≥ 80% | ≥ 95% |
| Phase 1 数值误差 | < 1e-2 (fp16) | < 1e-3 |
| Phase 2 Ascend 编译通过率 | ≥ 60% | ≥ 80% |
| Phase 2 AMD HIP 编译通过率 | ≥ 60% | ≥ 75% |

> **重要**：Phase 0 现在就能跑，是最快验证 Agent 核心能力的方式。
> Phase 1/2 是锦上添花，证明生成的代码在真实硬件上可用。
