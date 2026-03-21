# FPAA 实验代码工作目录

本目录用于集中管理当前实验需要用到的脚本、配置、日志与结果索引。

## 目录结构

- scripts/
  - 00_check_data.sh: Linux 数据完整性与样本数量快速检查。
  - 10_run_fm_baseline.sh: Linux 运行 FM 基线（MFNet 或 SAM_RS）。
  - 20_run_glgan_baseline.sh: Linux 运行 GLGAN LoveDA 对照（R2U/U2R）。
  - 30_run_ablation_matrix.sh: Linux 按配置矩阵批量运行实验。
  - *.ps1: Windows PowerShell 兼容版本（可选）。
- configs/
  - experiment.env.example: 环境变量示例。
  - ablation_matrix.csv: 小规模消融矩阵样例。
- logs/
  - 各次运行的控制台日志。
- results/
  - 建议放主表、消融表、可视化结果索引。
- docs/
  - 实验代码说明文档。

## 推荐执行顺序

1. 先运行 scripts/00_check_data.sh 检查数据。
2. 运行 scripts/10_run_fm_baseline.sh 获取 LoveDA 基线。
3. 运行 scripts/20_run_glgan_baseline.sh 获取 UDA 对照。
4. 按需修改 configs/ablation_matrix.csv，再运行 scripts/30_run_ablation_matrix.sh。

## auto-dl Linux 快速开始

1. 进入目录并赋予执行权限。
2. 先跑数据检查。
3. 再跑基线和对照。

示例命令：

```bash
cd experiments/fpaa
chmod +x scripts/*.sh

# 数据检查
bash scripts/00_check_data.sh /root/autodl-tmp/dataset

# FM 基线: 参数依次是 Backbone Direction Seed PythonExe
bash scripts/10_run_fm_baseline.sh MFNet R2U 3407 python

# GLGAN 对照: 参数依次是 Direction Seed PythonExe
bash scripts/20_run_glgan_baseline.sh R2U 3407 python

# 矩阵批跑: 参数依次是 MatrixCsv PythonExe
bash scripts/30_run_ablation_matrix.sh configs/ablation_matrix.csv python
```

## 注意事项

1. 当前仓库多数训练脚本不是 argparse 风格，核心参数在脚本内部设置。
2. 运行前请先核对各子模块中的数据根目录、DATASET 选择与预训练权重路径。
3. 脚本会把 seed、direction 等信息写入环境变量，便于你在后续代码中读取。
