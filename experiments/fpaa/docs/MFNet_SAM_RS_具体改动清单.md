# MFNet / SAM_RS 具体改动清单（面向 FPAA）

## 0. 目标

将当前仓库从“纯监督或边界/目标辅助监督”扩展为“质量感知伪标签自训练 + 可选轻量适配”，并尽量保持最小侵入改动。

## 1. 必须改（第一阶段）

### 1.1 配置入口统一（避免每次手改常量）

修改文件：

1. MFNet/utils.py
2. SAM_RS/utils.py
3. SAM_RS/utils_loveda.py

建议改动：

1. 将 DATASET、FOLDER、MODE、LOSS、BATCH_SIZE 等硬编码常量改为“环境变量优先，默认值兜底”。
2. 新增以下环境变量读取：
- FPAA_ENABLE_PSEUDO（是否启用伪标签）
- FPAA_QUALITY_MODE（none/conf/conf_boundary/full）
- FPAA_QUALITY_TAU（基础阈值）
- FPAA_PSEUDO_ROOT（伪标签缓存目录）
- EXP_DIRECTION（R2U/U2R）

原因：

- 当前脚本主要是硬编码配置，不利于自动批跑和消融复现。

### 1.2 在 Dataset 中增加伪标签与质量图读取

修改文件：

1. MFNet/utils.py 的 ISPRS_dataset
2. SAM_RS/utils.py 的 ISPRS_dataset
3. SAM_RS/utils_loveda.py 的 ISPRS_dataset

建议改动：

1. 在 __init__ 中新增 pseudo_files 与 quality_files 索引。
2. 在 __getitem__ 中新增伪标签和质量图读取分支。
3. 返回值扩展为：
- MFNet：data, dsm, target, pseudo_label, quality_map
- SAM_RS：data, boundary, object, target, pseudo_label, quality_map
4. 当 FPAA_ENABLE_PSEUDO=false 时，返回全 1 的 quality_map 与空伪标签占位。

原因：

- 训练阶段要做质量门控与加权，必须从 dataloader 直接拿到质量信息。

### 1.3 在训练循环中接入质量感知无监督损失

修改文件：

1. MFNet/train.py
2. SAM_RS/train.py

建议改动：

1. 在 train 函数增加无监督损失项：
- L_sup：现有有监督交叉熵
- L_unsup：伪标签交叉熵（质量门控 + 质量权重）
- 总损失：L = L_sup + λ_u * L_unsup (+ 现有边界/目标损失)
2. 门控逻辑：Q(x) >= tau 时该像素进入无监督损失。
3. tau 支持动态调度（前高后低）。
4. 训练日志打印新增：
- loss_unsup
- mask_ratio（被采纳像素占比）
- quality_mean

原因：

- FPAA 的核心增益来自质量感知伪监督，不接入 train 主循环就无法验证方法。

### 1.4 评估函数增加目标域指标输出一致化

修改文件：

1. MFNet/train.py 的 test
2. SAM_RS/train.py 的 test
3. MFNet/utils.py 的 metrics / metrics_loveda
4. SAM_RS/utils.py（及 utils_loveda.py）的 metrics

建议改动：

1. 固化输出字段：mIoU、mF1、BoundaryF1。
2. 若没有 BoundaryF1，先补边界提取+统计函数并统一打印格式。
3. 评估结果写入 csv（run_id 对齐）。

原因：

- 你后续需要做主表与消融表，指标输出必须标准化。

## 2. 建议改（第二阶段）

### 2.1 新建伪标签离线生成脚本

建议新增文件：

1. MFNet/scripts/export_pseudo.py（或 experiments/fpaa/scripts/export_pseudo.sh + python 实现）

建议功能：

1. 遍历目标域样本，导出：
- pseudo_label（png/tif）
- conf_map
- boundary_consistency_map
- connectivity_map
- quality_map（融合后）
2. 支持缓存复用与断点续跑。

### 2.2 统一 run_id 与日志目录

建议修改：

1. MFNet/train.py
2. SAM_RS/train.py

建议功能：

1. run_id = {dataset}_{direction}_{method}_{seed}_{date}
2. 每次训练落盘：
- logs/{run_id}.log
- results/{run_id}.csv
- ckpt/{run_id}/best.pth

### 2.3 修复方向耦合

现状问题：

1. SAM_RS 中使用 DATASET='Urban' 这类写法，方向信息不足。
2. MFNet 更偏 ISPRS/Hunan，LoveDA 方向并非原生主入口。

建议：

1. 增加 DOMAIN_SOURCE / DOMAIN_TARGET 概念。
2. 由 EXP_DIRECTION 决定加载 source/target 列表。

## 3. 可选增强（第三阶段）

### 3.1 教师-学生 EMA

建议新增：

1. 在 MFNet/train.py 与 SAM_RS/train.py 内增加 teacher model。
2. 采用 EMA 更新 teacher 参数。
3. teacher 负责生成更稳的伪标签。

### 3.2 CAM 提示生成链路

建议新增文件：

1. SAM_RS/SAM_utils.py 扩展或新建 prompt_generator.py

建议功能：

1. 生成 CAM 热图。
2. 提取 point prompts 与 box prompts。
3. 将提示质量分数并入 quality_map。

### 3.3 轻量域适配分支

建议新增：

1. 在训练后半程启用轻量对齐损失（特征统计对齐或弱判别器）。
2. 仅作为增益项，不替代主流程。

## 4. 按文件的最小改动顺序

1. 先改配置读取：MFNet/utils.py、SAM_RS/utils.py、SAM_RS/utils_loveda.py。
2. 再改 Dataset 输出：三处 ISPRS_dataset.__getitem__。
3. 再改训练主循环：MFNet/train.py、SAM_RS/train.py。
4. 最后改评估和日志导出。

## 5. 关键锚点（便于你直接定位）

1. MFNet 训练入口与损失位置：MFNet/train.py
2. MFNet 数据与损失工具：MFNet/utils.py
3. SAM_RS 训练入口与组合损失：SAM_RS/train.py
4. SAM_RS 数据读取与边界/对象损失：SAM_RS/utils.py
5. SAM_RS LoveDA 工具版本：SAM_RS/utils_loveda.py

## 6. 风险提醒

1. 先不要一次性重构全部数据管线，优先保证“可跑通一条 R2U 训练链路”。
2. 先做三组小消融（none/conf/conf+boundary），通过后再加连通性和 EMA。
3. 当前代码中部分路径是历史绝对路径，迁移到 auto-dl 时要统一替换为环境变量驱动。
