# 线上学习 EEG 一体化实验系统 v2.1

这是从 v1.3 独立复制并按实验方案 v2.1 改造的 pilot 版本。原 v1.3 没有被覆盖。系统整合双通道 MindBridge BLE EEG 采集、六段课堂视频、BBBD 脑内连续减 7、thought probe、完整 block 问卷、课堂题、实时 QC、规范化表格、MAT 导出和会话完整性报告。

当前配置 `frozen_for_formal=false`，所以程序会拒绝创建 `formal` 会话。完成 2 人 smoke test 和 6–8 人 pilot、由负责人冻结参数后，才能把配置升级为正式版本。

## 关键改动

- 删除新会话中的屏幕口算、F/J 作答和旧 `math_*`/`arithmetic_*` 事件；历史数据仍可读取。
- 用 G01–G12 锁定每名被试的六个 `video_id × condition × block_order`，smoke、pilot、formal 分开循环分配。
- A 为 `focused` 条件，B 为 `bbbd_subtraction` 条件；A/B 只是 condition weak label。瞬时注意标签只来自 probe 的 `ON/OFF/AMBIGUOUS`。
- 每个 block 预生成 4 个 probe。首个不早于 45 秒、末个距结尾至少 30 秒、相邻至少 45 秒；无法容纳 4 个时禁止开始。
- `K` 为 self-caught 标记键；导出中提供 ±2 秒 motor mask 和按键前 10 秒 sensitivity mask。
- 每个 block 后填写 4 个通用量表；B 额外填写减 7 执行比例、难度和最终数字；每段视频有 4 道四选一理解题。
- 休息顺序固定为 30、30、180、30、30 秒。pilot/formal 不能提前结束休息。
- 采前 QC 固定为 60 秒睁眼静息；人工放行必须记录实验员和原因。
- 连续约 2 秒没有新 EEG 时，视频自动暂停；数据恢复后须由实验员确认同步再继续。
- 结束时生成 `probes.csv`、`block_ratings.csv`、`quiz_responses.csv`、`probe_epochs.csv`、`windows.csv`、`session_qc_report.json` 和 `session.mat`。

## 50 Hz 处理政策

按本次负责人要求，采集阶段完全不计算、不显示、也不以 50 Hz 指标判定 QC。`qc.csv` 没有 50 Hz 功率比字段。

原始 `eeg.csv` 和 `eeg_raw.bin` 永不改写。离线预处理时运行：

```powershell
.\.venv\Scripts\python.exe preprocess_eeg.py data\sub-001\ses-001\run-xxxx
```

程序在连续数据段内对副本做 50 Hz 零相位陷波/带阻，输出 `eeg_preprocessed.csv` 和 `preprocessing_report.json`。术语上，去除 50 Hz 应使用“陷波/带阻”；“50 Hz 带通”会保留 50 Hz，效果相反。

## 首次安装与启动

要求 Windows 10/11、Python 3.10+、Chrome/Edge 和可用蓝牙。

1. 双击 `setup_environment.cmd`，在项目内创建 `.venv` 并安装依赖。
2. 双击 `start_integrated_experiment.cmd`。
3. 在采集窗口填写匿名被试号、会话号、研究阶段和实验员编号；采用系统推荐的 G 组。
4. 网页一次选择 `materials/videos/` 下的全部六段视频，系统会按配置自动匹配并核对时长。
5. 先完成减 7 练习确认，再连接设备并完成固定 60 秒睁眼静息 QC。
6. 严格按 Block 1–6 运行；每个 block 的视频和条件由 G 组锁定。
7. Block 6 完成后查看网页预检查；保持网页打开，再关闭采集窗口，等待权威文件全部落盘。

现场步骤与异常处理见 [实验组使用说明.md](./实验组使用说明.md)。

## 材料与单一配置源

所有可冻结参数都集中在 `config/protocol_v2.1.json`：

- 软件、协议、schema 和材料版本；
- G01–G12 表；
- 六段视频文件名、实际时长和每段 4 道题；
- probe、休息、mask 和 QC 参数；
- 正式阶段冻结开关；
- 采集端不测 50 Hz、离线副本做 50 Hz 陷波的政策。

六段正式视频在 `materials/videos/`，原始题库在 `materials/source/课程题目.docx`。目前没有收到 2–3 段候选备用视频，配置中的 `candidate_videos` 因此为空；这是材料缺口，不由程序虚构补齐。

## 输出目录

每次会话自动建立：

```text
data/sub-xxx/ses-xxx/run-YYYYMMDD_HHMMSS/
├─ eeg.csv                    原始逐样本 EEG 与 condition weak label
├─ eeg_raw.bin                原始 BLE 包
├─ events.csv                 一等字段化事件表
├─ qc.csv                     采集 QC；不含 50 Hz 指标
├─ probes.csv                 每个 probe 一行
├─ block_ratings.csv          每个 block 一行问卷
├─ quiz_responses.csv         每道理解题一行
├─ probe_epochs.csv           probe 前 10 秒 epoch 索引
├─ windows.csv                Dataset A/B 的 4 秒窗、2 秒步长索引与 mask
├─ metadata.json              版本、哈希、G 组、实际流程和参数
├─ session_qc_report.json     PASS/WARN/FAIL 与缺失项、会话 QC 汇总
└─ session.mat                上述新增表、索引、报告和版本信息
```

`probe_attention` 只能为 `ON/OFF/AMBIGUOUS/NONE`。窗口 `quality_label` 为 `GOOD/USABLE/REJECT`，所有排除通过 `reject_reason` 和 mask 表达，不删除 EEG。训练/测试必须按被试、会话、完整 block 或完整 `probe_id` 分组，不能随机拆散同一 block/probe 的滑窗。

字段、枚举和值域见 [docs/数据字典_v2.1.md](./docs/数据字典_v2.1.md)。逐项改造结论见 [docs/修改清单逐项验收_v2.1.md](./docs/修改清单逐项验收_v2.1.md)。

## 测试与验收样例

```powershell
.\run_tests.cmd
```

当前自动验收为 22 个 Python 测试和 22 个浏览器协议检查。`validation_samples/完整会话_PASS/` 与 `validation_samples/缺项会话_FAIL/` 是合成完整性测试夹具，分别应生成 PASS 和明确列出缺失 probe、问卷、课堂题及 B 专属评分的 FAIL。它们不是科学 EEG 数据。

旧版文档和测试已移入 `docs/legacy_v1.3/` 与 `legacy_tests_v1.3/`，仅用于历史复现，禁止指导新采集。
