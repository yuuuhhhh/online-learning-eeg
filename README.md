# 原始 EEG 与实验标签同步采集系统 v2.2

本系统只负责双通道 MindBridge BLE 原始数据、逐样本 ADC counts、实验事件和行为标签的同步采集、保存与完整性检查。采集流程不会重参考、滤波、去伪迹、降采样、切窗、标准化、划分数据集，也不会生成训练可用性结论。

当前协议已冻结（`frozen_for_formal=true`），允许创建 `smoke`、`pilot` 和 `formal` 会话。正式采集请选择 `formal`；该模式强制按顺序完成六个 Block，并禁止提前结束计划休息。

## 启动

1. 双击 `setup_environment.cmd` 安装环境。
2. 双击 `start_integrated_experiment.cmd`。
3. 网页会自动从项目内 `materials/videos` 读取六段正式视频，被试无需选择或上传文件。
4. 完成连续减 7 练习后，连接设备并启动数据流。
5. 采集固定 30 秒睁眼静息原始基线；基线仅作为 `phase=baseline` 保存，只有无 EEG 数据时才阻止继续。
6. 按锁定的 G01–G12 顺序完成六个 Block；关闭采集窗口前保持网页打开，等待事件队列清空。

现场操作见 [实验组使用说明.md](./实验组使用说明.md)。

## 采集期状态

界面只显示连接、数据是否持续到达、样本数、估计采样率、丢包、重复包、断流、饱和、平线、事件队列和原始文件写入状态。异常只作提示和事件记录；已经收到的样本始终保留，不插值、不填充、不删除。

连续约 2 秒没有 EEG 时视频自动暂停，并记录断流；恢复后由实验员确认是否继续。恢复不会生成断流期间的伪造样本。

## 会话输出

```text
data/sub-xxx/ses-xxx/run-YYYYMMDD_HHMMSS/
├─ eeg.csv                         连续逐样本原始 ADC counts
├─ eeg_raw.bin                     按接收顺序保存的原始 33-byte BLE 包
├─ events.csv                      权威事件与标签时间表
├─ acquisition_qc.csv              仅采集状态指标
├─ probes.csv                      thought probe 原始回答汇总
├─ block_ratings.csv               Block 问卷原始回答
├─ quiz_responses.csv              理解题原始回答
├─ metadata.json                   协议、schema、解析器和换算验证状态
├─ session_acquisition_report.json 采集与标签完整性报告
├─ checksums.sha256                eeg.csv 与 eeg_raw.bin 的 SHA-256
└─ session_raw.mat                 可选的原始连续数据、事件、标签和元数据
```

新会话不会生成 `probe_epochs.csv`、`windows.csv`、`eeg_preprocessed.csv`、`preprocessing_report.json` 或任何训练标签。`channel_N_uv` 只是按元数据公式换算的便利字段；硬件换算未确认前，其验证状态明确标为 unverified，`channel_N_raw` 才是权威原始值。

历史会话中的窗口和预处理文件不删除，但应视为 legacy derived output。显式离线分析工具已隔离到 [`offline_analysis/`](./offline_analysis/README.md)，采集程序不会导入或调用它们。

## 测试

```powershell
.\run_tests.cmd
```

测试覆盖原始包/ADC 一致性、丢包不填充、重复包标记、重连分段、异常时继续保存、事件对齐与去重、结束时无自动切窗/预处理、哈希不可变、报告失败不损坏原始文件，以及输出无训练可用性结论。
