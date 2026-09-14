# 自动测试记录

- 执行日期：2026-09-14
- Python：28 tests，全部通过
- 浏览器协议检查：28 checks，全部通过
- 浏览器恢复行为：8 tests，全部通过
- Python 全仓语法编译：通过
- Git whitespace 检查：通过

覆盖重点：原始 BLE/ADC 一致性、丢包不填充、重复包标记、重连分段、事件对齐/未对齐/去重、采集异常不阻断保存、基线不产生信号质量门禁、结束时不切窗或预处理、原始文件 SHA-256、报告与 MAT 失败非致命、原始 MAT 不含窗口/epoch。

执行命令：

```powershell
python -m unittest discover -s tests -p "test_*.py"
node tests\test_browser_v21.js
node tests\test_browser_recovery_v21.js
python -m compileall -q .
git diff --check
```

自动测试不能替代真实蓝牙设备 smoke test 和 pilot。
