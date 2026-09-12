# 自动测试记录

- 执行日期：2026-09-12
- Python：22 tests，全部通过
- 浏览器协议检查：22 checks，全部通过
- JavaScript 语法：通过 Node `new Function` 编译检查
- 合成完整性夹具：完整会话 PASS；缺项会话 FAIL 且列出 4 类缺失

执行命令：

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p "test_*.py"
node tests\test_browser_v21.js
.\.venv\Scripts\python.exe tools\generate_validation_sessions.py
```

说明：自动测试不能替代真实蓝牙设备、两名参与者 smoke test 或 6–8 人 pilot。
