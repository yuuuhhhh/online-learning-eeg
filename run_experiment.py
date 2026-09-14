"""MindBridge BLE bridge with synchronized attention-experiment recording."""

from __future__ import annotations

import asyncio
import json
import socket
import threading
import time
import tkinter as tk
import uuid
from pathlib import Path
from tkinter import messagebox, ttk

import tornado.platform.asyncio
import tornado.web
from bleak import BleakClient, BleakScanner

try:
    from .order_balance import recommend_counterbalance_group
    from .session_recorder import CONDITIONS, PROTOCOL_CONFIG, REST_DURATIONS_AFTER_BLOCK, ExperimentRecorder, SessionConfig
except ImportError:
    from order_balance import recommend_counterbalance_group
    from session_recorder import CONDITIONS, PROTOCOL_CONFIG, REST_DURATIONS_AFTER_BLOCK, ExperimentRecorder, SessionConfig


PROJECT_DIR = Path(__file__).resolve().parent
SAMPLE_RATE = 250
OPENBCI_START_BYTE = 0xA0
OPENBCI_STOP_NIBBLE = 0xC0
OPENBCI_PACKET_SIZE = 33
BATCH_SIZE = 10

SERVICE_UUID = uuid.UUID("0000ae30-0000-1000-8000-00805f9b34fb")
WRITE_CHAR_UUID = uuid.UUID("0000ae01-0000-1000-8000-00805f9b34fb")
NOTIFY_CHAR_UUID = uuid.UUID("0000ae02-0000-1000-8000-00805f9b34fb")
TARGET_NAME = "MindBridge-v3.11"
INTEGRATED_BROWSER_MODE = True

udp_target_ip = "127.0.0.1"
udp_target_port = 12345
udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
udp_batch_buffer = bytearray()
rx_ble_buffer = bytearray()
packets_sent_counter = 0

ble_connected = False
ble_client = None
active_recorder: ExperimentRecorder | None = None
background_loop: asyncio.AbstractEventLoop | None = None


def iter_openbci_packets(buffer: bytearray):
    while True:
        start = buffer.find(bytes([OPENBCI_START_BYTE]))
        if start < 0:
            if len(buffer) > OPENBCI_PACKET_SIZE - 1:
                del buffer[: -(OPENBCI_PACKET_SIZE - 1)]
            return
        if start:
            del buffer[:start]
        if len(buffer) < OPENBCI_PACKET_SIZE:
            return

        packet = bytes(buffer[:OPENBCI_PACKET_SIZE])
        if (packet[-1] & 0xF0) != OPENBCI_STOP_NIBBLE:
            del buffer[0]
            continue
        del buffer[:OPENBCI_PACKET_SIZE]
        yield packet


def normalize_two_channel_packet(packet: bytes) -> bytes:
    if len(packet) != OPENBCI_PACKET_SIZE:
        raise ValueError("Invalid packet length")
    if packet[0] != OPENBCI_START_BYTE or (packet[-1] & 0xF0) != OPENBCI_STOP_NIBBLE:
        raise ValueError("Invalid packet framing")
    output = bytearray(OPENBCI_PACKET_SIZE)
    output[0] = OPENBCI_START_BYTE
    output[1] = packet[1]
    output[2:8] = packet[2:8]
    output[8:32] = bytes(24)
    output[32] = 0xC0
    return bytes(output)


def notify_handler(_sender: int, data: bytearray):
    global udp_batch_buffer, packets_sent_counter
    if not data:
        return
    rx_ble_buffer.extend(data)

    for packet in iter_openbci_packets(rx_ble_buffer):
        try:
            normalized = normalize_two_channel_packet(packet)
            if active_recorder is not None:
                active_recorder.record_packet(packet, normalized)
        except Exception as exc:
            print(f"Packet/recording error: {exc}")
            continue

        udp_batch_buffer.extend(normalized)
        packets_sent_counter += 1
        if len(udp_batch_buffer) >= OPENBCI_PACKET_SIZE * BATCH_SIZE:
            try:
                udp_socket.sendto(udp_batch_buffer, (udp_target_ip, udp_target_port))
            except Exception as exc:
                print(f"UDP send error: {exc}")
            udp_batch_buffer = bytearray()


async def send_device_command(command: bytes):
    if ble_client is None or not ble_client.is_connected:
        raise RuntimeError("耳机尚未连接")
    await ble_client.write_gatt_char(WRITE_CHAR_UUID, command, response=True)
    print(f"Sent device command: {command!r}")


async def ble_task():
    global ble_client, ble_connected
    connection_count = 0
    while True:
        try:
            print(f"Scanning for {TARGET_NAME} ...")
            device = await BleakScanner.find_device_by_filter(
                lambda d, ad: d.name == TARGET_NAME, timeout=10
            )
            if device is None:
                print("Device not found; retrying")
                await asyncio.sleep(2)
                continue

            async with BleakClient(device, timeout=10) as client:
                ble_client = client
                ble_connected = True
                print(f"Connected to {device.address}")
                await client.start_notify(NOTIFY_CHAR_UUID, notify_handler)
                if active_recorder is not None:
                    event_type = "device_connected" if connection_count == 0 else "device_reconnected"
                    active_recorder.log_event(event_type, event_value=str(device.address))
                connection_count += 1
                while client.is_connected:
                    await asyncio.sleep(0.1)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            print(f"BLE error: {exc}")
            if active_recorder is not None:
                active_recorder.log_event("device_error", event_value=str(exc))
        finally:
            if ble_connected and active_recorder is not None:
                active_recorder.mark_stream_discontinuity()
                active_recorder.log_event(
                    "device_disconnected", exclude_before_sec=0, exclude_after_sec=0
                )
            ble_connected = False
            ble_client = None
            rx_ble_buffer.clear()
            udp_batch_buffer.clear()
        await asyncio.sleep(2)


async def throughput_monitor():
    global packets_sent_counter
    while True:
        await asyncio.sleep(1)
        saved = active_recorder.received_order if active_recorder is not None else 0
        print(f"EEG rate: {packets_sent_counter} samples/s; saved={saved}")
        packets_sent_counter = 0


async def qc_monitor():
    while True:
        interval = active_recorder.config.qc_interval_sec if active_recorder is not None else 2.0
        await asyncio.sleep(interval)
        if active_recorder is not None:
            try:
                active_recorder.update_qc_snapshot()
            except Exception as exc:
                print(f"QC calculation error: {exc}")


class BaseHandler(tornado.web.RequestHandler):
    def set_default_headers(self):
        self.set_header("Access-Control-Allow-Origin", "*")
        self.set_header("Access-Control-Allow-Headers", "Content-Type")
        self.set_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.set_header("Content-Type", "application/json; charset=utf-8")
        self.set_header("Cache-Control", "no-store")

    def options(self, *_args, **_kwargs):
        self.set_status(204)
        self.finish()


class BoardHandler(BaseHandler):
    def get(self):
        self.finish({
            "board_connected": ble_connected,
            "board_type": "cyton",
            "num_channels": 2,
            "sample_rate_hz": SAMPLE_RATE,
        })


class SessionHandler(BaseHandler):
    def get(self):
        if active_recorder is None:
            self.set_status(409)
            self.finish({"ok": False, "error": "No active session"})
            return
        client_id = self.get_query_argument("client_id", "")
        try:
            pending = int(self.get_query_argument("pending", "0"))
        except ValueError:
            pending = 0
        active_recorder.register_browser(client_id, pending)
        self.finish({
            "ok": True,
            "subject_id": active_recorder.config.subject_id,
            "session_id": active_recorder.config.session_id,
            "run_id": active_recorder.run_id,
            "run_directory": active_recorder.session_dir.name,
            "server_timestamp": active_recorder.clock_time(),
            "shutdown_request_id": active_recorder.shutdown_request_id,
            "counterbalance_group": active_recorder.config.counterbalance_group,
            "study_phase": active_recorder.config.study_phase,
            "protocol_config": PROTOCOL_CONFIG,
            "planned_blocks": active_recorder.planned_blocks,
            "planned_sequence": active_recorder.planned_sequence,
            "conditions": CONDITIONS,
            "b_start_numbers": active_recorder.b_start_numbers,
            "subtraction_practice": active_recorder.subtraction_practice,
            "rest_durations_after_block_sec": REST_DURATIONS_AFTER_BLOCK,
            "probe_config": active_recorder.probe_config,
            "probe_schedules": active_recorder.probe_schedules,
            "current_block": active_recorder.current_block,
            "current_video_id": active_recorder.current_video_id,
            "completed_blocks": sorted(active_recorder.completed_blocks),
            "phase": active_recorder.phase,
            "received_order": active_recorder.received_order,
            "event_count": active_recorder.event_count,
            "qc_ready_for_experiment": active_recorder.qc_ready_for_experiment,
            "qc": active_recorder.get_qc_status(),
        })


class QcHandler(BaseHandler):
    def get(self):
        if active_recorder is None:
            self.set_status(409)
            self.finish({"ok": False, "error": "No active session"})
            return
        self.finish({"ok": True, **active_recorder.get_qc_status()})


class ShutdownReadyHandler(BaseHandler):
    def post(self):
        try:
            if active_recorder is None:
                raise RuntimeError("No active session")
            payload = json.loads(self.request.body.decode("utf-8"))
            active_recorder.acknowledge_shutdown(payload)
            self.finish({"ok": True})
        except Exception as exc:
            self.set_status(409)
            self.finish({"ok": False, "error": str(exc)})


class TcpHandler(BaseHandler):
    def post(self):
        global udp_target_ip, udp_target_port
        payload = json.loads(self.request.body.decode("utf-8"))
        udp_target_ip = str(payload.get("ip", udp_target_ip))
        udp_target_port = int(payload.get("port", udp_target_port))
        self.finish({"connected": True, "ip": udp_target_ip, "port": udp_target_port})


class TriggerHandler(BaseHandler):
    def post(self):
        if active_recorder is None:
            self.set_status(409)
            self.finish({"ok": False, "error": "No active session"})
            return
        try:
            payload = json.loads(self.request.body.decode("utf-8")) if self.request.body else {}
            accepted = active_recorder.handle_browser_event(payload)
            self.finish({
                "ok": True,
                "accepted": accepted,
                "received_order": active_recorder.received_order,
                "block_id": active_recorder.current_block,
                "phase": active_recorder.phase,
            })
        except Exception as exc:
            self.set_status(503 if isinstance(exc, OSError) else 400)
            self.finish({"ok": False, "error": str(exc)})


class StreamStartHandler(BaseHandler):
    async def get(self):
        try:
            await send_device_command(b"S")
            self.finish({"ok": True})
        except Exception as exc:
            self.set_status(409)
            self.finish({"ok": False, "error": str(exc)})


def server_loop(loop: asyncio.AbstractEventLoop):
    asyncio.set_event_loop(loop)
    tornado.platform.asyncio.AsyncIOMainLoop().install()
    app = tornado.web.Application([
        (r"/board", BoardHandler),
        (r"/session", SessionHandler),
        (r"/qc", QcHandler),
        (r"/shutdown/ready", ShutdownReadyHandler),
        (r"/tcp", TcpHandler),
        (r"/trigger", TriggerHandler),
        (r"/event", TriggerHandler),
        (r"/stream/start", StreamStartHandler),
    ])
    try:
        app.listen(80)
        print("HTTP event server listening on port 80")
    except OSError:
        app.listen(9932)
        print("HTTP event server listening on port 9932")
    loop.create_task(ble_task())
    loop.create_task(throughput_monitor())
    loop.create_task(qc_monitor())
    loop.run_forever()


class ExperimentApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("线上学习注意力实验采集")
        self.root.geometry("820x650")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.status_var = tk.StringVar(value="请填写匿名编号并开始会话")
        self.connection_var = tk.StringVar(value="耳机：未连接")
        self.condition_var = tk.StringVar(value="-")
        self._order_refresh_job = None
        self._disconnect_alerted = False
        self._storage_error_alerted = False
        self._build_setup()
        self._refresh_status()

    def _build_setup(self):
        self.setup = ttk.LabelFrame(self.root, text="会话设置", padding=12)
        self.setup.pack(fill="x", padx=12, pady=10)
        self.subject_var = tk.StringVar(value="sub-001")
        self.session_var = tk.StringVar(value="ses-001")
        self.group_var = tk.StringVar(value="G01")
        self.study_phase_var = tk.StringVar(value="pilot")
        self.operator_var = tk.StringVar(value="")
        self.notes_var = tk.StringVar(value="")

        fields = [
            ("匿名被试编号", self.subject_var),
            ("会话编号", self.session_var),
            ("操作员编号（可选）", self.operator_var),
            ("备注（可选）", self.notes_var),
        ]
        for row, (label, variable) in enumerate(fields):
            ttk.Label(self.setup, text=label).grid(row=row, column=0, sticky="w", pady=3)
            ttk.Entry(self.setup, textvariable=variable, width=38).grid(row=row, column=1, sticky="ew")
        ttk.Label(self.setup, text="研究阶段").grid(row=4, column=0, sticky="w", pady=3)
        phase_box = ttk.Combobox(self.setup, textvariable=self.study_phase_var, values=["smoke", "pilot", "formal"], state="readonly", width=12)
        phase_box.grid(row=4, column=1, sticky="w")
        phase_box.bind("<<ComboboxSelected>>", lambda _event: self._refresh_order_recommendation())
        ttk.Label(self.setup, text="Counterbalance组").grid(row=5, column=0, sticky="w", pady=3)
        ttk.Combobox(self.setup, textvariable=self.group_var, values=list(PROTOCOL_CONFIG["counterbalance_groups"]), state="readonly", width=12).grid(
            row=5, column=1, sticky="w"
        )
        ttk.Label(
            self.setup,
            text="G01-G12同时锁定每个Block的视频、条件和位置；smoke/pilot/formal分别循环分配。",
            foreground="#1f4e79",
            font=("Microsoft YaHei", 10, "bold"),
        ).grid(row=6, column=0, columnspan=2, sticky="w", pady=(8, 2))
        ttk.Button(self.setup, text="开始新会话并连接耳机", command=self.start_session).grid(
            row=7, column=0, columnspan=2, pady=10
        )
        self.setup.columnconfigure(1, weight=1)
        ttk.Label(self.root, textvariable=self.status_var, foreground="#1f4e79").pack(fill="x", padx=16)
        self.subject_var.trace_add("write", lambda *_: self._schedule_order_recommendation())
        self._refresh_order_recommendation()

    def _schedule_order_recommendation(self):
        if self._order_refresh_job is not None:
            self.root.after_cancel(self._order_refresh_job)
        self._order_refresh_job = self.root.after(350, self._refresh_order_recommendation)

    def _refresh_order_recommendation(self):
        self._order_refresh_job = None
        recommendation = recommend_counterbalance_group(
            PROJECT_DIR, self.subject_var.get(), self.study_phase_var.get()
        )
        self.group_var.set(recommendation["counterbalance_group"])

    def start_session(self):
        global active_recorder, background_loop
        recommendation = recommend_counterbalance_group(PROJECT_DIR, self.subject_var.get(), self.study_phase_var.get())
        selected_group = self.group_var.get()
        group_override_note = ""
        if selected_group != recommendation["counterbalance_group"]:
            kind = "该被试既有分配" if recommendation["existing_subject"] else "交替分配建议"
            if not messagebox.askyesno(
                "顺序不一致",
                f"{kind}为 {recommendation['counterbalance_group']}，当前选择为 {selected_group}。\n\n"
                "仅在实验方案明确要求时才应继续。确定保留当前选择吗？",
            ):
                self.group_var.set(recommendation["counterbalance_group"])
                return
            group_override_note = (
                f"counterbalance_override: recommended={recommendation['counterbalance_group']}, selected={selected_group}"
            )
        try:
            notes = self.notes_var.get().strip()
            if group_override_note:
                notes = f"{notes}; {group_override_note}" if notes else group_override_note
            config = SessionConfig(
                subject_id=self.subject_var.get(),
                session_id=self.session_var.get(),
                counterbalance_group=selected_group,
                study_phase=self.study_phase_var.get(),
                operator_id=self.operator_var.get(),
                notes=notes,
            )
            active_recorder = ExperimentRecorder(config, PROJECT_DIR)
            session_dir = active_recorder.start()
        except Exception as exc:
            messagebox.showerror("无法开始会话", str(exc))
            return

        background_loop = asyncio.new_event_loop()
        thread = threading.Thread(target=server_loop, args=(background_loop,), daemon=True)
        thread.start()
        self.setup.destroy()
        self._build_controls()
        self.status_var.set(f"会话已开始：{session_dir}")

    def _build_controls(self):
        top = ttk.Frame(self.root, padding=10)
        top.pack(fill="x")
        self.connection_label = ttk.Label(top, textvariable=self.connection_var, font=("Microsoft YaHei", 10, "bold"))
        self.connection_label.pack(side="left")
        for label, command in [("发送 S", b"S"), ("发送 b", b"b"), ("复位 R", b"R"), ("注意力 I", b"I")]:
            ttk.Button(top, text=label, command=lambda c=command: self.send_command(c)).pack(side="right", padx=3)

        sequence = "-".join(active_recorder.planned_sequence)
        ttk.Label(
            self.root,
            text=f"本次分配：{active_recorder.config.counterbalance_group}　{sequence}",
            foreground="#7a2e00",
            font=("Microsoft YaHei", 14, "bold"),
        ).pack(fill="x", padx=16, pady=(0, 8))

        self._build_qc_panel()

        if INTEGRATED_BROWSER_MODE:
            self.root.geometry("900x690")
            ttk.Label(
                self.root,
                text=(
                    "一体化模式：连续减7练习、Block、视频、注意力评分和课后题全部在网页中操作；"
                    "请先启动数据流并完成30秒原始基线采集，网页才会允许开始Block。"
                ),
                wraplength=840,
                foreground="#1f4e79",
                font=("Microsoft YaHei", 10, "bold"),
            ).pack(fill="x", padx=16, pady=12)
            artifacts = ttk.LabelFrame(self.root, text="人工异常标记（仅记录事件，不删除或排除数据）", padding=12)
            artifacts.pack(fill="x", padx=12, pady=8)
            for name in ["咳嗽", "说话", "转头", "电极异常", "其他运动"]:
                ttk.Button(artifacts, text=name, command=lambda n=name: self.mark_artifact(n)).pack(
                    side="left", padx=5
                )
            ttk.Label(
                self.root,
                text="网页须显示顺序、EEG数据流和事件队列状态。关闭本窗口会结束会话并生成原始数据校验报告。",
                foreground="#555",
            ).pack(fill="x", padx=16, pady=10)
            return

        block = ttk.LabelFrame(self.root, text="Block与标签", padding=10)
        block.pack(fill="x", padx=12, pady=6)
        self.block_var = tk.IntVar(value=1)
        self.video_var = tk.StringVar(value="V1")
        self.block_var.trace_add("write", lambda *_: self.update_condition())
        ttk.Label(block, text="Block编号").grid(row=0, column=0, sticky="w")
        ttk.Spinbox(block, from_=1, to=6, textvariable=self.block_var, width=7).grid(row=0, column=1, sticky="w")
        ttk.Label(block, text="条件").grid(row=0, column=2, padx=(20, 3))
        ttk.Label(block, textvariable=self.condition_var, font=("Arial", 11, "bold")).grid(row=0, column=3)
        ttk.Label(block, text="视频编号").grid(row=1, column=0, sticky="w", pady=5)
        ttk.Entry(block, textvariable=self.video_var, width=15).grid(row=1, column=1, sticky="w")
        ttk.Button(block, text="1 开始条件提示", command=self.begin_block).grid(row=2, column=0, padx=3, pady=6)
        ttk.Button(block, text="2 开始视频", command=self.start_video).grid(row=2, column=1, padx=3)
        ttk.Button(block, text="3 结束视频", command=self.end_video).grid(row=2, column=2, padx=3)
        ttk.Button(block, text="6 结束Block", command=self.end_block).grid(row=2, column=3, padx=3)
        self.update_condition()

        rating = ttk.LabelFrame(self.root, text="4 注意力评分", padding=10)
        rating.pack(fill="x", padx=12, pady=6)
        self.rating_var = tk.IntVar(value=3)
        ttk.Label(rating, text="课程注意力 1–5").grid(row=0, column=0, sticky="w")
        ttk.Combobox(rating, textvariable=self.rating_var, values=[1, 2, 3, 4, 5], state="readonly", width=5).grid(row=0, column=1)
        ttk.Button(rating, text="保存评分", command=self.save_rating).grid(row=0, column=2, padx=8)

        summary = ttk.LabelFrame(self.root, text="5 行为结果汇总", padding=10)
        summary.pack(fill="x", padx=12, pady=6)
        self.course_correct = tk.IntVar(value=0)
        self.course_total = tk.IntVar(value=3)
        self.course_rt = tk.StringVar(value="")
        self._summary_row(summary, 0, "课堂题", self.course_correct, self.course_total, self.course_rt, self.save_course_summary)

        artifacts = ttk.LabelFrame(self.root, text="人工异常标记（仅记录事件，不删除或排除数据）", padding=10)
        artifacts.pack(fill="x", padx=12, pady=6)
        for name in ["咳嗽", "说话", "转头", "电极异常", "其他运动"]:
            ttk.Button(artifacts, text=name, command=lambda n=name: self.mark_artifact(n)).pack(side="left", padx=4)

        rest = ttk.Frame(self.root, padding=8)
        rest.pack(fill="x")
        ttk.Button(rest, text="休息开始", command=lambda: self.set_phase("rest", "rest_start")).pack(side="left", padx=5)
        ttk.Button(rest, text="休息结束", command=lambda: self.set_phase("idle", "rest_end")).pack(side="left", padx=5)
        ttk.Label(self.root, text=(
            "A/B仅记录实验条件：A=focused，B=bbbd_subtraction；本阶段不产生瞬时注意力真值标签。"
        ), wraplength=700, foreground="#555").pack(fill="x", padx=16, pady=6)

    def _build_qc_panel(self):
        qc = ttk.LabelFrame(self.root, text="EEG采集状态（每2秒检查数据流，不判断训练可用性）", padding=10)
        qc.pack(fill="x", padx=12, pady=5)

        controls = ttk.Frame(qc)
        controls.grid(row=0, column=0, columnspan=4, sticky="ew", pady=(0, 8))
        ttk.Label(controls, text="原始基线：固定30秒，睁眼静息；仅无数据时阻止继续").pack(side="left")
        ttk.Button(controls, text="开始基线采集", command=self.start_baseline_qc).pack(side="left", padx=8)

        self.baseline_progress_var = tk.DoubleVar(value=0)
        ttk.Progressbar(
            qc, variable=self.baseline_progress_var, maximum=100
        ).grid(row=1, column=0, columnspan=4, sticky="ew", pady=(0, 8))

        for column, text in enumerate(["通道", "饱和率", "最长相同值"]):
            ttk.Label(qc, text=text, font=("Microsoft YaHei", 9, "bold")).grid(
                row=2, column=column, padx=12, pady=2, sticky="w"
            )
        self.qc_metric_vars = []
        for channel in range(2):
            values = {
                "saturation": tk.StringVar(value="--"),
                "flatline": tk.StringVar(value="--"),
            }
            self.qc_metric_vars.append(values)
            ttk.Label(qc, text=f"通道 {channel + 1}").grid(row=3 + channel, column=0, padx=12, sticky="w")
            ttk.Label(qc, textvariable=values["saturation"]).grid(row=3 + channel, column=1, padx=12, sticky="w")
            ttk.Label(qc, textvariable=values["flatline"]).grid(row=3 + channel, column=2, padx=12, sticky="w")

        self.qc_shared_var = tk.StringVar(value="样本：0　采样率：--　丢包率：--　事件队列：0　原始文件：写入中")
        ttk.Label(qc, textvariable=self.qc_shared_var).grid(row=5, column=0, columnspan=4, sticky="w", padx=12, pady=(5, 0))
        self.qc_status_var = tk.StringVar(value="等待数据。先按设备要求发送 S 或 b 启动数据流。")
        self.qc_status_label = ttk.Label(
            qc, textvariable=self.qc_status_var, wraplength=840, font=("Microsoft YaHei", 9, "bold")
        )
        self.qc_status_label.grid(row=6, column=0, columnspan=4, sticky="ew", padx=12, pady=(5, 0))
        for column in range(4):
            qc.columnconfigure(column, weight=1)

    def start_baseline_qc(self):
        try:
            active_recorder.start_baseline(30.0)
            self.status_var.set("30秒睁眼静息原始基线已开始；请坐稳并尽量不动。")
        except Exception as exc:
            messagebox.showerror("无法开始基线采集", str(exc))

    @staticmethod
    def _format_metric(value, suffix=""):
        return "--" if value is None else f"{float(value):.2f}{suffix}"

    def _summary_row(self, parent, row, label, correct_var, total_var, rt_var, command):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=3)
        ttk.Label(parent, text="正确/总数").grid(row=row, column=1)
        ttk.Entry(parent, textvariable=correct_var, width=5).grid(row=row, column=2)
        ttk.Label(parent, text="/").grid(row=row, column=3)
        ttk.Entry(parent, textvariable=total_var, width=5).grid(row=row, column=4)
        ttk.Label(parent, text="平均RT(ms)").grid(row=row, column=5, padx=(12, 2))
        ttk.Entry(parent, textvariable=rt_var, width=9).grid(row=row, column=6)
        ttk.Button(parent, text="保存", command=command).grid(row=row, column=7, padx=8)

    def update_condition(self):
        if active_recorder is None:
            return
        try:
            self.condition_var.set(active_recorder.condition_for_block(int(self.block_var.get())))
        except Exception:
            self.condition_var.set("-")

    def action(self, callback, success_text):
        try:
            callback()
            self.status_var.set(success_text)
        except Exception as exc:
            messagebox.showerror("操作失败", str(exc))

    def begin_block(self):
        self.action(
            lambda: active_recorder.begin_block(int(self.block_var.get()), self.video_var.get()),
            f"Block {self.block_var.get()} 条件提示开始",
        )

    def start_video(self):
        self.action(active_recorder.start_video, "视频开始：EEG开始写入A/B实验条件")

    def end_video(self):
        self.action(active_recorder.end_video, "视频结束：后续评分和答题不会作为训练标签")

    def save_rating(self):
        self.action(
            lambda: active_recorder.log_attention_rating(int(self.rating_var.get())),
            "注意力评分已保存",
        )

    def save_course_summary(self):
        def save():
            active_recorder.set_phase("quiz")
            active_recorder.log_summary(
                "course_quiz_summary", int(self.course_correct.get()), int(self.course_total.get()), self.course_rt.get()
            )
        self.action(save, "课堂题结果已保存")

    def end_block(self):
        def finish():
            finished = int(self.block_var.get())
            active_recorder.end_block()
            if finished < 6:
                self.block_var.set(finished + 1)
                self.video_var.set(f"V{finished + 1}")
        self.action(finish, "Block已结束")

    def mark_artifact(self, artifact):
        self.action(lambda: active_recorder.mark_artifact(artifact), f"已标记：{artifact}")

    def set_phase(self, phase, event):
        self.action(lambda: (active_recorder.set_phase(phase), active_recorder.log_event(event)), f"已记录：{event}")

    def send_command(self, command):
        if background_loop is None:
            return
        future = asyncio.run_coroutine_threadsafe(send_device_command(command), background_loop)
        future.add_done_callback(lambda f: self.root.after(0, self._command_result, f))

    def _command_result(self, future):
        try:
            future.result()
            self.status_var.set("耳机命令已发送")
        except Exception as exc:
            messagebox.showerror("命令发送失败", str(exc))

    def _refresh_status(self):
        if active_recorder is None or not hasattr(self, "qc_status_var"):
            self.connection_var.set("耳机：已连接" if ble_connected else "耳机：未连接/扫描中")
            self.root.after(500, self._refresh_status)
            return

        qc = active_recorder.get_qc_status()
        data_age = qc.get("data_age_sec")
        no_eeg = data_age is not None and data_age >= active_recorder.config.eeg_absence_alert_sec
        if not ble_connected:
            self.connection_var.set("耳机：未连接/正在重连")
            color = "#c62828"
        elif no_eeg:
            self.connection_var.set(f"EEG中断：{data_age:.1f}秒无数据，请重新连接")
            color = "#c62828"
        else:
            rate_text = "等待数据" if data_age is None else f"数据龄 {data_age:.1f}秒"
            self.connection_var.set(f"耳机：已连接｜{rate_text}")
            color = "#17683f" if data_age is not None else "#8a5200"
        self.connection_label.configure(foreground=color)

        for channel_index, channel in enumerate(qc.get("channels", [])):
            if channel_index >= len(self.qc_metric_vars):
                break
            values = self.qc_metric_vars[channel_index]
            values["saturation"].set(self._format_metric(channel.get("saturation_rate_pct"), "%"))
            values["flatline"].set(self._format_metric(channel.get("longest_unchanged_sec"), "秒"))

        loss_text = self._format_metric(qc.get("packet_loss_rate_pct"), "%")
        rate_text = self._format_metric(qc.get("estimated_sample_rate_hz"), " Hz")
        pending = int(qc.get("browser_pending_events", 0))
        save_status = "异常" if qc.get("raw_save_status") == "error" else "写入中"
        self.qc_shared_var.set(
            f"样本：{int(qc.get('received_samples_total', 0))}　采样率：{rate_text}　"
            f"丢包率：{loss_text}　事件队列：{pending}　原始文件：{save_status}"
        )
        if qc.get("raw_save_error") and not self._storage_error_alerted:
            self._storage_error_alerted = True
            messagebox.showerror("原始数据写入失败", str(qc["raw_save_error"]))
        baseline = qc.get("baseline", {})
        target = max(1.0, float(baseline.get("target_sec", 30.0)))
        recorded = float(baseline.get("recorded_sec", 0.0))
        self.baseline_progress_var.set(min(100.0, recorded / target * 100.0))

        if baseline.get("complete") and baseline.get("passed"):
            prefix = "基线原始数据已采集，可以开始正式实验。"
        elif baseline.get("complete"):
            prefix = "基线期间没有持续收到EEG，请恢复数据流后重新采集。"
        elif baseline.get("started"):
            prefix = f"基线采集中：已记录 {recorded:.1f}/{target:.0f} 秒。"
        else:
            prefix = "尚未开始基线采集。"
        details = " ".join(str(item) for item in qc.get("messages", []))
        self.qc_status_var.set(f"{prefix} {details}".strip())
        status_color = {"good": "#17683f", "warning": "#8a5200", "bad": "#c62828"}.get(
            qc.get("status"), "#555"
        )
        self.qc_status_label.configure(foreground=status_color)
        self.root.after(500, self._refresh_status)

    def on_close(self):
        if getattr(self, "_closing", False):
            return
        if active_recorder is None:
            self.root.destroy()
            return
        if not messagebox.askyesno("结束实验", "确定结束本次会话、校验原始文件并生成原始MAT吗？"):
            return
        self._closing = True
        if active_recorder._active:
            active_recorder.request_shutdown()
        self._save_deadline = time.monotonic() + 15.0
        self.status_var.set("正在等待网页事件队列保存，请保持实验网页打开……")
        self.root.after(100, self._finish_close)

    def _finish_close(self):
        global active_recorder
        if active_recorder._active and not active_recorder.shutdown_ready:
            if time.monotonic() < self._save_deadline:
                self.root.after(100, self._finish_close)
                return
            active_recorder.cancel_shutdown()
            self._closing = False
            self.status_var.set("保存未结束：网页队列尚未确认，请恢复网页连接后再次关闭。")
            messagebox.showerror("事件尚未保存完", "网页未确认最后一个事件已写入。采集会话仍保留，请恢复实验网页连接后重试。")
            return
        try:
            mat_path = active_recorder.stop(export_mat=True)
            unresolved = sum(len(items) for items in active_recorder.unresolved_browser_events.values())
            recovery_note = f"\n\n另有 {unresolved} 个事件未通过校验，原始内容已保存到 pending_browser_events.json，需后续核对。" if unresolved else ""
            mat_note = f"\n\n原始MAT：{mat_path}" if mat_path else "\n\n原始MAT导出失败，但CSV、事件和原始包均已保留。"
            messagebox.showinfo("保存完成", f"原始EEG、事件、原始包、校验和与采集报告已保存到：\n{active_recorder.session_dir}{mat_note}{recovery_note}")
        except Exception as exc:
            self._closing = False
            if active_recorder._active:
                active_recorder.cancel_shutdown()
            self.status_var.set("保存未完成；已接收的原始数据仍保留，可再次关闭以重试。")
            messagebox.showerror("保存未完成", f"尚未确认全部保存成功，现有数据保留。\n{exc}")
            return
        if background_loop is not None:
            background_loop.call_soon_threadsafe(background_loop.stop)
        active_recorder = None
        self.root.destroy()


def main():
    root = tk.Tk()
    ExperimentApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
