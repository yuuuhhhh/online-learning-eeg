"""Real local HTTP requests against recording handlers, without BLE or Tk."""
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import tornado.web
from tornado.testing import AsyncHTTPTestCase

import run_experiment as server
from session_recorder import ExperimentRecorder, SessionConfig


def tearDownModule():
    server.udp_socket.close()


class RecordingHttpTests(AsyncHTTPTestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.recorder = ExperimentRecorder(SessionConfig(subject_id="sub-http"), Path(self.tmp.name))
        self.recorder.start()
        self.recorder_patch = patch.object(server, "active_recorder", self.recorder)
        self.recorder_patch.start()
        super().setUp()

    def get_app(self):
        return tornado.web.Application([
            (r"/session", server.SessionHandler),
            (r"/trigger", server.TriggerHandler),
            (r"/shutdown/ready", server.ShutdownReadyHandler),
        ])

    def tearDown(self):
        super().tearDown()
        self.recorder_patch.stop()
        if self.recorder._active:
            self.recorder.browser_clients.clear()
            self.recorder.stop(export_mat=False)
        self.tmp.cleanup()

    def post(self, path, data):
        return self.fetch(path, method="POST", headers={"Content-Type": "application/json"}, body=json.dumps(data))

    def payload(self, **changes):
        return {"recorder_run_id": self.recorder.run_id, "subject_id": "sub-http", "session_id": "ses-001",
                "block_id": "1", "condition": "A", "client_id": "browser-http", "client_event_id": "last-event",
                "event_type": "browser_shutdown_ready", "event_timestamp": self.recorder.clock_time(), **changes}

    def test_session_sync_event_ack_and_final_save(self):
        response = self.fetch("/session?client_id=browser-http&pending=1")
        self.assertEqual(response.code, 200)
        session = json.loads(response.body)
        self.assertEqual(session["run_id"], self.recorder.run_id)
        self.assertIsInstance(session["server_timestamp"], float)
        self.assertEqual(session["conditions"]["A"]["condition_type"], "focused")
        self.assertEqual(session["conditions"]["B"]["condition_type"], "bbbd_subtraction")
        self.assertEqual(list(session["rest_durations_after_block_sec"].values()), [30, 30, 180, 30, 30])
        self.assertEqual(len(session["b_start_numbers"]), 3)
        request_id = self.recorder.request_shutdown()
        event = self.payload()
        response = self.post("/trigger", event)
        self.assertTrue(json.loads(response.body)["accepted"])
        response = self.post("/trigger", event)
        self.assertFalse(json.loads(response.body)["accepted"])
        response = self.post("/shutdown/ready", {"run_id": session["run_id"], "request_id": request_id,
                                                  "client_id": "browser-http", "pending": 0, "last_event_id": "last-event"})
        self.assertEqual(response.code, 200)
        self.assertTrue(self.recorder.shutdown_ready)
        self.recorder.stop(export_mat=True)
        self.assertTrue(self.recorder.mat_path.exists())
        self.assertIn("last-event", self.recorder.events_path.read_text(encoding="utf-8"))

    def test_wrong_run_returns_error_and_does_not_change_event_count(self):
        count = self.recorder.event_count
        response = self.post("/trigger", self.payload(recorder_run_id="other-run"))
        self.assertEqual(response.code, 400)
        self.assertIn("run_id", json.loads(response.body)["error"])
        self.assertEqual(count, self.recorder.event_count)

    def test_shutdown_before_final_event_is_not_acknowledged(self):
        self.fetch("/session?client_id=browser-http&pending=1")
        request_id = self.recorder.request_shutdown()
        response = self.post("/shutdown/ready", {"run_id": self.recorder.run_id, "request_id": request_id,
                                                  "client_id": "browser-http", "pending": 0, "last_event_id": "not-written"})
        self.assertEqual(response.code, 409)
        self.assertFalse(self.recorder.shutdown_ready)

    def test_disk_error_is_retryable_http_error(self):
        with patch.object(self.recorder, "handle_browser_event", side_effect=OSError("write unavailable")):
            response = self.post("/trigger", self.payload())
        self.assertEqual(response.code, 503)

    def test_window_retries_export_without_reopening_closed_event_queue(self):
        self.recorder.stop(export_mat=False)
        self.recorder.browser_clients["previous-browser"] = {}
        app = object.__new__(server.ExperimentApp)
        app.root = Mock()
        app.status_var = Mock()
        with patch.object(self.recorder, "request_shutdown") as request, \
                patch.object(server.messagebox, "askyesno", return_value=True), \
                patch.object(server.messagebox, "showinfo"):
            app.on_close()
            request.assert_not_called()
            app._finish_close()
        app.root.destroy.assert_called_once()
        self.assertTrue(self.recorder.mat_path.exists())
