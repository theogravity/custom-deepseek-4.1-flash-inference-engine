"""Exercise activation and rollback without using Docker or touching serving GPUs."""
import contextlib
import importlib.machinery
import importlib.util
from pathlib import Path
import subprocess
import unittest
from unittest.mock import patch


launcher = Path(__file__).resolve().parents[1] / "engine"
loader = importlib.machinery.SourceFileLoader("dsv41_launcher", str(launcher))
spec = importlib.util.spec_from_loader(loader.name, loader)
engine = importlib.util.module_from_spec(spec)
loader.exec_module(engine)


class SwitchTests(unittest.TestCase):
    candidate = Path("/build/candidate")

    def setUp(self):
        stack = self.enterContext(contextlib.ExitStack())
        stack.enter_context(patch.object(engine, "build", return_value=self.candidate))
        self.previous = stack.enter_context(patch.object(engine, "previous_service", return_value="stock"))
        self.idle = stack.enter_context(patch.object(engine, "idle"))
        self.run = stack.enter_context(patch.object(engine, "run"))

    def test_absent_container_status(self):
        with patch.object(engine.subprocess, "check_output", return_value=""), patch("builtins.print") as display:
            engine.status()
        display.assert_called_once_with("stopped (container absent)")
        self.run.assert_not_called()

    def test_nominal_activation_uses_candidate_and_waits(self):
        engine.switch(True)
        self.idle.assert_called_once_with()
        calls = [call for call in self.run.call_args_list if "up" in call.args]
        self.assertEqual(len(calls), 1)
        self.assertIn(str(engine.ROOT / "compose.native.yml"), calls[0].args)
        self.assertIn("--wait", calls[0].args)
        self.assertEqual(calls[0].kwargs["env"]["DSV41_ENGINE_BUILD"], str(self.candidate))

    def test_busy_service_keeps_serving(self):
        self.idle.side_effect = SystemExit("active requests")
        with self.assertRaises(SystemExit):
            engine.switch(True)
        self.assertFalse(any("up" in c.args or "stop" in c.args for c in self.run.call_args_list))

    def test_operator_recovers_when_metrics_time_out(self):
        self.idle.side_effect = TimeoutError("worker did not answer")
        engine.switch(False, interrupt=True)
        self.idle.assert_not_called()
        self.assertTrue(any("up" in c.args for c in self.run.call_args_list))

    def test_conflicting_port_or_gpu_refuses_activation(self):
        self.previous.return_value = None
        for port_status, gpu_pids in [(0, ""), (111, "1458\n")]:
            with self.subTest(port_status=port_status, gpu_pids=gpu_pids):
                with patch("socket.socket") as sock, patch.object(engine.subprocess, "check_output", return_value=gpu_pids):
                    sock.return_value.__enter__.return_value.connect_ex.return_value = port_status
                    with self.assertRaises(SystemExit):
                        engine.switch(True)
                self.assertFalse(any("up" in c.args or "stop" in c.args for c in self.run.call_args_list))

    def test_failure_or_interrupt_preserves_previous_runtime(self):
        for previous, failure in [
            (None, subprocess.CalledProcessError(1, "docker compose up")),
            ("stock", subprocess.CalledProcessError(1, "docker compose up")),
            (Path("/build/prior"), subprocess.CalledProcessError(1, "docker compose up")),
            (self.candidate, KeyboardInterrupt()),
        ]:
            with self.subTest(previous=previous, failure=type(failure).__name__):
                self.previous.return_value = previous
                self.run.reset_mock()
                with patch.object(engine, "check_free"), patch.object(engine, "activate", side_effect=[failure, None]) as up:
                    with self.assertRaises(type(failure)):
                        engine.switch(True)
                self.assertTrue(any("stop" in c.args for c in self.run.call_args_list))
                versions = [call.args[0] for call in up.call_args_list]
                self.assertEqual(versions, [self.candidate, previous] if previous else [self.candidate])


if __name__ == "__main__":
    unittest.main()
