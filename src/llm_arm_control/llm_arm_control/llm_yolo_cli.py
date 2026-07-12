"""Interactive terminal client for the Fairino LLM-YOLO task server."""

from __future__ import annotations

import getpass
import json
import readline  # noqa: F401  # Enable Unicode-aware input editing.
import select
import sys
import termios
import threading
import time
import tty
import uuid

from action_msgs.msg import GoalStatus
from llm_arm_interfaces.action import ExecutePreview
from llm_arm_interfaces.srv import PreviewCommand
import rclpy
from rclpy.action import ActionClient
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import String
from std_srvs.srv import Trigger

from . import deepseek_credentials


HELP = """Commands:
  <natural language>  create a task preview
  y / n               execute or discard the pending preview
  retry               preview a suspended placement retry; y still confirms
  h                   stop, open gripper, and go Home immediately
  r                   clear stop state; never resumes a cancelled task
  clear               clear the 10-turn language session
  status              show server state
  key set|status|delete  run this at the llm-arm> prompt, not the Linux shell
  help / quit
During execution: SPACE stops immediately; h stops, opens, and goes Home.
"""


def execution_key_effect(key: str) -> tuple[str, bool]:
    """Return the safety command and whether the ROS action should be cancelled."""
    if key in (" ", "\x03"):
        return "stop", True
    if key.lower() == "h":
        # Reset owns the complete stop -> open -> Home recovery. Sending a
        # later action cancel would race that recovery and turn it into stop.
        return "reset", False
    return "", False


def should_offer_cached_box_fallback(terminal_state: str, message: str) -> bool:
    return (
        terminal_state == "HOLDING_RECOVERY"
        and "0 fresh frames, 0 unstable windows" in str(message)
    )


class LlmYoloCli(Node):
    def __init__(self):
        super().__init__("llm_yolo_cli")
        self.preview_client = self.create_client(PreviewCommand, "/llm_arm/preview_command")
        self.status_client = self.create_client(Trigger, "/llm_arm/status")
        self.execute_client = ActionClient(self, ExecutePreview, "/llm_arm/execute_preview")
        self.command_pub = self.create_publisher(String, "/motion_control/command", 10)
        self.clear_session_pub = self.create_publisher(String, "/llm_arm/clear_session", 10)
        self.session_id = uuid.uuid4().hex
        self.preview_id = ""
        self._last_feedback = ""

    @staticmethod
    def _wait_future(future, timeout_sec=None):
        deadline = None if timeout_sec is None else time.monotonic() + float(timeout_sec)
        while not future.done():
            if deadline is not None and time.monotonic() >= deadline:
                return None
            time.sleep(0.02)
        return future.result()

    def preview(self, instruction: str):
        try:
            if not self.preview_client.wait_for_service(timeout_sec=2.0):
                print("Preview service is unavailable. Is llm_yolo_control.launch.py running?")
                return
            req = PreviewCommand.Request(session_id=self.session_id, instruction=instruction)
            response = self._wait_future(
                self.preview_client.call_async(req), timeout_sec=45.0
            )
        except KeyboardInterrupt:
            print("\nPreview cancelled. The CLI is still active.")
            return
        except Exception as exc:
            print(f"Preview request failed: {exc}")
            print("The CLI is still active; enter `key set` at the next llm-arm> prompt if needed.")
            return
        if response is None:
            print("Preview request timed out.")
            return
        self.preview_id = response.preview_id if response.accepted else ""
        if response.preview_json:
            try:
                print(json.dumps(json.loads(response.preview_json), ensure_ascii=False, indent=2))
            except json.JSONDecodeError:
                print(response.preview_json)
        print(response.message)
        if (
            not response.accepted
            and deepseek_credentials.MISSING_MESSAGE in str(response.message)
        ):
            print("Enter `key set` at this llm-arm> prompt, then paste the key when asked.")
            print("Do not run `key set` at the Linux robot@...$ shell.")
        if response.accepted:
            print("Execute this complete plan? [y/N]")

    def _feedback(self, feedback_msg):
        feedback = feedback_msg.feedback
        text = f"[{feedback.step_index}/{feedback.step_count}] {feedback.phase}: {feedback.message}"
        if text != self._last_feedback:
            print(text, flush=True)
            self._last_feedback = text

    def _publish_command(self, command: str):
        self.command_pub.publish(String(data=command))

    def _read_execution_key(self, stream):
        try:
            readable, _, _ = select.select([stream], [], [], 0.0)
        except (OSError, ValueError):
            return ""
        if not readable:
            return False
        key = stream.read(1)
        command, cancel_action = execution_key_effect(key)
        if command == "stop":
            self._publish_command(command)
            print("\nSTOP requested.", flush=True)
        elif command == "reset":
            self._publish_command(command)
            print("\nHOME reset requested.", flush=True)
        return cancel_action

    def execute_pending(self):
        if not self.preview_id:
            print("No executable preview. Enter an instruction first.")
            return
        if not self.execute_client.wait_for_server(timeout_sec=2.0):
            print("Execute action is unavailable.")
            return
        goal = ExecutePreview.Goal(session_id=self.session_id, preview_id=self.preview_id)
        goal_handle = self._wait_future(
            self.execute_client.send_goal_async(goal, feedback_callback=self._feedback),
            timeout_sec=5.0,
        )
        if goal_handle is None or not goal_handle.accepted:
            print("Task execution was rejected.")
            self.preview_id = ""
            return

        result_future = goal_handle.get_result_async()
        stream = sys.stdin
        old_settings = None
        try:
            if stream.isatty():
                old_settings = termios.tcgetattr(stream.fileno())
                tty.setcbreak(stream.fileno())
            while not result_future.done():
                cancel_action = self._read_execution_key(stream) if old_settings is not None else False
                if cancel_action:
                    goal_handle.cancel_goal_async()
                time.sleep(0.02)
        finally:
            if old_settings is not None:
                termios.tcsetattr(stream.fileno(), termios.TCSADRAIN, old_settings)

        wrapped = result_future.result()
        self.preview_id = ""
        if wrapped is None:
            print("Task ended without a result.")
            return
        result = wrapped.result
        ok = result.success and wrapped.status == GoalStatus.STATUS_SUCCEEDED
        print(f"{'SUCCESS' if ok else 'FAILED'} [{result.terminal_state}]: {result.message}")
        if should_offer_cached_box_fallback(result.terminal_state, result.message):
            self.preview("__retry_pending_place__")

    def show_status(self):
        if not self.status_client.wait_for_service(timeout_sec=1.0):
            print("Status service is unavailable.")
            return
        response = self._wait_future(self.status_client.call_async(Trigger.Request()), timeout_sec=2.0)
        print(response.message if response is not None else "Status request timed out.")

    def key_command(self, words):
        subcommand = words[1].lower() if len(words) > 1 else "status"
        try:
            if subcommand == "status":
                print(f"DeepSeek credential source: {deepseek_credentials.credential_status()}")
            elif subcommand == "set":
                try:
                    api_key = getpass.getpass("DeepSeek API key: ")
                except (EOFError, KeyboardInterrupt):
                    print("\nKey entry cancelled; the CLI is still active.")
                    return
                deepseek_credentials.set_deepseek_api_key(api_key)
                print("DeepSeek API key saved to GNOME Keyring.")
            elif subcommand == "delete":
                deleted = deepseek_credentials.delete_deepseek_api_key()
                if deleted:
                    print("Key deleted. Enter `key set` here before the next LLM request.")
                else:
                    print("No Keyring key was configured.")
            else:
                print("Use: key set | key status | key delete")
        except (ValueError, deepseek_credentials.DeepSeekCredentialError) as exc:
            print(f"Credential error: {exc}")

    def run(self):
        print(HELP)
        while rclpy.ok():
            try:
                line = input("llm-arm> ").strip()
            except EOFError:
                print()
                break
            except KeyboardInterrupt:
                print("\nCLI remains active; enter `quit` to exit.")
                continue
            if not line:
                continue
            words = line.split()
            command = words[0].lower()
            if command in ("quit", "exit"):
                break
            if command == "help":
                print(HELP)
            elif command == "key":
                self.key_command(words)
            elif command == "clear":
                self.clear_session_pub.publish(String(data=self.session_id))
                self.session_id = uuid.uuid4().hex
                self.preview_id = ""
                print("Language context cleared.")
            elif command == "status":
                self.show_status()
            elif command == "retry":
                self.preview("__retry_pending_place__")
            elif command == "h":
                self._publish_command("reset")
                print("HOME reset requested.")
            elif command == "r":
                self._publish_command("resume")
                print("Stop state clear requested; cancelled trajectories will not resume.")
            elif command == "y":
                self.execute_pending()
            elif command == "n":
                self.preview_id = ""
                print("Preview discarded.")
            else:
                self.preview(line)


def main(args=None):
    rclpy.init(args=args)
    node = LlmYoloCli()
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    thread = threading.Thread(target=executor.spin, daemon=True)
    thread.start()
    try:
        node.run()
    finally:
        executor.shutdown()
        thread.join(timeout=2.0)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
