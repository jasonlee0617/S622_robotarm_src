#!/usr/bin/env python3
import json
import os

from NodeGraphQt import BaseNode
from openai import OpenAI

from utils.general import find_nodes_folder
from .llm_yolo_pick_logic import make_preview, parse_selected_index, preview_is_confirmable


__all__ = ["LLMYoloPickPreviewNode"]


class LLMYoloPickPreviewNode(BaseNode):
    __identifier__ = find_nodes_folder(__file__)[1]
    NODE_NAME = "LLM YOLO pick preview"

    def __init__(self):
        super().__init__()
        self.add_input("text_in")
        self.add_input("yolo_obb")
        self.add_output("preview")
        self.add_checkbox("confirm_pick", text="Confirm pick")
        self.add_text_input("preview_max_age_sec", label="Preview max age (s)")
        self.set_property("preview_max_age_sec", "2.0")
        self.text_out = ""
        self._preview = None

        api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("Set DEEPSEEK_API_KEY in the GraphExecuter terminal")
        self.client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

    def _emit(self, message):
        self.text_out = message
        self.messageSignal.emit(message)

    def _upstream_node(self, input_index):
        ports = self.input(input_index).connected_ports()
        return ports[0].node() if ports else None

    def _preview_max_age_sec(self):
        try:
            return max(0.0, float(self.get_property("preview_max_age_sec")))
        except (TypeError, ValueError):
            return 2.0

    def _confirm(self, yolo_node):
        if self._preview is None:
            self.set_property("confirm_pick", False)
            self._emit("No pending preview; generate one before confirming.")
            return
        current_stamp = yolo_node.current_pick_frame_stamp_ns()
        if not preview_is_confirmable(self._preview, current_stamp, self._preview_max_age_sec()):
            self._preview = None
            self.set_property("confirm_pick", False)
            self._emit("Preview expired or detection changed; generate a new preview before confirming.")
            return
        if not yolo_node.publish_pick_target(self._preview.target):
            self._emit("Pick confirmation failed: target publish was rejected.")
            return
        self._emit(
            "Confirmed pick: %s[%d], target=(%.3f, %.3f, %.3f)." % (
                self._preview.class_name,
                self._preview.index,
                *self._preview.target,
            )
        )
        self._preview = None
        self.set_property("confirm_pick", False)

    def execute(self):
        text_node = self._upstream_node(0)
        yolo_node = self._upstream_node(1)
        if text_node is None or yolo_node is None:
            self._emit("Connect Text input and yolo_obb before running this node.")
            return
        if not all(hasattr(yolo_node, name) for name in (
            "get_pick_candidates", "preview_pick_candidate", "current_pick_frame_stamp_ns", "publish_pick_target"
        )):
            self._emit("The yolo_obb input is not a compatible YoloObbNode.")
            return
        if self.get_property("confirm_pick"):
            self._confirm(yolo_node)
            return

        text_in = str(getattr(text_node, "text_out", "")).strip()
        candidates = yolo_node.get_pick_candidates()
        if not text_in:
            self._emit("Enter a pick instruction before generating a preview.")
            return
        if not candidates:
            self._emit("No current YOLO candidates are available for preview.")
            return

        messages = [
            {
                "role": "system",
                "content": (
                    "Select exactly one candidate for the user's pick request. "
                    "Return only JSON: {\\\"selected_index\\\": integer}. "
                    "You may choose only an index listed in candidates."
                ),
            },
            {
                "role": "user",
                "content": json.dumps({"instruction": text_in, "candidates": candidates}, ensure_ascii=False),
            },
        ]
        try:
            completion = self.client.chat.completions.create(messages=messages, model="deepseek-chat")
            selected_index = parse_selected_index(completion.choices[0].message.content, candidates)
        except Exception as exc:
            self._emit(f"LLM pick preview rejected: {exc}")
            return

        candidate = yolo_node.preview_pick_candidate(selected_index)
        if candidate is None:
            self._emit("LLM selected a candidate, but its depth or TF target is unavailable.")
            return
        self._preview = make_preview(candidate)
        self._emit(
            "Preview: %s[%d], target=(%.3f, %.3f, %.3f). Enable Confirm pick and run again to execute." % (
                self._preview.class_name,
                self._preview.index,
                *self._preview.target,
            )
        )

    def set_messageSignal(self, messageSignal):
        self.messageSignal = messageSignal
