import pytest


pytest.importorskip("rclpy")

from llm_arm_control.llm_yolo_cli import (  # noqa: E402
    execution_key_effect,
    should_offer_cached_box_fallback,
)


def test_space_and_ctrl_c_stop_and_cancel_action():
    assert execution_key_effect(" ") == ("stop", True)
    assert execution_key_effect("\x03") == ("stop", True)


def test_home_reset_does_not_race_with_action_cancel():
    assert execution_key_effect("h") == ("reset", False)
    assert execution_key_effect("H") == ("reset", False)


def test_other_execution_keys_are_ignored():
    assert execution_key_effect("y") == ("", False)


def test_cached_box_fallback_is_offered_only_for_zero_fresh_frames():
    message = (
        "box was not stable within 5 seconds: 0 fresh frames, "
        "0 unstable windows; requires 5 stable frames"
    )
    assert should_offer_cached_box_fallback("HOLDING_RECOVERY", message)
    assert not should_offer_cached_box_fallback("FAILED", message)
    assert not should_offer_cached_box_fallback(
        "HOLDING_RECOVERY",
        "box was not stable within 5 seconds: 3 fresh frames, 1 unstable windows",
    )
