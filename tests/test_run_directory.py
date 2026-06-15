"""Tests for run directory utilities.

These functions live in src/logging/run_directory.py.
"""

import json
from pathlib import Path

import pytest

from src.logging.run_directory import parse_env_id, create_run_directory, save_run_config


# ---------------------------------------------------------------------------
# parse_env_id
# ---------------------------------------------------------------------------

def test_babyai_standard():
    assert parse_env_id("BabyAI-GoToRedBall-v0") == ("BabyAI", "GoToRedBall")


def test_babyai_multisegment_task():
    # Task name contains a hyphen — only the first split should happen
    assert parse_env_id("BabyAI-GoToRedBallNoDists-v0") == ("BabyAI", "GoToRedBallNoDists")


def test_minigrid_compound_task():
    assert parse_env_id("MiniGrid-Empty-5x5-v0") == ("MiniGrid", "Empty-5x5")


def test_strips_version_suffix():
    family, task = parse_env_id("BabyAI-GoToDoor-v0")
    assert not task.endswith("v0")
    assert not task.endswith("-v0")


def test_no_hyphen_fallback():
    family, task = parse_env_id("SomeEnv")
    assert family == "SomeEnv"
    assert task == "unknown"


# ---------------------------------------------------------------------------
# create_run_directory
# ---------------------------------------------------------------------------

def test_creates_directory(tmp_path):
    run_dir = create_run_directory(
        base_log_directory=tmp_path,
        env_family="BabyAI",
        task="GoToRedBall",
        model="openai/gpt-oss-20b",
        pipeline="with_descriptor",
        prompt_variant="rich",
        conversation_mode="single_turn",
        reasoning=False,
    )
    assert run_dir.exists()
    assert run_dir.is_dir()


def test_directory_structure(tmp_path):
    run_dir = create_run_directory(
        base_log_directory=tmp_path,
        env_family="BabyAI",
        task="GoToRedBall",
        model="openai/gpt-oss-20b",
        pipeline="with_descriptor",
        prompt_variant="rich",
        conversation_mode="single_turn",
        reasoning=False,
    )
    parts = run_dir.parts
    assert "BabyAI" in parts
    assert "GoToRedBall" in parts
    assert "with_descriptor" in parts
    assert "single_turn" in parts
    assert "no_reasoning" in parts


def test_model_slash_sanitised(tmp_path):
    run_dir = create_run_directory(
        base_log_directory=tmp_path,
        env_family="BabyAI",
        task="GoToRedBall",
        model="openai/gpt-oss-20b",
        pipeline="with_descriptor",
        prompt_variant="rich",
        conversation_mode="single_turn",
        reasoning=False,
    )
    # Forward slash in model name must not create unintended nesting
    assert "openai--gpt-oss-20b" in str(run_dir)
    assert "openai/gpt-oss-20b" not in str(run_dir)


def test_reasoning_flag_in_path(tmp_path):
    run_dir_on = create_run_directory(
        base_log_directory=tmp_path,
        env_family="BabyAI",
        task="GoToDoor",
        model="model",
        pipeline="with_descriptor",
        prompt_variant="rich",
        conversation_mode="single_turn",
        reasoning=True,
    )
    run_dir_off = create_run_directory(
        base_log_directory=tmp_path,
        env_family="BabyAI",
        task="GoToDoor",
        model="model",
        pipeline="with_descriptor",
        prompt_variant="rich",
        conversation_mode="single_turn",
        reasoning=False,
    )
    assert "reasoning" in str(run_dir_on)
    assert "no_reasoning" in str(run_dir_off)


# ---------------------------------------------------------------------------
# save_run_config
# ---------------------------------------------------------------------------

def test_saves_json(tmp_path):
    config = {"env": "BabyAI-GoToRedBall-v0", "episodes": 10}
    save_run_config(tmp_path, config)
    config_path = tmp_path / "run_config.json"
    assert config_path.exists()
    loaded = json.loads(config_path.read_text())
    assert loaded["env"] == "BabyAI-GoToRedBall-v0"
    assert loaded["episodes"] == 10
