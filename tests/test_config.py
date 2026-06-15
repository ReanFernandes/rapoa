"""Tests for src/utils/config.py — load_prompts()."""

from pathlib import Path
import pytest
from src.utils.config import load_prompts


def _write_babyai_prompts(base: Path, include_task: bool = False) -> None:
    """Write minimal prompt files under base/prompts/babyai/."""
    family_dir = base / "prompts" / "babyai"
    family_dir.mkdir(parents=True)
    (family_dir / "environment_layer.txt").write_text("env layer content")
    (family_dir / "agent_instructions_rich.txt").write_text("PLAN:\nACTION: agent instructions")
    (family_dir / "descriptor_instructions_rich.txt").write_text("descriptor instructions")
    (family_dir / "agent_instructions_minimal.txt").write_text("PLAN:\nACTION: minimal agent instructions")
    (family_dir / "descriptor_instructions_minimal.txt").write_text("minimal descriptor instructions")
    if include_task:
        task_dir = family_dir / "tasks" / "gotoredball"
        task_dir.mkdir(parents=True)
        (task_dir / "task_layer.txt").write_text("go to red ball task layer")


def test_load_prompts_returns_shared_keys(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_babyai_prompts(tmp_path)
    result = load_prompts()
    assert "environment_layer" in result
    assert "agent_instructions" in result
    assert "descriptor_instructions" in result


def test_load_prompts_no_task_layer_without_task(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_babyai_prompts(tmp_path)
    result = load_prompts()
    assert "task_layer" not in result


def test_load_prompts_returns_task_layer_when_task_given(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_babyai_prompts(tmp_path, include_task=True)
    result = load_prompts(task="gotoredball")
    assert "task_layer" in result
    assert result["task_layer"] == "go to red ball task layer"


def test_load_prompts_no_task_layer_for_unknown_task(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_babyai_prompts(tmp_path)
    result = load_prompts(task="nonexistent_task")
    assert "task_layer" not in result


def test_agent_instructions_contains_critical_markers(tmp_path, monkeypatch):
    """Agent instructions must contain PLAN/ACTION markers the parser depends on."""
    monkeypatch.chdir(tmp_path)
    _write_babyai_prompts(tmp_path)
    result = load_prompts()
    assert "PLAN:" in result["agent_instructions"]
    assert "ACTION:" in result["agent_instructions"]


def test_load_prompts_env_family_parameter(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_babyai_prompts(tmp_path)
    result = load_prompts(env_family="babyai")
    assert "agent_instructions" in result


def test_load_prompts_missing_family_returns_empty(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    _write_babyai_prompts(tmp_path)
    result = load_prompts(env_family="minihack")
    assert result == {}
