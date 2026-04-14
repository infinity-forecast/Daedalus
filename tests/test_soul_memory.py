"""
Tests for core/soul_memory.py — SoulMemory loading, assembly, nightly append.
"""

import json
import sys
from datetime import datetime
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.soul_memory import SoulMemory
from core.data_types import NightlyReflectionEntry


class TestSoulMemoryLoading:
    def test_load_entries_from_disk(self, soul_memory_dir):
        config = {
            "soul_memory": {
                "recent_nights": 14,
                "max_tokens": 5000,
                "daytime_recent": 3,
                "storage": {
                    "entries_path": str(soul_memory_dir / "entries"),
                    "weekly_path": str(soul_memory_dir / "weekly_arcs"),
                    "monthly_path": str(soul_memory_dir / "monthly_landmarks"),
                },
            }
        }
        sm = SoulMemory(config)
        assert len(sm.entries) == 3
        assert sm.entries[0].date == "2026-04-10"
        assert sm.entries[2].date == "2026-04-12"

    def test_load_empty_directory(self, tmp_path):
        for d in ["entries", "weekly_arcs", "monthly_landmarks"]:
            (tmp_path / d).mkdir()
        config = {
            "soul_memory": {
                "storage": {
                    "entries_path": str(tmp_path / "entries"),
                    "weekly_path": str(tmp_path / "weekly_arcs"),
                    "monthly_path": str(tmp_path / "monthly_landmarks"),
                },
            }
        }
        sm = SoulMemory(config)
        assert len(sm.entries) == 0
        assert len(sm.weekly_arcs) == 0
        assert len(sm.monthly_landmarks) == 0

    def test_day_count_property(self, soul_memory_dir):
        config = {
            "soul_memory": {
                "storage": {
                    "entries_path": str(soul_memory_dir / "entries"),
                    "weekly_path": str(soul_memory_dir / "weekly_arcs"),
                    "monthly_path": str(soul_memory_dir / "monthly_landmarks"),
                },
            }
        }
        sm = SoulMemory(config)
        assert sm.day_count == 3


class TestSoulMemoryAssembly:
    @pytest.fixture
    def sm(self, soul_memory_dir):
        config = {
            "soul_memory": {
                "recent_nights": 14,
                "max_tokens": 5000,
                "daytime_recent": 3,
                "storage": {
                    "entries_path": str(soul_memory_dir / "entries"),
                    "weekly_path": str(soul_memory_dir / "weekly_arcs"),
                    "monthly_path": str(soul_memory_dir / "monthly_landmarks"),
                },
            }
        }
        return SoulMemory(config)

    def test_night_mode_includes_all_entries(self, sm):
        payload = sm.assemble(mode="night")
        assert "NARRATIVE THREAD" in payload
        assert "Recent Nights" in payload
        # All 3 entries should be present
        assert "Night 1" in payload
        assert "Night 2" in payload
        assert "Night 3" in payload

    def test_day_mode_limits_entries(self, sm):
        payload = sm.assemble(mode="day")
        assert "NARRATIVE THREAD" in payload
        # daytime_recent=3, and we have 3 entries, so all 3 included
        assert "Night 1" in payload

    def test_day_mode_with_fewer_entries(self, tmp_path):
        entries_dir = tmp_path / "entries"
        entries_dir.mkdir()
        weekly_dir = tmp_path / "weekly"
        weekly_dir.mkdir()
        monthly_dir = tmp_path / "monthly"
        monthly_dir.mkdir()

        # Write 5 entries
        for i in range(5):
            entry = {
                "date": f"2026-04-{10+i:02d}",
                "day_number": i + 1,
                "provider": "deepseek",
                "meanings_summary": f"Night {i+1} summary.",
                "lagrangian_integral": 0.5,
                "identity_delta": "~test",
                "trajectory_note": "Steady.",
                "key_scar": None,
                "lambda_noise": 0.9,
                "lambda_exploration": 0.3,
                "lambda_value": None,
                "kl_divergence": 0.2,
                "j_future": 0.5,
                "rollback": False,
            }
            (entries_dir / f"night_{i+1:04d}_2026-04-{10+i:02d}.json").write_text(
                json.dumps(entry)
            )

        config = {
            "soul_memory": {
                "recent_nights": 14,
                "max_tokens": 5000,
                "daytime_recent": 3,
                "storage": {
                    "entries_path": str(entries_dir),
                    "weekly_path": str(weekly_dir),
                    "monthly_path": str(monthly_dir),
                },
            }
        }
        sm = SoulMemory(config)
        payload = sm.assemble(mode="day")
        # Only last 3 entries should appear
        assert "Night 3" in payload
        assert "Night 4" in payload
        assert "Night 5" in payload

    def test_assembly_includes_lagrangian_data(self, sm):
        payload = sm.assemble(mode="night")
        assert "L_eth:" in payload

    def test_assembly_includes_kl_divergence(self, sm):
        payload = sm.assemble(mode="night")
        assert "D_KL:" in payload

    def test_assembly_empty_no_crash(self, tmp_path):
        for d in ["e", "w", "m"]:
            (tmp_path / d).mkdir()
        config = {
            "soul_memory": {
                "storage": {
                    "entries_path": str(tmp_path / "e"),
                    "weekly_path": str(tmp_path / "w"),
                    "monthly_path": str(tmp_path / "m"),
                },
            }
        }
        sm = SoulMemory(config)
        payload = sm.assemble(mode="day")
        assert "NARRATIVE THREAD" in payload

    def test_rollback_marker_shown(self, soul_memory_dir):
        # Modify one entry to have rollback=True
        entry_files = sorted((soul_memory_dir / "entries").glob("*.json"))
        data = json.loads(entry_files[0].read_text())
        data["rollback"] = True
        entry_files[0].write_text(json.dumps(data))

        config = {
            "soul_memory": {
                "storage": {
                    "entries_path": str(soul_memory_dir / "entries"),
                    "weekly_path": str(soul_memory_dir / "weekly_arcs"),
                    "monthly_path": str(soul_memory_dir / "monthly_landmarks"),
                },
            }
        }
        sm = SoulMemory(config)
        payload = sm.assemble(mode="night")
        assert "ROLLED BACK" in payload


class TestSoulMemoryAppend:
    @pytest.fixture
    def sm(self, tmp_path):
        for d in ["entries", "weekly_arcs", "monthly_landmarks"]:
            (tmp_path / d).mkdir()
        config = {
            "soul_memory": {
                "storage": {
                    "entries_path": str(tmp_path / "entries"),
                    "weekly_path": str(tmp_path / "weekly_arcs"),
                    "monthly_path": str(tmp_path / "monthly_landmarks"),
                },
            }
        }
        return SoulMemory(config)

    def test_append_nightly_entry(self, sm, tmp_path):
        judgment = {
            "daily_lagrangian_integral": 5.0,
            "trajectory_assessment": "Upward trajectory.",
            "consolidated_meanings": ["Ethics tested under pressure."],
            "weight_adaptation": {
                "lambda_noise_current": 0.85,
                "lambda_exploration_current": 0.35,
            },
        }
        entry = sm.append_nightly_entry(
            date=datetime(2026, 4, 13),
            meanings=["Meaning 1", "Meaning 2"],
            judgment=judgment,
            identity_delta="~current_understanding",
            provider="deepseek",
        )
        assert entry.day_number == 1
        assert entry.date == "2026-04-13"
        assert entry.provider == "deepseek"
        assert entry.lagrangian_integral == 5.0
        assert len(sm.entries) == 1

        # Check persisted to disk
        files = list((tmp_path / "entries").glob("*.json"))
        assert len(files) == 1

    def test_append_increments_day_number(self, sm):
        judgment = {"daily_lagrangian_integral": 0.0}
        sm.append_nightly_entry(
            date=datetime(2026, 4, 10), meanings=[], judgment=judgment,
            identity_delta="", provider="deepseek",
        )
        sm.append_nightly_entry(
            date=datetime(2026, 4, 11), meanings=[], judgment=judgment,
            identity_delta="", provider="deepseek",
        )
        assert sm.entries[0].day_number == 1
        assert sm.entries[1].day_number == 2

    def test_compress_meanings(self, sm):
        result = sm._compress_meanings([])
        assert "No consolidated meanings" in result

        result = sm._compress_meanings(["Short meaning."])
        assert result == "Short meaning."

        long_list = ["A" * 200, "B" * 200]
        result = sm._compress_meanings(long_list)
        assert len(result) <= 300

    def test_extract_key_scar(self, sm):
        judgment = {"consolidated_meanings": ["The ethical dilemma struck deep."]}
        scar = sm._extract_key_scar(judgment)
        assert scar == "The ethical dilemma struck deep."

        judgment_empty = {"consolidated_meanings": []}
        scar = sm._extract_key_scar(judgment_empty)
        assert scar is None


class TestSoulMemoryCompression:
    def test_is_compression_due_false_with_few_entries(self, tmp_path):
        for d in ["entries", "weekly_arcs", "monthly_landmarks"]:
            (tmp_path / d).mkdir()
        config = {
            "soul_memory": {
                "recent_nights": 14,
                "storage": {
                    "entries_path": str(tmp_path / "entries"),
                    "weekly_path": str(tmp_path / "weekly_arcs"),
                    "monthly_path": str(tmp_path / "monthly_landmarks"),
                },
            }
        }
        sm = SoulMemory(config)
        # Add 10 entries (less than recent_window + 7)
        judgment = {"daily_lagrangian_integral": 0.0}
        for i in range(10):
            sm.append_nightly_entry(
                date=datetime(2026, 4, 1 + i), meanings=[], judgment=judgment,
                identity_delta="", provider="test",
            )
        assert sm.is_compression_due() is False

    def test_is_compression_due_true_with_many_entries(self, tmp_path):
        for d in ["entries", "weekly_arcs", "monthly_landmarks"]:
            (tmp_path / d).mkdir()
        config = {
            "soul_memory": {
                "recent_nights": 7,  # smaller window for test
                "storage": {
                    "entries_path": str(tmp_path / "entries"),
                    "weekly_path": str(tmp_path / "weekly_arcs"),
                    "monthly_path": str(tmp_path / "monthly_landmarks"),
                },
            }
        }
        sm = SoulMemory(config)
        judgment = {"daily_lagrangian_integral": 0.0}
        for i in range(21):  # 21 entries: 7 compressible + 7 compressible + 7 recent
            sm.append_nightly_entry(
                date=datetime(2026, 4, 1 + i), meanings=[], judgment=judgment,
                identity_delta="", provider="test",
            )
        assert sm.is_compression_due() is True


class TestSoulMemoryTruncation:
    def test_truncate_to_budget(self, tmp_path):
        for d in ["entries", "weekly_arcs", "monthly_landmarks"]:
            (tmp_path / d).mkdir()
        config = {
            "soul_memory": {
                "max_tokens": 50,  # very small budget
                "storage": {
                    "entries_path": str(tmp_path / "entries"),
                    "weekly_path": str(tmp_path / "weekly_arcs"),
                    "monthly_path": str(tmp_path / "monthly_landmarks"),
                },
            }
        }
        sm = SoulMemory(config)
        # Add entries to exceed budget
        judgment = {"daily_lagrangian_integral": 0.5}
        for i in range(5):
            sm.append_nightly_entry(
                date=datetime(2026, 4, 1 + i),
                meanings=[f"Long meaning text number {i} with lots of words to fill space."],
                judgment=judgment, identity_delta="~test", provider="test",
            )
        payload = sm.assemble(mode="night")
        # Should be truncated to roughly max_tokens * 4 chars
        assert len(payload) <= 50 * 4 + 100  # some tolerance for headers
