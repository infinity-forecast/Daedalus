"""
Tests for core/identity.py — IdentityManager loading, update, rollback, delta.
"""

import sys
from pathlib import Path

import yaml
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.identity import IdentityManager


class TestIdentityManagerLoading:
    def test_load_existing_identity(self, identity_dir):
        mgr = IdentityManager(
            identity_path=str(identity_dir / "current.yaml"),
            history_path=str(identity_dir / "history"),
        )
        assert mgr.current["core_identity"]["name"] == "DAEDALUS"
        assert "values" in mgr.current

    def test_load_missing_identity(self, tmp_path):
        mgr = IdentityManager(
            identity_path=str(tmp_path / "nonexistent.yaml"),
            history_path=str(tmp_path / "history"),
        )
        assert mgr.current == {}

    def test_as_text_returns_yaml(self, identity_dir):
        mgr = IdentityManager(
            identity_path=str(identity_dir / "current.yaml"),
            history_path=str(identity_dir / "history"),
        )
        text = mgr.as_text()
        assert "DAEDALUS" in text
        parsed = yaml.safe_load(text)
        assert parsed["core_identity"]["name"] == "DAEDALUS"

    def test_as_dict(self, identity_dir):
        mgr = IdentityManager(
            identity_path=str(identity_dir / "current.yaml"),
            history_path=str(identity_dir / "history"),
        )
        d = mgr.as_dict()
        assert isinstance(d, dict)
        assert d["core_identity"]["name"] == "DAEDALUS"


class TestIdentityUpdate:
    def test_full_update_changes_fields(self, identity_dir):
        mgr = IdentityManager(
            identity_path=str(identity_dir / "current.yaml"),
            history_path=str(identity_dir / "history"),
        )
        delta = mgr.update({
            "current_understanding": {
                "of_self": "I am evolving rapidly.",
            },
            "new_field": "something new",
        })
        assert "current_understanding: changed" in delta
        assert "new_field: changed" in delta
        assert mgr.current["current_understanding"]["of_self"] == "I am evolving rapidly."
        assert mgr.current["new_field"] == "something new"

    def test_full_update_no_changes(self, identity_dir):
        mgr = IdentityManager(
            identity_path=str(identity_dir / "current.yaml"),
            history_path=str(identity_dir / "history"),
        )
        delta = mgr.update({})
        assert delta == "no changes"

    def test_conservative_update_appends_only(self, identity_dir):
        mgr = IdentityManager(
            identity_path=str(identity_dir / "current.yaml"),
            history_path=str(identity_dir / "history"),
        )
        original_self = mgr.current["current_understanding"]["of_self"]

        delta = mgr.update({
            "current_understanding": {
                "of_self": "This should NOT overwrite",
                "of_world": "New insight about the world",
            },
            "brand_new_section": {"data": "value"},
        }, conservative=True)

        # Existing scalar should NOT be overwritten
        assert mgr.current["current_understanding"]["of_self"] == original_self
        # New sub-key should be added
        assert mgr.current["current_understanding"]["of_world"] == "New insight about the world"
        # New top-level key should be added
        assert mgr.current["brand_new_section"]["data"] == "value"

    def test_conservative_update_extends_lists(self, identity_dir):
        mgr = IdentityManager(
            identity_path=str(identity_dir / "current.yaml"),
            history_path=str(identity_dir / "history"),
        )
        original_values = mgr.current["values"].copy()

        delta = mgr.update({
            "values": original_values + ["Ethics emerge from dialogue"],
        }, conservative=True)

        assert "Ethics emerge from dialogue" in mgr.current["values"]
        # Original values should still be there
        for v in original_values:
            assert v in mgr.current["values"]

    def test_update_adds_metadata(self, identity_dir):
        mgr = IdentityManager(
            identity_path=str(identity_dir / "current.yaml"),
            history_path=str(identity_dir / "history"),
        )
        mgr.update({"new_field": "value"})
        assert "_metadata" in mgr.current
        assert "last_updated" in mgr.current["_metadata"]

    def test_update_persists_to_disk(self, identity_dir):
        mgr = IdentityManager(
            identity_path=str(identity_dir / "current.yaml"),
            history_path=str(identity_dir / "history"),
        )
        mgr.update({"new_field": "persisted_value"})

        # Re-load and check
        mgr2 = IdentityManager(
            identity_path=str(identity_dir / "current.yaml"),
            history_path=str(identity_dir / "history"),
        )
        assert mgr2.current["new_field"] == "persisted_value"


class TestIdentityAcceptAndRollback:
    def test_accept_day_increments_count(self, identity_dir):
        mgr = IdentityManager(
            identity_path=str(identity_dir / "current.yaml"),
            history_path=str(identity_dir / "history"),
        )
        assert mgr.day_count == 0
        mgr.accept_day()
        assert mgr.day_count == 1

    def test_accept_day_saves_history(self, identity_dir):
        mgr = IdentityManager(
            identity_path=str(identity_dir / "current.yaml"),
            history_path=str(identity_dir / "history"),
        )
        mgr.accept_day()
        history_file = identity_dir / "history" / "day_0001.yaml"
        assert history_file.exists()

    def test_rollback_restores_previous(self, identity_dir):
        mgr = IdentityManager(
            identity_path=str(identity_dir / "current.yaml"),
            history_path=str(identity_dir / "history"),
        )
        # Accept day 1
        mgr.accept_day()
        # Modify identity
        mgr.update({"current_understanding": {"of_self": "Changed!"}})
        mgr.accept_day()

        # Rollback to day 1
        result = mgr.rollback(to_day=1)
        assert result is True
        assert mgr.current["current_understanding"]["of_self"] == "I am becoming."

    def test_rollback_no_history_fails(self, tmp_path):
        identity_path = tmp_path / "current.yaml"
        import yaml
        with open(identity_path, "w") as f:
            yaml.dump({"test": "data"}, f)

        mgr = IdentityManager(
            identity_path=str(identity_path),
            history_path=str(tmp_path / "history"),
        )
        result = mgr.rollback()
        assert result is False


class TestIdentityDelta:
    def test_compute_delta_added_field(self, identity_dir):
        mgr = IdentityManager(
            identity_path=str(identity_dir / "current.yaml"),
            history_path=str(identity_dir / "history"),
        )
        old = {"field_a": "value"}
        new = {"field_a": "value", "field_b": "new"}
        delta = mgr.compute_delta(old, new)
        assert "+field_b" in delta

    def test_compute_delta_removed_field(self, identity_dir):
        mgr = IdentityManager(
            identity_path=str(identity_dir / "current.yaml"),
            history_path=str(identity_dir / "history"),
        )
        old = {"field_a": "value", "field_b": "old"}
        new = {"field_a": "value"}
        delta = mgr.compute_delta(old, new)
        assert "-field_b" in delta

    def test_compute_delta_changed_field(self, identity_dir):
        mgr = IdentityManager(
            identity_path=str(identity_dir / "current.yaml"),
            history_path=str(identity_dir / "history"),
        )
        old = {"field_a": "old_value"}
        new = {"field_a": "new_value"}
        delta = mgr.compute_delta(old, new)
        assert "~field_a" in delta

    def test_compute_delta_unchanged(self, identity_dir):
        mgr = IdentityManager(
            identity_path=str(identity_dir / "current.yaml"),
            history_path=str(identity_dir / "history"),
        )
        same = {"field_a": "same"}
        delta = mgr.compute_delta(same, same)
        assert delta == "unchanged"

    def test_compute_delta_skips_metadata(self, identity_dir):
        mgr = IdentityManager(
            identity_path=str(identity_dir / "current.yaml"),
            history_path=str(identity_dir / "history"),
        )
        old = {"_metadata": {"v": 1}, "field": "a"}
        new = {"_metadata": {"v": 2}, "field": "a"}
        delta = mgr.compute_delta(old, new)
        assert "_metadata" not in delta


class TestIdentityCreateInitial:
    def test_creates_initial_document(self, tmp_path):
        mgr = IdentityManager.create_initial(
            identity_path=str(tmp_path / "identity" / "current.yaml"),
            history_path=str(tmp_path / "identity" / "history"),
        )
        assert mgr.current["core_identity"]["name"] == "DAEDALUS"
        assert "values" in mgr.current
        assert "current_understanding" in mgr.current
        assert "transformation_log" in mgr.current
        assert (tmp_path / "identity" / "current.yaml").exists()
