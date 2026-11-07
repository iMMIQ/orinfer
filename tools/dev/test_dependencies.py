"""Release cooling period boundaries and stale lock-file proof rejection."""

from copy import deepcopy
from datetime import UTC, datetime, timedelta
import unittest

from tools.dev.dependencies import MINIMUM_AGE, timestamp, verify


class DependencyAgeTests(unittest.TestCase):
    def setUp(self):
        self.now = datetime(2026, 11, 8, 12, tzinfo=UTC)
        self.files = {"Cargo.lock": "frozen lock identity"}
        self.record = dict(
            schema=1,
            minimum_age_days=7,
            files=self.files.copy(),
            releases={"cargo/indirect@1.0.0": dict(published=(self.now - MINIMUM_AGE).isoformat())},
        )

    def test_exactly_seven_days_passes_and_newer_transitive_artifact_fails(self):
        verify(self.record, self.files, self.now)
        changed = deepcopy(self.record)
        changed["releases"]["uv.lock/new-wheel@1.0.0"] = dict(
            published=(self.now - MINIMUM_AGE + timedelta(microseconds=1)).isoformat()
        )
        with self.assertRaisesRegex(ValueError, "not aged 7 days"):
            verify(changed, self.files, self.now)

    def test_changed_lock_or_manifest_requires_new_publication_proof(self):
        for files in [
            {},
            dict(self.files, **{"Cargo.lock": "new indirect version"}),
            dict(self.files, **{"pyproject.toml": "new build dependency"}),
        ]:
            with self.subTest(files=files), self.assertRaisesRegex(ValueError, "inputs changed"):
                verify(self.record, files, self.now)

    def test_missing_policy_and_timezone_are_rejected(self):
        for field, value in [("schema", 2), ("minimum_age_days", 0), ("releases", {})]:
            invalid = dict(self.record, **{field: value})
            with self.subTest(field=field), self.assertRaises(ValueError):
                verify(invalid, self.files, self.now)
        with self.assertRaisesRegex(ValueError, "timezone"):
            timestamp("2026-11-01T12:00:00")
