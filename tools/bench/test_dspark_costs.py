import json
from pathlib import Path
import tempfile
import unittest

from tools.bench.dspark_costs import profile


class CostProfileTests(unittest.TestCase):
    def test_profiles_keep_execution_modes_separate_and_use_robust_timings(self):
        identity = {"schema_version": 2, "execution_package": "test", "ranks": 8}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "service.log"
            lines = ["DSPARK_COST_IDENTITY " + json.dumps(identity)]
            for replay, times in ((False, (1.0, 1.1, 9.0)), (True, (0.1,) * 32)):
                for index, seconds in enumerate(times):
                    lines.append(
                        "DSPARK_COST "
                        + json.dumps(
                            dict(
                                replay=replay,
                                seconds=seconds,
                                setup_s=0.3 if replay and index == 0 else 0.0,
                                work=dict(rows=16, prefill_rows=0, context=2048),
                            )
                        )
                    )
            path.write_text("\n".join(lines))
            result = profile([path])
            self.assertEqual(result["identity"], identity)
            self.assertEqual(
                {p["replay"]: p["seconds"] for p in result["points"]},
                {False: 1.1, True: 0.1},
            )
            self.assertEqual(next(p["setup_s"] for p in result["points"] if p["replay"]), 0.3)
            path.write_text(
                "\n".join(lines + ['DSPARK_COST_IDENTITY {"schema_version":2,"ranks":4}'])
            )
            with self.assertRaisesRegex(ValueError, "identities"):
                profile([path])


if __name__ == "__main__":
    unittest.main()
