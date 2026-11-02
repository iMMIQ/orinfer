import json
from pathlib import Path
import tempfile
import unittest

from tools.bench.dspark_costs import profile


class CostProfileTests(unittest.TestCase):
    def test_profiles_keep_execution_modes_separate_and_use_robust_timings(self):
        identity = {"schema_version": 1, "execution_package": "test", "ranks": 8}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "service.log"
            lines = ["DSPARK_COST_IDENTITY " + json.dumps(identity)]
            for replay, times in ((False, (1.0, 1.1, 9.0)), (True, (0.1,))):
                for seconds in times:
                    lines.append(
                        "DSPARK_COST "
                        + json.dumps(
                            dict(
                                replay=replay,
                                seconds=seconds,
                                setup_s=0.0,
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
            path.write_text("\n".join(lines + ['DSPARK_COST_IDENTITY {"ranks": 4}']))
            with self.assertRaisesRegex(ValueError, "identities"):
                profile([path])


if __name__ == "__main__":
    unittest.main()
