import unittest
from tools.bench.acceptance import validate, check


class AcceptanceTests(unittest.TestCase):
    def spec(self):
        return dict(
            cases=[
                dict(
                    id="color",
                    request=dict(seed=20261002, temperature=0),
                    expected_words=["red", "blue"],
                )
            ],
            rounds=[dict(id="arrival", jobs=[dict(case="color", delay_s=0.5)])],
        )

    def test_label_checks_preserve_order_and_reject_partial_answers(self):
        case = self.spec()["cases"][0]
        self.assertTrue(check(dict(content="Red, BLUE."), case)["task_passed"])
        for content in ["blue red", "red", "red blue green", ""]:
            with self.assertRaises(ValueError):
                check(dict(content=content), case)

    def test_rejects_bad_reproducibility_arrivals_and_membership(self):
        self.assertEqual(list(validate(self.spec())), ["color"])
        spec = self.spec()
        spec["cases"][0]["request"]["seed"] = 0
        with self.assertRaises(ValueError):
            validate(spec)
        for job in [
            dict(case="missing"),
            dict(case="color", delay_s=-1),
            dict(case="color", delay_s=61),
        ]:
            spec = self.spec()
            spec["rounds"][0]["jobs"] = [job]
            with self.assertRaises(ValueError):
                validate(spec)
        spec = self.spec()
        spec["rounds"][0]["jobs"] *= 129
        with self.assertRaises(ValueError):
            validate(spec)


if __name__ == "__main__":
    unittest.main()
