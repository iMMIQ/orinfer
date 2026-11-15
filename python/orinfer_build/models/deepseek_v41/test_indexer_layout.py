from copy import deepcopy

import unittest

from orinfer_build.models.deepseek_v41.indexer_layout import INDEXER, PREPARE, group_queries


def fixture():
    columns = [dict(buffer=f"input{i}", row_bytes=None) for i in range(9)]
    step = dict(
        op="kernel",
        manifest="full/kernel.json",
        owner_work=dict(op="constant", value=64),
        buffers=["work", "pointers", "active"],
    )
    rank = dict(
        buffers=[dict(name="pointers", bytes=32 * 10 * 8)],
        packets=[dict(body=[step, dict(step, manifest="bounded/kernel.json")])],
    )
    model = dict(
        batch=dict(
            owners=32,
            private=[],
            columns=columns,
            pointers={"pointers": list(range(9))},
            partitioned_tables={"work": 8, "pointers": 8},
            decoder_replay=dict(tables=["work", "pointers"]),
        ),
        ranks=[deepcopy(rank), deepcopy(rank)],
    )
    full = dict(
        op=INDEXER,
        parameters=dict(rows=256, length=131072, tile_rows=8, _owner_token_capacity=256),
        operands=[dict(shape=[8192, 2]), dict(shape=[32, 10]), dict(shape=[1])],
    )
    bounded = deepcopy(full)
    bounded["parameters"].update(candidate_blocks=2048, tile_rows=1)
    return model, {"full/kernel.json": full, "bounded/kernel.json": bounded}


class QueryLayoutTests(unittest.TestCase):
    def test_grouping_keeps_owner_views_and_decoder_replay(self):
        model, descriptions = fixture()
        bounded = deepcopy(descriptions["bounded/kernel.json"])
        updates, report = group_queries(model, descriptions)
        contract = model["batch"]
        assert report == dict(changed=2, additional_owner_bytes=1081344, query_group=4)
        assert descriptions["bounded/kernel.json"] == bounded
        for rank in model["ranks"]:
            prep, grouped, original_bounded = rank["packets"][0]["body"]
            assert original_bounded["manifest"] == "bounded/kernel.json"
            assert prep["buffers"][0] == grouped["buffers"][0] == "work"
            prep_table = prep["buffers"][1]
            grouped_table = grouped["buffers"][1]
            assert prep_table in contract["decoder_replay"]["tables"]
            assert grouped_table in contract["decoder_replay"]["tables"]
            assert contract["partitioned_tables"][prep_table] == 8
            assert contract["partitioned_tables"][grouped_table] == 8
            inputs = contract["pointers"][prep_table]
            consumers = contract["pointers"][grouped_table]
            assert inputs[:3] == [0, 1, 4]
            assert consumers == [inputs[3], inputs[4], 2, 3, inputs[5], 5, 6, 7, 8]
            for index in inputs[3:]:
                column = contract["columns"][index]
                assert column["buffer"] in contract["private"]
                assert column["row_bytes"] is None
                assert sum(b["name"] == column["buffer"] for b in rank["buffers"]) == 1
            prototype = updates[prep["manifest"]]
            assert prototype["op"] == PREPARE
            assert prototype["operands"][1]["shape"] == [32, 7]
        with self.assertRaisesRegex(ValueError, "capacity"):
            group_queries(model, descriptions)

    def test_grouping_rejects_incompatible_tiles_without_mutation(self):
        for query_group, tile_rows in [(3, 8), (4, 2)]:
            with self.subTest(query_group=query_group, tile_rows=tile_rows):
                model, descriptions = fixture()
                descriptions["full/kernel.json"]["parameters"]["tile_rows"] = tile_rows
                original = deepcopy(model)
                with self.assertRaises(ValueError):
                    group_queries(model, descriptions, query_group)
                self.assertEqual(model, original)
