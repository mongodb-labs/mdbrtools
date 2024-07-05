import unittest
from copy import deepcopy

from mdbrtools.workload import (
    DEFAULT_OPERATOR_CONFIG,
    MAX_VALUES_FOR_IN_NIN,
    Workload,
)


class TestWorkload(unittest.TestCase):
    def test_workload_assert(self):
        collection = 9

        def should_assert():
            Workload().generate(collection, 10, min_predicates=1, max_predicates=1)

        self.assertRaisesRegex(TypeError, "Invalid collection type", should_assert)

    def test_workload_bool(self):
        collection = [{"a": True}, {"a": False}]
        workload = Workload().generate(
            collection, 10, min_predicates=1, max_predicates=1
        )

        self.assertEqual(len(workload), 10)
        for query in workload:
            self.assertTrue(query.predicates[0].column == "a")
            self.assertIn(
                query.predicates[0].op,
                DEFAULT_OPERATOR_CONFIG["types"]["bool"]["operators"],
            )
            self.assertTrue(query.predicates[0].values[0] in [True, False])

    def test_workload_str(self):
        collection = [{"a": "foo"}, {"a": "bar"}]

        workload = Workload().generate(
            collection, 10, min_predicates=1, max_predicates=1
        )

        self.assertEqual(len(workload), 10)
        for query in workload:
            self.assertTrue(query.predicates[0].column == "a")
            self.assertIn(
                query.predicates[0].op,
                DEFAULT_OPERATOR_CONFIG["types"]["str"]["operators"],
            )
            self.assertTrue(query.predicates[0].values[0] in ["foo", "bar"])

    def test_workload_str_force_eq(self):
        collection = [{"a": "foo"}, {"a": "bar"}]

        # only allow eq on strings
        config = deepcopy(DEFAULT_OPERATOR_CONFIG)
        config["types"]["str"] = {"operators": ["eq"], "weights": [1.0]}

        workload = Workload().generate(
            collection, 10, min_predicates=1, max_predicates=1, operator_config=config
        )

        self.assertEqual(len(workload), 10)
        for query in workload:
            self.assertTrue(query.predicates[0].column == "a")
            self.assertEqual(query.predicates[0].op, "eq")
            self.assertTrue(query.predicates[0].values[0] in ["foo", "bar"])

    def test_workload_num_preds(self):
        collection = [{"a": "foo", "b": 1, "c": False}, {"a": "bar", "b": 2, "c": True}]

        workload = Workload().generate(
            collection, 10, min_predicates=3, max_predicates=3
        )

        self.assertEqual(len(workload), 10)
        for query in workload:
            self.assertEqual(len(query.predicates), 3)

    def test_workload_force_exists(self):
        collection = [{"a": "foo", "b": 1}, {"a": "bar"}]

        # force exists if field has missing values
        config = deepcopy(DEFAULT_OPERATOR_CONFIG)
        config["operators"]["$exists"] = {"enabled": True, "chance": 1.0}

        workload = Workload().generate(
            collection, 10, min_predicates=2, max_predicates=2, operator_config=config
        )

        self.assertEqual(len(workload), 10)
        for query in workload:
            self.assertTrue(len(query.predicates), 2)
            pred_b = next(filter(lambda x: x.column == "b", query.predicates))
            self.assertEqual(pred_b.op, "exists")

    def test_workload_multitype(self):
        collection = [{"a": "foo"}, {"a": True}]

        # force $type if field has multiple values
        config = deepcopy(DEFAULT_OPERATOR_CONFIG)
        config["operators"]["$type"] = {"enabled": True, "chance": 1.0}

        workload = Workload().generate(
            collection, 10, min_predicates=1, max_predicates=1, operator_config=config
        )

        self.assertEqual(len(workload), 10)
        for query in workload:
            self.assertEqual(query.predicates[0].op, "type")
            self.assertIn(query.predicates[0].values[0], ["str", "bool"])

    def test_workload_disable_special_operator(self):
        collection = [{"a": "foo"}, {"a": True}]

        # force $type if field has multiple values
        config = deepcopy(DEFAULT_OPERATOR_CONFIG)
        config["operators"]["$type"] = {"enabled": False, "chance": 1.0}

        workload = Workload().generate(
            collection, 100, min_predicates=1, max_predicates=1, operator_config=config
        )

        self.assertEqual(len(workload), 100)
        for query in workload:
            self.assertNotEqual(query.predicates[0].op, "type")

    def test_workload_size(self):
        collection = [{"a": [1, 2, 3]}, {"a": [1, 2, 3, 4]}]

        # force $type if field has multiple values
        config = deepcopy(DEFAULT_OPERATOR_CONFIG)
        config["operators"]["$size"] = {"enabled": True, "chance": 1.0}

        workload = Workload().generate(
            collection, 10, min_predicates=1, max_predicates=1, operator_config=config
        )

        self.assertEqual(len(workload), 10)
        for query in workload:
            self.assertEqual(query.predicates[0].op, "size")
            self.assertIsInstance(query.predicates[0].values[0], int)

    def test_workload_in_nin_multiple_values(self):
        collection = [{"a": "foo"}, {"a": "bar"}, {"a": "baz"}, {"a": "boo"}]

        # force $in for string types
        config = deepcopy(DEFAULT_OPERATOR_CONFIG)
        config["types"]["str"] = {"operators": ["in"], "weights": [1.0]}

        workload = Workload().generate(collection, 10, operator_config=config)

        self.assertEqual(len(workload), 10)
        for query in workload:
            self.assertTrue(query.predicates[0].column == "a")
            self.assertEqual(query.predicates[0].op, "in")
            self.assertTrue(len(query.predicates[0].values) <= MAX_VALUES_FOR_IN_NIN)

    def test_workload_subdocs(self):
        collection = [{"a": {"b": "foo"}}, {"a": {"b": "bar"}}]

        workload = Workload().generate(
            collection, 10, min_predicates=1, max_predicates=1
        )

        self.assertEqual(len(workload), 10)
        for query in workload:
            self.assertTrue(query.predicates[0].column == "a.b")

    def test_workload_subdocs_with_arrays(self):
        collection = [{"a": {"b": [1, 2, 3]}}, {"a": {"b": [1, 2, 3, 4]}}]

        workload = Workload().generate(
            collection, 10, min_predicates=1, max_predicates=1
        )

        self.assertEqual(len(workload), 10)
        for query in workload:
            self.assertTrue(query.predicates[0].column == "a.b")

    def test_workload_arrays_with_subdocs(self):
        collection = [{"a": [{"b": 1}, {"b": 2}, {"b": 3}]}]

        workload = Workload().generate(
            collection, 10, min_predicates=1, max_predicates=1
        )

        self.assertEqual(len(workload), 10)
        for query in workload:
            self.assertTrue(query.predicates[0].column == "a.b")

    def test_workload_allowed_fields(self):
        collection = [
            {"a": 1, "b": 2, "c": True, "d": {"e": 1, "f": 2, "g": 3}},
        ]

        workload = Workload().generate(
            collection,
            10,
            min_predicates=1,
            max_predicates=1,
            allowed_fields=["d.f", "does_not_exist"],
        )

        self.assertEqual(len(workload), 10)
        for query in workload:
            self.assertTrue(query.predicates[0].column == "d.f")
