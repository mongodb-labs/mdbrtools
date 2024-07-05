import unittest

from mdbrtools.schema import (
    Field,
    generate_array_dive_combinations,
    parse_schema,
)


class TestParseSchema(unittest.TestCase):
    def test_simple(self):
        schema = parse_schema([{"foo": 1, "bar": 2, "baz": 3}])
        self.assertEqual(schema.count, 1)

        # fields
        self.assertEqual(schema.fields["foo"].count, 1)
        self.assertEqual(schema.fields["bar"].count, 1)
        self.assertEqual(schema.fields["baz"].count, 1)

        # types
        self.assertEqual(list(schema.fields["foo"].types.keys()), ["int"])
        self.assertEqual(list(schema.fields["bar"].types.keys()), ["int"])
        self.assertEqual(list(schema.fields["baz"].types.keys()), ["int"])

        # values
        self.assertEqual(schema.fields["foo"].types["int"].values, [1])
        self.assertEqual(schema.fields["bar"].types["int"].values, [2])
        self.assertEqual(schema.fields["baz"].types["int"].values, [3])

    def test_missing(self):
        schema = parse_schema([{"foo": 1}, {"foo": 2, "bar": 3}])

        self.assertEqual(schema.count, 2)
        self.assertFalse(schema.fields["foo"].has_missing)
        self.assertTrue(schema.fields["bar"].has_missing)

    def test_array(self):
        schema = parse_schema([{"foo": [1, 2, 3]}])

        self.assertEqual(schema.count, 1)
        self.assertEqual(schema.fields["foo"].types["array"].count, 1)
        self.assertEqual(schema.fields["foo"].types["array"].array_count, 3)
        self.assertEqual(
            schema.fields["foo"].types["array"].types["int"].values, [1, 2, 3]
        )

    def test_array_values(self):
        schema = parse_schema(
            [
                {"foo": [1, 2, 3]},
                {"foo": [4, 5, 6]},
            ]
        )

        self.assertEqual(schema.count, 2)
        self.assertEqual(schema.fields["foo"].types["array"].count, 2)
        self.assertEqual(schema.fields["foo"].types["array"].array_count, 6)
        self.assertEqual(schema.fields["foo"].types["array"].values, {1, 2, 3, 4, 5, 6})

    def test_array_values_with_docs(self):
        schema = parse_schema(
            [
                {"foo": [1, 2, 3]},
                {"foo": [{"bar": True}, "test"]},
            ]
        )

        self.assertEqual(schema.count, 2)
        self.assertEqual(schema.fields["foo"].types["array"].count, 2)
        self.assertEqual(schema.fields["foo"].types["array"].array_count, 5)
        self.assertSetEqual(
            set(schema["foo"]["array"].types.keys()),
            {"int", "str", "document"},
        )
        self.assertEqual(schema["foo"]["array"].values, {1, 2, 3, "test"})

    def test_subdocs(self):
        schema = parse_schema(
            [
                {"foo": {"bar": 1}},
                {"foo": {"bar": 2, "baz": 3}},
            ]
        )

        self.assertEqual(schema.count, 2)
        self.assertEqual(schema["foo"].types["document"].count, 2)
        self.assertFalse(schema["foo"].has_missing)
        self.assertEqual(schema["foo.bar"].types["int"].values, [1, 2])
        self.assertFalse(schema["foo.bar"].has_missing)

        self.assertEqual(schema["foo.baz"]["int"].values, [3])
        self.assertTrue(schema["foo.baz"].has_missing)

    def test_flat_fields(self):
        schema = parse_schema(
            [
                {"a": 1, "b": {"b1": 2}, "c": 3},
                {"b": 3, "c": {"c1": {"c12": 4}, "c2": 5}},
            ]
        )
        self.assertSetEqual(
            set(schema.flat_fields.keys()),
            {"a", "b", "b.b1", "c", "c.c1", "c.c1.c12", "c.c2"},
        )

        for field in schema.flat_fields.values():
            self.assertIsInstance(field, Field)

    def test_schema_getitem(self):
        schema = parse_schema(
            [
                {"a": 1, "b": {"b1": 2}, "c": 3},
                {"b": 3, "c": {"c1": {"c12": 4}, "c2": 5}},
            ]
        )
        self.assertEqual(schema["a"], schema.flat_fields["a"])
        self.assertEqual(schema["b"], schema.flat_fields["b"])
        self.assertEqual(schema["b.b1"], schema.flat_fields["b.b1"])
        self.assertEqual(schema["c"], schema.flat_fields["c"])
        self.assertEqual(schema["c.c1"], schema.flat_fields["c.c1"])
        self.assertEqual(schema["c.c1.c12"], schema.flat_fields["c.c1.c12"])
        self.assertEqual(schema["c.c2"], schema.flat_fields["c.c2"])

    def test_typecontainer_getitem(self):
        schema = parse_schema(
            [
                {"a": 1, "b": {"b1": True}, "c": 3},
                {"b": 3, "c": {"c1": {"c12": "test"}, "c2": 5}},
            ]
        )

        self.assertEqual(schema["a"]["int"].values, [1])
        self.assertEqual(schema["c.c1.c12"]["str"].values, ["test"])

    def test_field_is_leaf(self):
        schema = parse_schema(
            [
                {"a": 1, "b": {"b1": True}, "d": [1, 2, 3], "e": [{"e1": 1}, 5]},
                {"a": 3, "b": 5, "c": {"c1": {"c12": "test"}, "c2": 5}},
            ]
        )

        self.assertTrue(schema["a"].is_leaf)
        self.assertTrue(schema["c.c1.c12"].is_leaf)

        self.assertFalse(schema["b"].is_leaf)
        self.assertFalse(schema["c.c1"].is_leaf)
        self.assertFalse(schema["d"].is_leaf)
        self.assertFalse(schema["e"].is_leaf)

    def test_leaf_fields(self):
        schema = parse_schema(
            [
                {"a": 1, "b": {"b1": True}, "d": [1, 2, 3], "e": [{"e1": 1}, 5]},
                {"a": 3, "b": 5, "c": {"c1": {"c12": "test"}, "c2": 5}},
            ]
        )

        leaf_fields = schema.leaf_fields
        self.assertSetEqual(
            set(leaf_fields.keys()),
            {"a", "b.b1", "c.c1.c12", "c.c2", "d.[]", "e.[].e1"},
        )

    def test_schema_to_dict(self):
        schema = parse_schema(
            [
                {"a": 1, "b": {"b1": True}, "d": [1, 2, 3], "e": [{"e1": 1}, 5]},
                {"a": 3, "b": 5, "c": {"c1": {"c12": "test"}, "c2": 5}},
            ]
        )

        self.assertDictEqual(
            dict(schema),
            {
                "a": [{"type": "int", "counter": 2}],
                "b": [
                    {"type": "document", "counter": 1},
                    {"type": "int", "counter": 1},
                ],
                "b.b1": [{"type": "bool", "counter": 1}],
                "d": [{"type": "array", "counter": 1}],
                "d.[]": [{"type": "int", "counter": 3}],
                "e": [{"type": "array", "counter": 1}],
                "e.[]": [
                    {"type": "document", "counter": 1},
                    {"type": "int", "counter": 1},
                ],
                "e.[].e1": [{"type": "int", "counter": 1}],
                "c": [{"type": "document", "counter": 1}],
                "c.c1": [{"type": "document", "counter": 1}],
                "c.c1.c12": [{"type": "str", "counter": 1}],
                "c.c2": [{"type": "int", "counter": 1}],
            },
        )

    def test_field_get_prim_values(self):
        schema = parse_schema(
            [
                {
                    "a": 1,
                    "b": {"b1": True},
                    "d": [1, 2, 3],
                    "e": [{"e1": 1}, 5],
                    "f": [[2]],  # 2 is not a primitive type for f
                },
                {"a": 3, "b": 5, "c": {"c1": {"c12": "test"}}, "d": [4]},
                {"a": "test", "b": True, "c": {}, "e": 7},
            ]
        )

        a = schema.get_prim_values("a")
        self.assertSetEqual(a, {1, 3, "test"})

        b = schema.get_prim_values("b")
        self.assertSetEqual(b, {True, 5})

        c = schema.get_prim_values("c")
        self.assertSetEqual(c, set())

        d = schema.get_prim_values("d")
        self.assertSetEqual(d, {1, 2, 3, 4})

        e = schema.get_prim_values("e")
        self.assertSetEqual(e, {5, 7})

        f = schema.get_prim_values("f")
        self.assertSetEqual(f, set())

        c1 = schema.get_prim_values("c.c1")
        self.assertSetEqual(c1, set())

        c12 = schema.get_prim_values("c.c1.c12")
        self.assertSetEqual(c12, {"test"})

    def test_schema_get_prim_values(self):
        schema = parse_schema(
            [
                {
                    "a": 1,
                    "b": {"b1": True},
                    "d": [1, 2, 3],
                    "e": [{"e1": 1}, 5],
                    "f": [[2]],  # 2 is not a primitive type for f
                },
                {"a": 3, "b": 5, "c": {"c1": {"c12": "test"}}, "d": [4]},
                {"a": "test", "b": True, "c": {}, "e": 7},
            ]
        )

        a = schema.get_prim_values("a")
        self.assertSetEqual(a, {1, 3, "test"})

        b = schema.get_prim_values("b")
        self.assertSetEqual(b, {True, 5})

        c = schema.get_prim_values("c")
        self.assertSetEqual(c, set())

        d = schema.get_prim_values("d")
        self.assertSetEqual(d, {1, 2, 3, 4})

        e = schema.get_prim_values("e")
        self.assertSetEqual(e, {5, 7})

        f = schema.get_prim_values("f")
        self.assertSetEqual(f, set())

        c1 = schema.get_prim_values("c.c1")
        self.assertSetEqual(c1, set())

        c12 = schema.get_prim_values("c.c1.c12")
        self.assertSetEqual(c12, {"test"})

    def test_get_prim_values_array(self):
        schema = parse_schema(
            [
                {"a": [{"b": 1}, {"b": 2}, {"b": 3}]},
            ]
        )

        vals = schema.get_prim_values("a.b", dive_into_arrays=True)
        self.assertSetEqual(vals, {1, 2, 3})

        vals = schema.get_prim_values("a.b", dive_into_arrays=False)
        self.assertSetEqual(vals, set())

    def test_get_prim_values_double_nested(self):
        schema = parse_schema(
            [
                {"a": [{"b": [1]}, {"b": [[2]]}, {"b": 3}]},
            ]
        )

        # dives one level deep to find the 1, but not 2 levels deep to find the 2.
        vals = schema.get_prim_values("a.b", dive_into_arrays=True)
        self.assertSetEqual(vals, {1, 3})

        vals = schema.get_prim_values("a.b", dive_into_arrays=False)
        self.assertSetEqual(vals, set())

    def test_generate_array_dive_combinations(self):
        result = generate_array_dive_combinations("foo.bar.baz")

        expected = [
            "foo.bar.baz",
            "foo.[].bar.baz",
            "foo.bar.[].baz",
            "foo.[].bar.[].baz",
            "foo.bar.baz.[]",
            "foo.[].bar.baz.[]",
            "foo.bar.[].baz.[]",
            "foo.[].bar.[].baz.[]",
        ]

        self.assertListEqual(result, expected)
