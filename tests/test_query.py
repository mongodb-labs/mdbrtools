import unittest

from mdbrtools.query import (
    LogicalExpression,
    Operator,
    Predicate,
    Query,
    parse_from_mql,
)


class TestLogicalExpression(unittest.TestCase):
    def test_from_mql(self):
        query = Query()

        query.add_predicates([Predicate("Suspension_Indicator", "eq", ("Y",))])

        query2 = Query()

        query2.add_predicates([Predicate("County", "eq", ("NASSAU",))])

        query3 = Query()

        query3.add_predicates([Predicate("City", "eq", ("NEW YORK",))])

        q = parse_from_mql(
            {
                "$and": [
                    {"Suspension_Indicator": "Y"},
                    {"$or": [{"County": "NASSAU"}, {"City": "NEW YORK"}]},
                ]
            }
        )

        expre = LogicalExpression(
            Operator.AND,
            query,
            LogicalExpression(
                Operator.OR,
                query2,
                query3,
            ),
        )

        self.assertEqual(q.op, Operator.AND)
        self.assertEqual(str(q.left), str(query))
        self.assertTrue(isinstance(q.right, LogicalExpression))
        self.assertEqual(str(q), str(expre))

    def test_to_mql(self):
        query = Query()

        query.add_predicates([Predicate("Suspension_Indicator", "eq", ("Y",))])

        query2 = Query()

        query2.add_predicates([Predicate("County", "eq", ("NASSAU",))])

        query3 = Query()

        query3.add_predicates([Predicate("City", "eq", ("NEW YORK",))])

        expre = LogicalExpression(
            Operator.AND,
            query,
            LogicalExpression(
                Operator.OR,
                query2,
                query3,
            ),
        )

        q = expre.to_mql()

        self.assertEqual(
            q,
            {
                "$and": [
                    {"Suspension_Indicator": "Y"},
                    {"$or": [{"County": "NASSAU"}, {"City": "NEW YORK"}]},
                ]
            },
        )

    def test_parse_non_binary(self):
        q = parse_from_mql(
            {
                "$or": [
                    {"County": "NASSAU"},
                    {"City": "NEW YORK"},
                    {"Suspension_Indicator": "Y"},
                ]
            }
        )

        query = Query()

        query.add_predicates([Predicate("Suspension_Indicator", "eq", ("Y",))])

        query2 = Query()

        query2.add_predicates([Predicate("County", "eq", ("NASSAU",))])

        query3 = Query()

        query3.add_predicates([Predicate("City", "eq", ("NEW YORK",))])

        self.assertEqual(str(q.left), str(query2))
        self.assertTrue(isinstance(q.right, LogicalExpression))
        self.assertEqual(q.op, Operator.OR)
        self.assertEqual(q.right.op, Operator.OR)
        self.assertEqual(str(q.right.left), str(query3))
        self.assertEqual(str(q.right.right), str(query))

    def test_to_dnf(self):
        expre = parse_from_mql(
            {
                "$and": [
                    {"Suspension_Indicator": "Y"},
                    {"$or": [{"County": "NASSAU"}, {"City": "NEW YORK"}]},
                ]
            }
        )

        self.assertFalse(expre.is_dnf())

        dnf = expre.to_dnf()

        self.assertEqual(
            str(dnf),
            "OR(Query(filter={'Suspension_Indicator': 'Y', 'County': 'NASSAU'}), Query(filter={'Suspension_Indicator': 'Y', 'City': 'NEW YORK'}))",
        )

        self.assertEqual(
            dnf.to_mql(),
            {
                "$or": [
                    {"Suspension_Indicator": "Y", "County": "NASSAU"},
                    {"Suspension_Indicator": "Y", "City": "NEW YORK"},
                ]
            },
        )

        self.assertTrue(dnf.is_dnf())

    def test_to_dnf_only_and(self):
        query = Query()

        query.add_predicates([Predicate("Suspension_Indicator", "eq", ("Y",))])

        query2 = Query()

        query2.add_predicates([Predicate("County", "eq", ("NASSAU",))])

        expre = LogicalExpression(Operator.AND, query, query2)

        self.assertTrue(isinstance(expre, LogicalExpression))

        dnf = expre.to_dnf()

        self.assertTrue(isinstance(dnf, Query))

        self.assertEqual(
            dnf.to_mql(), {"Suspension_Indicator": "Y", "County": "NASSAU"}
        )

    def test_to_dnf_nested_on_both_sides(self):
        query = Query()

        query.add_predicates(
            [
                Predicate("Suspension_Indicator", "eq", ("Y",)),
            ]
        )

        query2 = Query()

        query2.add_predicates([Predicate("County", "eq", ("NASSAU",))])

        query3 = Query()

        query3.add_predicates([Predicate("City", "eq", ("NEW YORK",))])

        query4 = Query()

        query4.add_predicates([Predicate("Make", "in", ("INFIN", "HYUND"))])

        expre = LogicalExpression(
            Operator.AND,
            LogicalExpression(
                Operator.OR,
                query,
                query4,
            ),
            LogicalExpression(
                Operator.OR,
                query2,
                query3,
            ),
        )

        q = parse_from_mql(
            {
                "$and": [
                    {
                        "$or": [
                            {"Suspension_Indicator": "Y"},
                            {"Make": {"$in": ["INFIN", "HYUND"]}},
                        ]
                    },
                    {"$or": [{"County": "NASSAU"}, {"City": "NEW YORK"}]},
                ]
            }
        )

        self.assertEqual(str(q), str(expre))

        dnf = q.to_dnf()

        self.assertEqual(
            dnf.to_mql(),
            {
                "$or": [
                    {
                        "$or": [
                            {"County": "NASSAU", "Suspension_Indicator": "Y"},
                            {"County": "NASSAU", "Make": {"$in": ["INFIN", "HYUND"]}},
                        ]
                    },
                    {
                        "$or": [
                            {"City": "NEW YORK", "Suspension_Indicator": "Y"},
                            {"City": "NEW YORK", "Make": {"$in": ["INFIN", "HYUND"]}},
                        ]
                    },
                ]
            },
        )

    def test_query_equality(self):
        query = Query()
        query.add_predicates([Predicate("Suspension_Indicator", "eq", ("Y",))])

        query2 = Query()
        query2.add_predicates([Predicate("Suspension_Indicator", "eq", ("Y",))])

        self.assertEqual(query, query2)

    def test_query_equality_multi_values(self):
        query = Query()
        query.add_predicate(Predicate("Year", "in", (2020, 2024)))

        query2 = Query()
        query2.add_predicate(Predicate("Year", "in", (2020, 2024)))

        self.assertEqual(query, query2)

    def test_query_equality_multi_predicates(self):
        query = Query()
        query.add_predicate(Predicate("Year", "in", (2024, 2020)))
        query.add_predicate(Predicate("Title", "eq", ("The Matrix",)))

        query2 = Query()
        query2.add_predicate(Predicate("Year", "in", (2024, 2020)))
        query2.add_predicate(Predicate("Title", "eq", ("The Matrix",)))

        self.assertEqual(query, query2)

    def test_query_to_mql_types(self):
        query = Query()
        query.add_predicate(Predicate("year", "type", ("int",)))
        self.assertDictEqual(query.to_mql(), {"year": {"$type": "int"}})

        query = Query()
        query.add_predicate(Predicate("name", "type", ("str",)))
        self.assertDictEqual(query.to_mql(), {"name": {"$type": "string"}})

        query = Query()
        query.add_predicate(Predicate("_id", "type", ("ObjectId",)))
        self.assertDictEqual(query.to_mql(), {"_id": {"$type": "objectId"}})

        query = Query()
        query.add_predicate(Predicate("numbers", "type", ("list",)))
        self.assertDictEqual(query.to_mql(), {"numbers": {"$type": "array"}})

        query = Query()
        query.add_predicate(Predicate("birthday", "type", ("datetime",)))
        self.assertDictEqual(query.to_mql(), {"birthday": {"$type": "date"}})

        query = Query()
        query.add_predicate(Predicate("foo", "type", ("non-existing",)))
        self.assertDictEqual(query.to_mql(), {"foo": {"$type": "non-existing"}})
