import unittest
import math
from mdbrtools.common import OPERATORS, eq_with_nan, ge_with_nan, le_with_nan


class TestOperators(unittest.TestCase):
    def test_operator_gt(self):
        gt = OPERATORS["gt"]
        self.assertTrue(gt(3, [1]))
        self.assertFalse(gt(2, [2]))

        # None type
        self.assertFalse(gt(None, [None]))

        # cross-type is always False
        self.assertFalse(gt(3, [None]))
        self.assertFalse(gt(3, ["a"]))
        self.assertFalse(gt("a", [3]))

    def test_operator_gte(self):
        gte = OPERATORS["gte"]
        self.assertTrue(gte(3, [1]))
        self.assertTrue(gte(2, [2]))
        self.assertFalse(gte(2, [5]))

        self.assertTrue(gte(None, [None]))

        # cross-type is always False
        self.assertFalse(gte(3, [None]))
        self.assertFalse(gte(3, ["a"]))
        self.assertFalse(gte("a", [3]))

    def test_operator_lt(self):
        lt = OPERATORS["lt"]
        self.assertFalse(lt(3, [1]))
        self.assertFalse(lt(2, [2]))
        self.assertTrue(lt(2, [4]))

        # None type
        self.assertFalse(lt(None, [None]))

        # cross-type is always False
        self.assertFalse(lt(3, ["a"]))
        self.assertFalse(lt("a", [3]))

    def test_operator_lte(self):
        lte = OPERATORS["lte"]
        self.assertFalse(lte(3, [1]))
        self.assertTrue(lte(2, [2]))
        self.assertTrue(lte(2, [5]))

        self.assertTrue(lte(None, [None]))

        # cross-type is always False
        self.assertFalse(lte(3, ["a"]))
        self.assertFalse(lte("a", [3]))

    def test_operator_eq(self):
        eq = OPERATORS["eq"]
        self.assertFalse(eq(3, [1]))
        self.assertTrue(eq(2, [2]))

        # None type
        self.assertTrue(eq(None, [None]))

        # cross-type is always False
        self.assertFalse(eq(3, ["a"]))
        self.assertFalse(eq(3, [None]))
        self.assertFalse(eq("a", [3]))

    def test_operator_ne(self):
        ne = OPERATORS["ne"]
        self.assertTrue(ne(3, [1]))
        self.assertFalse(ne(2, [2]))
        self.assertTrue(ne(2, [5]))

        # None type
        self.assertFalse(ne(None, [None]))

        # cross-type is always True
        self.assertTrue(ne(3, ["a"]))
        self.assertTrue(ne(3, [None]))
        self.assertTrue(ne("a", [3]))

    def test_operator_in(self):
        inop = OPERATORS["in"]
        self.assertTrue(inop(3, [1, 2, 3]))
        self.assertFalse(inop(2, [1, 3, 4]))
        self.assertFalse(inop(2, []))

        # None type
        self.assertFalse(inop(3, [None]))
        self.assertTrue(inop(None, [None]))

        # cross-type works as expected
        self.assertFalse(inop(3, ["a", "b", 1, 4]))
        self.assertTrue(inop("a", [1, 4, "a", "b", None]))

    def test_operator_nin(self):
        nin = OPERATORS["nin"]
        self.assertFalse(nin(3, [1, 2, 3]))
        self.assertTrue(nin(2, [1, 3, 4]))
        self.assertTrue(nin(2, []))

        # None type
        self.assertTrue(nin(3, [None]))
        self.assertFalse(nin(None, [None]))

        # cross-type works as expected
        self.assertTrue(nin(3, ["a", "b", 1, 4]))
        self.assertTrue(nin(None, ["a", "b", 1, 4]))
        self.assertFalse(nin("a", [1, 4, "a", "b"]))
        self.assertFalse(nin(None, [1, 4, None, "a", "b"]))

    def test_operator_type(self):
        typeop = OPERATORS["type"]
        self.assertTrue(typeop(1, ["int"]))
        self.assertTrue(typeop("foo", ["string"]))
        self.assertTrue(typeop(False, ["bool"]))
        self.assertTrue(typeop([1, 2, 3], ["array"]))
        self.assertTrue(typeop({"a": 1}, ["object"]))
        self.assertTrue(typeop(None, ["null"]))
        self.assertTrue(typeop(1.23, ["double"]))

        # check for generic number match
        self.assertTrue(typeop(1, ["number"]))
        self.assertTrue(typeop(1.23, ["number"]))

        # cross-type is always False
        self.assertFalse(typeop(1, ["array"]))
        self.assertFalse(typeop("foo", ["object"]))

    def test_operator_size(self):
        sizeop = OPERATORS["size"]
        self.assertTrue(sizeop([1, 2, 3], [3]))
        self.assertTrue(sizeop([1, 2, 3, 4], [4]))
        self.assertFalse(sizeop([1, 2], [4]))

        # cross-type is always False
        self.assertFalse(sizeop(1, [3]))
        self.assertFalse(sizeop("foo", [3]))
        self.assertFalse(sizeop({"a": 1}, [1]))

    def test_eq_with_nan(self):
        self.assertTrue(eq_with_nan(math.nan, math.nan))
        self.assertFalse(eq_with_nan(1, math.nan))
        self.assertTrue(eq_with_nan(1, 1))

    def test_ge_with_nan(self):
        self.assertTrue(ge_with_nan(math.nan, math.nan))
        self.assertFalse(ge_with_nan(1, math.nan))
        self.assertTrue(ge_with_nan(1, 1))
        self.assertFalse(ge_with_nan(1, 2))

    def test_le_with_nan(self):
        self.assertTrue(le_with_nan(math.nan, math.nan))
        self.assertFalse(le_with_nan(1, math.nan))
        self.assertTrue(le_with_nan(1, 1))
        self.assertFalse(le_with_nan(2, 1))
