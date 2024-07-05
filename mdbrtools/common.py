import operator
from types import NoneType
import math
import pandas as pd


class QueryRegionEmptyException(Exception):
    pass


def try_compare(x, op, y, default=False):
    try:
        return op(x, y)
    except TypeError:
        # if the types are not comparable, it's not a match
        # MongoDB calls this type bracketing
        return default


def is_nan(a):
    return a is None or pd.isna(a)


def eq_with_nan(a, b):
    if is_nan(a) and is_nan(b):
        return True
    return operator.eq(a, b)


def ge_with_nan(a, b):
    if eq_with_nan(a, b):
        return True
    return operator.ge(a, b)


def le_with_nan(a, b):
    if eq_with_nan(a, b):
        return True
    return operator.le(a, b)


def type_eq(a, b):
    return isinstance(a, TYPE_CLASSES[b])


def size_eq(a, b):
    if isinstance(a, list):
        return len(a) == b
    return False


TYPE_CLASSES = {
    "double": float,
    "string": str,
    "object": dict,
    "array": list,
    "bool": bool,
    "null": NoneType,
    "int": int,
    "number": (int, float, complex),
}

OPERATORS = {
    "gt": lambda a, val: try_compare(a, operator.gt, val[0]),
    "lt": lambda a, val: try_compare(a, operator.lt, val[0]),
    "gte": lambda a, val: try_compare(a, ge_with_nan, val[0]),
    "lte": lambda a, val: try_compare(a, le_with_nan, val[0]),
    "eq": lambda a, val: try_compare(a, eq_with_nan, val[0]),
    "ne": lambda a, val: try_compare(a, operator.ne, val[0], True),
    "in": lambda a, val: try_compare(val, operator.contains, a),
    "nin": lambda a, val: try_compare(
        val, lambda x, y: operator.not_(operator.contains(x, y)), a, True
    ),
    "type": lambda a, val: try_compare(a, type_eq, val[0]),
    "size": lambda a, val: try_compare(a, size_eq, val[0]),
    "exists": lambda a, val: (a is not None and not math.isnan(a)) and val[0],
}


PYTHON_BSON_TYPE_MAP = {
    "str": "string",
    "int": "int",
    "float": "double",
    "bool": "bool",
    "datetime": "date",
    "ObjectId": "objectId",
    "dict": "object",
    "list": "array",
    "NoneType": "null",
}
