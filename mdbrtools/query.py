from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable, Optional, Tuple

import numpy as np
import pandas as pd

from enum import Enum
from typing import Union

from .common import OPERATORS, PYTHON_BSON_TYPE_MAP


@dataclass
class Predicate:
    column: str
    op: str
    values: tuple = field(default_factory=tuple)

    def __hash__(self):
        return hash((self.column, self.op, self.values))

    def __eq__(self, other):
        return (
            self.column == other.column
            and self.op == other.op
            and Counter(self.values) == Counter(other.values)
        )


def parse_from_mql(query):
    """converts MQL syntax to a Query object."""
    q = Query()

    if not query:
        return q

    for key, value in query.items():
        if key in ("$or", "$and"):
            left = parse_from_mql(value[0])
            right = (
                parse_from_mql(value[1])
                if (len(value) == 2)
                else parse_from_mql({key: value[1:]})
            )
            p = Operator.AND if key == "$and" else Operator.OR
            q = LogicalExpression(p, left, right)

        elif key.startswith("$"):
            raise AssertionError(f"unsupported operator '{key}'")

        elif isinstance(value, (str, int)):
            q.add_predicate(Predicate(column=key, op="eq", values=(value,)))

        elif isinstance(value, dict):
            ops = value.keys()
            for item in ops:
                if item in ["$in", "$nin"]:
                    q.add_predicate(
                        Predicate(
                            column=key, op=item.lstrip("$"), values=tuple(value[item])
                        )
                    )
                else:
                    q.add_predicate(
                        Predicate(
                            column=key, op=item.lstrip("$"), values=(value[item],)
                        )
                    )

    return q


class Query(object):
    """The Query class represents a query object, consisting of
    a number of Predicates. Predicates are named tuples of
    the form (column, op, values).
    This class also contains some utility methods, for example
    exporting a query to MQL syntax, intersecting a query
    with an index (for Index Selection), etc.
    """

    def __init__(self):
        self.predicates = []

        self._limit = None
        self._projection = None
        self._sort = None

    @property
    def limit(self):
        """returns the limit for this query."""
        return self._limit

    @limit.setter
    def limit(self, n: int):
        """set a limit for this query. None means no limit."""
        assert isinstance(n, int), "limit must be an integer"
        self._limit = n

    @property
    def projection(self):
        """returns the projection tuple of this query."""
        return self._projection

    @projection.setter
    def projection(self, p: Tuple[str, ...]):
        """set the projection tuple for this query."""
        self._projection = p

    @property
    def sort(self):
        """returns the sort tuple of this query."""
        return self._sort

    @sort.setter
    def sort(self, s: Tuple[str, ...]):
        """set the sort tuple for this query."""
        self._sort = s

    def sort_by_selectivity(self, predicate):
        order = {
            "eq": 1,
            "in": 2,
            "gt": 3,
            "lt": 3,
            "gte": 3,
            "lte": 3,
            "size": 4,
            "type": 5,
            "nin": 6,
            "ne": 7,
            "exists": 8,
        }

        return order[predicate.op]

    @property
    def fields(self):
        """return all fields of the query, whether they are part of the predicates,
        projection or sort.
        """
        preds = sorted(self.predicates, key=lambda x: self.sort_by_selectivity(x))
        fields = [pred.column for pred in preds]
        if self.sort:
            fields += list(self.sort)
        if self.projection:
            fields += list(self.projection)
        return fields

    def get_fields(self, exclude_exists_false=False):
        """return all fields of the query, optionally excluding any that have the predicate {"$exists": false}"""
        fields = self.fields
        if exclude_exists_false:
            fields = filter(lambda x: x not in self.exists_false_fields, fields)
        return fields

    @property
    def exists_false_fields(self):
        """return all fields of the query on which the predicate is $exists: False"""
        fields = [
            pred.column
            for pred in self.predicates
            if pred.op == "exists" and not pred.values[0]
        ]
        return sorted(fields)

    def add_predicate(self, predicate: Optional[Predicate]):
        """adds a predicate to the query."""
        if predicate is None:
            # skip empty predicates, useful for nested columns with proxy inner columns
            # that would otherwise be chosen too often
            return

        assert isinstance(predicate.values, tuple)
        assert (
            predicate.op in OPERATORS.keys()
        ), f"unsupported operator '${predicate.op}'"

        # simplify $in
        if predicate.op in ["in", "nin"]:
            assert len(predicate.values) >= 1, "in operator requires at least 1 value"
            if len(predicate.values) == 1:
                predicate.op == "eq" if predicate.op == "in" else "neq"
        else:
            assert len(predicate.values) == 1, "operator takes exactly 1 value"

        self.predicates.append(predicate)

    def add_predicates(self, preds: Iterable[Predicate]) -> None:
        """add multiple predicates to a query at once."""
        for pred in preds:
            self.add_predicate(pred)

    def index_intersect(self, index):
        """returns a copy of this query that only contains predicates on the
        provided index fields, left to right, up until an index field was
        not included in the query.

        Example: query is {a: true, b: {$lt: 20}, c: 5}
                 query.index_intersect(("b", "d", "c")) return the query
                 equivalent to {b: {$lt: 20}}, because d is not present
                 in the query and aborts the algorithm.

        """
        query = Query()

        for fld in index:
            # reset field indicator
            query_has_field = False
            for pred in self.predicates:
                if pred.column == fld:
                    query_has_field = True
                    query.add_predicate(pred)
            # if no predicates found for this field, stop.
            if not query_has_field:
                break
        return query

    def is_subset(self, index):
        """returns true if all predicate fields are included in the index.
        This is necessary, but not sufficient to be a covered by the index.
        It is used to determine if a limit caps the cost of the query or not.
        """
        predicate_fields = set(p.column for p in self.predicates)
        return predicate_fields.issubset(set(index))

    def is_dnf(self):
        return True

    def is_covered(self, index):
        """returns true if this query is covered by the index, false otherwise.
        A query without a projection is never covered. A query with projection
        is covered if the union of all predicate fields and projected fields
        appear in the index.
        """
        if self.projection is None:
            return False

        predicate_fields = tuple(p.column for p in self.predicates)
        fields_to_cover = set(predicate_fields + self.projection)

        return fields_to_cover.issubset(set(index))

    def can_use_sort(self, index):
        """returns true if this query has a sort and the index can be used to sort,
        false otherwise.
        A query can be sorted by an index if any of the following is true:
            a) the sort fields are the same as the index fields (incl. order)
            b) the sort fields are a prefix sequence of the index fields
            c) the sort fields are a sub-sequence of the index fields and
               the query only uses equality predicates on all fields preceeding
               the sort fields.

               Example: index on ('a', 'b', 'c', 'd')
                        query {a: 5, b: {$gt: 6}} with sort ('b', 'c')
                        The predicate preceeding the sort fields is 'a' and is
                        an equality predicate. The index can be used to sort.
        """
        if self.sort is None:
            # TODO check this: if the query doesn't need a sort, the sort fields sequence
            # is and empty list, and theoretically a prefix of any index.
            return True

        # cover case a) and b)
        if self.sort == index[: len(self.sort)]:
            return True

        # cover case c)
        sub_idx = -1
        for i in range(len(index) - len(self.sort)):
            if self.sort == index[i : i + len(self.sort)]:
                # sort fields are a sub-sequence of index fields
                sub_idx = i

        if sub_idx != -1:
            # check for preceeding equality predicates
            if all(pred.op == "eq" for pred in self.predicates[:sub_idx]):
                return True

        return False

    def __repr__(self):
        s = []
        s.append(f"filter={self.to_mql()}")
        if self.sort:
            s.append(f"sort={self.sort}")
        if self.limit:
            s.append(f"limit={self.limit}")
        if self.projection:
            s.append(f"projection={self.projection}")
        return f"Query({', '.join(s)})"

    def __len__(self):
        """The length of a query, using 'len(query)', is the number of
        predicates it contains."""
        return len(self.predicates)

    def __eq__(self, other):
        is_query = isinstance(other, self.__class__)
        if not is_query:
            return False

        return (
            Counter(self.predicates) == Counter(other.predicates)
            and self.sort == other.sort
            and self.limit == other.limit
            and self.projection == other.projection
        )

    def to_df_query(self, df):
        """Converts the query to be used on a Pandas DataFrame."""
        FUNC_MAP = {"lt": "lt", "lte": "le", "gt": "gt", "gte": "ge", "eq": "eq"}
        bools = pd.Series([True] * df.shape[0])
        for pred in self.predicates:
            fn = FUNC_MAP[pred.op]
            colname = pred.column.replace("_", " ")
            newbools = getattr(df[colname], fn)(pred.values[0])
            bools &= newbools
        return bools.sum()

    def _convert_dnf(self):
        return self

    def to_mql(self):
        """converts the query to MQL syntax."""
        query = {}

        def clean_value(v):
            if isinstance(v, str):
                return v.strip()
            elif isinstance(v, np.int64):
                return v.item()
            else:
                return v

        for predicate in self.predicates:
            name = predicate.column
            # remove leading/trailing whitespace
            values = [clean_value(v) for v in predicate.values]
            if predicate.op == "eq":
                if pd.isna(values[0]):
                    # special handling for missing values
                    query[name] = {"$exists": False}
                else:
                    query[name] = values[0]
            elif predicate.op in ["in", "nin"]:
                query[name] = {f"${predicate.op}": values}
            elif predicate.op == "type":
                query[name] = {"$type": PYTHON_BSON_TYPE_MAP.get(values[0], values[0])}
            else:
                if name not in query:
                    query[name] = {f"${predicate.op}": values[0]}
                else:
                    if isinstance(query[name], dict):
                        query[name][f"${predicate.op}"] = values[0]
                    else:
                        query[name] = {
                            "$eq": query[name],
                            f"${predicate.op}": values[0],
                        }
        return query


class Operator(Enum):
    AND = 1
    OR = 2
    NOT = 3


class LogicalExpression(object):
    """
    This class represents an expression in mql containing $and / $or.
    It builds a binary tree structure with left and right being other LogicalExpression
    objects, or Query objects (leaves).

    The to_dnf method converts any arbitary nested tree and $and and $or and turns it
    into it's disjunctive normal form using the following rules:

        Double negation elimination: NOT (NOT A) = A
        De Morgan's laws:
            NOT (A AND B) = NOT A OR NOT B
            NOT (A OR B) = NOT A AND NOT B
        The distributive law: A AND (B OR C) = (A AND B) OR (A AND C)"""

    def __init__(self, op, left, right):
        self.op = op
        self.left = left
        self.right = right
        self.flat = False

    @staticmethod
    def combine_and(
        left: Union[Query, "LogicalExpression"],
        right: Union[Query, "LogicalExpression"],
    ) -> Union[Query, "LogicalExpression"]:
        if isinstance(left, Query) and isinstance(right, Query):
            # if we see an AND, flatten it
            new_query = Query()
            new_query.add_predicates(preds=[*left.predicates, *right.predicates])
        else:
            new_query = LogicalExpression(Operator.AND, left, right)

        return new_query

    def remove_not(self):
        # recursively replace all instances of not with their required values
        # eq <-> neq
        # lt <-> gte
        # lte <-> gt
        # in <-> nin
        # TODO: not implemented in this ticket
        pass

    def is_dnf(self):
        """
        The _convert_dnf function is called multiple times because of the case where
        both sides of a $and are $or LogicalExpressions, the case (A || B) && (C || D).
        Using the distributive law it becomes ([A || B] && C) || ([A || B] && D) after the first pass
        it needs a second pass through to completely flatten it out"""

        # Check if the current tree is in DNF
        if self.flat:
            return True

        if self.op == Operator.AND:
            return False

        l = self.left.is_dnf()  # noqa: E741
        r = self.right.is_dnf()

        self.flat = l and r
        return l and r

    def to_dnf(self):
        dnf = self._convert_dnf()
        while not dnf.is_dnf():
            dnf = dnf._convert_dnf()
        return dnf

    def _convert_dnf(self):
        # If the tree is a leaf node, return it directly
        if isinstance(self, Query):
            return self

        # Recursively convert the left and right subtrees to DNF
        left = self.left._convert_dnf() if self.left else None
        right = self.right._convert_dnf() if self.right else None

        # Convert the current node to DNF, depending on the value at the node
        if self.op == Operator.NOT:
            # NOT (NOT A) = A
            if right and right.op == Operator.NOT:
                return right.right
            # NOT (A AND B) = NOT A OR NOT B
            elif right and right.op == Operator.AND:
                not_left = LogicalExpression(Operator.NOT, right.left)
                not_right = LogicalExpression(Operator.NOT, right.right)
                return LogicalExpression(Operator.OR, not_left, not_right)
            # NOT (A OR B) = NOT A AND NOT B
            elif right and right.op == Operator.OR:
                not_left = LogicalExpression(Operator.NOT, right.left)
                not_right = LogicalExpression(Operator.NOT, right.right)
                return LogicalExpression(Operator.AND, not_left, not_right)
            else:
                return self

        elif self.op == Operator.AND:
            # A AND (B OR C) = (A AND B) OR (A AND C)
            if isinstance(right, LogicalExpression) and right.op == Operator.OR:
                left_and_right = LogicalExpression.combine_and(left, right.right)
                left_and_left = LogicalExpression.combine_and(left, right.left)

                return LogicalExpression(Operator.OR, left_and_left, left_and_right)

            elif isinstance(left, LogicalExpression) and left.op == Operator.OR:
                right_and_left = LogicalExpression.combine_and(right, left.left)
                right_and_right = LogicalExpression.combine_and(right, left.right)
                return LogicalExpression(Operator.OR, right_and_left, right_and_right)

            else:
                # left and right are both not logical expressions, so just return a query with implicit and.
                return LogicalExpression.combine_and(left, right)

        # If the node is already in DNF, return it
        else:
            self.right = right
            self.left = left
            return self

    def __repr__(self):
        # If the tree is a leaf node, return the value at the node
        if not self.left and not self.right:
            return str(self)

        # Print the value at the current node, followed by the representations of the left and right subtrees
        return f"{self.op.name}({self.left.__repr__()}, {self.right.__repr__()})"

    def to_mql(self):
        """converts the query to MQL syntax."""
        query = {}

        if self.op == Operator.AND:
            name = "$and"
        elif self.op == Operator.OR:
            name = "$or"
        else:
            raise AssertionError(f"unsupported operator '{self.op}'")

        expr1 = self.left.to_mql()
        expr2 = self.right.to_mql()
        query[name] = [expr1, expr2]

        return query


if __name__ == "__main__":
    # Example Usage of a Query objet
    query = Query()

    query.add_predicates(
        [
            Predicate("Suspension_Indicator", "eq", ("Y",)),
            Predicate("Make", "in", ("INFIN", "HYUND")),
        ]
    )

    query2 = Query()

    query2.add_predicates([Predicate("County", "eq", ("NASSAU",))])

    query3 = Query()

    query3.add_predicates([Predicate("City", "eq", ("NEW YORK",))])

    # Example Usage of a LogicalExpression objet
    expre = LogicalExpression(
        Operator.AND,
        query,
        LogicalExpression(
            Operator.OR,
            query2,
            query3,
        ),
    )
    print(expre)
    print("\ndnf:")
    print(expre.to_dnf())
