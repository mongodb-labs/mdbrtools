import pickle
import random
import time
from copy import deepcopy
from typing import Any, Optional

import numpy as np
from tqdm.auto import tqdm

from .estimator import SampleEstimator

from .common import QueryRegionEmptyException
from .mongodb import MongoCollection
from .query import Predicate, Query
from .schema import parse_schema

MAX_PREDICATES_PER_QUERY = 12
MAX_VALUES_FOR_IN_NIN = 10
MAX_ARRAY_SIZE_FOR_QUERY = 5

ADAPTIVE_FACTOR_POSITIVE = 1.2
ADAPTIVE_FACTOR_NEGATIVE = 0.999

DEFAULT_OPERATOR_CONFIG = {
    "types": {
        # define supported query operators for ints
        "int": {
            "operators": ["lt", "lte", "gt", "gte", "eq"],
            "weights": [1.0, 1.0, 1.0, 1.0, 0.5],
        },
        # define supported query operators for floats
        "float": {
            "operators": ["lt", "lte", "gt", "gte"],
            "weights": [1.0, 1.0, 1.0, 1.0],
        },
        # define supported query operators for strings
        "str": {
            "operators": ["eq", "ne", "in", "nin", "gte", "lte"],
            "weights": [1.0, 0.2, 0.4, 0.2, 0.3, 0.3],
        },
        # define supported query operators for booleans
        "bool": {
            "operators": ["eq", "ne"],
            "weights": [1.0, 0.3],
        },
        "datetime": {
            "operators": ["lt", "lte", "gt", "gte", "eq"],
            "weights": [1.0, 1.0, 1.0, 1.0, 1.0],
        },
    },
    "operators": {
        # query for existence on fields with missing values
        "$exists": {"enabled": True, "chance": 0.3},
        # query for type on fields with multiple types
        "$type": {"enabled": True, "chance": 0.2},
        # query for size on arrays
        "$size": {"enabled": True, "chance": 0.1},
    },
}


class Workload(object):
    """Represents a workload consisting of a number of Query objects."""

    def __init__(self):
        self.queries = []

    @staticmethod
    def _adapt_distribution(
        values: list, weights: list, value: Any, factor: float
    ) -> None:
        if value in values:
            weights[values.index(value)] = np.clip(
                weights[values.index(value)] * factor, 0.00001, 1000
            )

    def generate(
        self,
        collection: MongoCollection | list[dict],
        num_queries: int = 20,
        limit: Optional[int] = 0,
        estimator: Optional[SampleEstimator] = None,
        min_selectivity: float = 0.0,
        max_selectivity: float = 1.0,
        min_predicates: int = 1,
        max_predicates: int = MAX_PREDICATES_PER_QUERY,
        operator_config: dict = DEFAULT_OPERATOR_CONFIG,
        adaptive_config: bool = False,
        max_timeout_secs: int = 60,
        allowed_fields: Optional[list[str]] = None,
    ):
        """Generates a random workload based on a number of parameters:

        collection - a MongoCollection object or a list of dictionaries to be used for
                     schema generation
        limit - only used in conjunction with a MongoCollection to limit the number of
                documents to be parsed. If 0, all documents will be parsed.
        num_queries - the number of queries to generate
        estimator - an Estimator object, used to estimate the selectivity of the queries.
                    Only needed if min_ and max_selectivity are set.
        min_selectivity - the minimum selectivity of the queries
        max_selectivity - the maximum selectivity of the queries
        min_predicates - the minimum number of predicates in the queries
        max_predicates - the maximum number of predicates in the queries
        operator_config - a dictionary containing the supported query operators
        max_timeout_secs - how many seconds before aborting (when choosing very specific
                           selectivities, the sampling process can take a long time.)
        adaptive_config - if True, use adaptive weights for random distributions. This will
                          track which operators and predicate numbers yielded valid queries
                          and increase the probabilities of using these values again. This
                          will speed up the query generation, especially for small selecivity
                          ranges, but will deviate from the provided config probabilities
                          and not use a uniform distribution for num_predicates.
        allowed_fields - a list of allowed fields to use in the queries. If None, all
                         fields in the schema can be used.

        """

        queries = []
        operator_config = deepcopy(operator_config)

        # get num_queries random seed documents from collection
        if isinstance(collection, MongoCollection):
            schema = parse_schema(
                collection.collection.find(
                    {},
                    projection=allowed_fields,
                    limit=limit,
                )
            )
            collection_count = limit or collection.count
            print(f"{collection_count} documents in collection. limit is {limit}")

        elif isinstance(collection, list):
            collection_count = len(collection)
            schema = parse_schema(collection)
        else:
            raise TypeError(
                "Invalid collection type in Workload.generate(), expected a "
                + f"MongoCollection or list of documents, but got {type(collection)}"
            )

        # determine which fields are allowed
        if allowed_fields is None:
            allowed_fields = list(schema.leaf_fields.values())
        else:
            # intersection of allowed fields and schema
            allowed_fields = [
                f
                for f in schema.leaf_fields.values()
                if f.path.rstrip(".[]") in allowed_fields
            ]

        # cap max predicates based on schema and/or allowed_fields
        max_predicates = min(max_predicates, len(allowed_fields))

        # initialise uniform distribution for number of predicates
        num_pred_values = list(range(min_predicates, max_predicates + 1))
        num_pred_weights = [1.0 for _ in num_pred_values]

        # get all supported types from configuration
        supported_types = operator_config["types"].keys()

        for n in tqdm(range(num_queries), desc="Generating workload"):
            # select number of predicates (between min_predicates and max_predicates)
            num_preds = random.choices(num_pred_values, weights=num_pred_weights, k=1)[
                0
            ]

            query_meets_criteria = False
            t = time.time()

            while not query_meets_criteria:
                query = Query()

                # abort if timeout is reached
                if (time.time() - t) >= max_timeout_secs:
                    raise TimeoutError(
                        f"Took more than max. {max_timeout_secs} seconds to generate workload."
                    )

                # select random fields from allowed fields
                fields = random.sample(allowed_fields, num_preds)

                for field in fields:
                    # select a random supported type from the field's types
                    try:
                        type_str, type_ = random.choice(
                            [f for f in field.types.items() if f[0] in supported_types]
                        )
                    except IndexError:
                        continue

                    # select random operator from the field type's operators
                    operator = random.choices(
                        operator_config["types"][type_str]["operators"],
                        weights=operator_config["types"][type_str]["weights"],
                        k=1,
                    )[0]

                    # select a random value from the field's values
                    if operator in ["in", "nin"]:
                        max_values = max(
                            2, min(MAX_VALUES_FOR_IN_NIN, len(type_.values))
                        )

                        values = tuple(
                            random.sample(
                                type_.values,
                                min(
                                    len(type_.values),
                                    np.random.randint(1, max_values),
                                ),
                            )
                        )
                    else:
                        values = (random.choice(type_.values),)

                    # potentially swap for $exists operator
                    opconf_exists = operator_config["operators"]["$exists"]
                    if (
                        schema[field.path].has_missing
                        and opconf_exists["enabled"]
                        and random.random() <= opconf_exists["chance"]
                    ):
                        operator = "exists"
                        values = (random.choice([True, False]),)

                    # potentially swap for $type operator
                    opconf_type = operator_config["operators"]["$type"]
                    if (
                        len(schema[field.path].types) > 1
                        and opconf_type["enabled"]
                        and random.random() <= opconf_type["chance"]
                    ):
                        operator = "type"
                        values = (random.choice(list(field.types.keys())),)

                    # potentially swap for $size operator
                    opconf_size = operator_config["operators"]["$size"]
                    if (
                        field.path.endswith(".[]")
                        and opconf_size["enabled"]
                        and random.random() <= opconf_size["chance"]
                    ):
                        operator = "size"
                        # these lengths are arbitrary but we don't track array lengths in schema
                        values = (random.randint(0, MAX_ARRAY_SIZE_FOR_QUERY),)

                    # remove .[] from field paths
                    path = field.path.replace(".[]", "")

                    query.add_predicate(
                        Predicate(column=path, op=operator, values=values)
                    )

                # estimate selectivity and only include in workload if it lies in the allowed range
                if min_selectivity == 0.0 and max_selectivity == 1.0:
                    # no restrictions, always include
                    query_meets_criteria = True
                else:
                    try:
                        est = estimator.estimate(query) / collection_count
                    except QueryRegionEmptyException:
                        est = 0.0
                    except AttributeError:
                        raise ValueError(
                            "Must provide estimator to generate workload when specifying selectivity."
                        )
                    query_meets_criteria = min_selectivity <= est <= max_selectivity

                # don't allow duplicate queries
                if query in self.queries:
                    query_meets_criteria = False

                if adaptive_config:
                    factor = (
                        ADAPTIVE_FACTOR_POSITIVE
                        if query_meets_criteria
                        else ADAPTIVE_FACTOR_NEGATIVE
                    )
                    self._adapt_distribution(
                        num_pred_values, num_pred_weights, num_preds, factor
                    )
                    if operator not in ["exists", "type", "size"]:
                        self._adapt_distribution(
                            operator_config["types"][type_str]["operators"],
                            operator_config["types"][type_str]["weights"],
                            operator,
                            factor,
                        )

            queries.append(query)

        self.queries = queries
        return self

    def __repr__(self):
        return f"<Workload: {len(self.queries)} queries>"

    def __iter__(self):
        return iter(self.queries)

    def __len__(self):
        return len(self.queries)

    def print(self):
        """Prints each query of the workload."""
        for i, query in enumerate(self.queries):
            print(f"{i:>4} {query}")

    def save(self, filename):
        """Save a workload to a file (without the model)"""
        # save workload
        with open(filename, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, filename):
        """Class method to load a workload from file."""
        # read workload from file
        with open(filename, "rb") as f:
            workload = pickle.load(f)

        return workload
