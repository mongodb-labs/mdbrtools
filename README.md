# mdbrtools

This package contains experimental tools for schema analysis and query workload generation used by MongoDB Research (MDBR).

## Disclaimer

This tool is not officially supported or endorsed by MongoDB Inc. The code is released for use "AS IS" without any warranties of any kind, including, but not limited to its installation, use, or performance. Do not run this tool in critical production systems.

## Installation

#### Installation with pip

This tool requires python 3.x and pip on your system. To install `mdbrtools`, run the following command:

```bash
pip install mdbrtools
```

#### Installation from source

Clone the respository from github. From the top-level directory, run:

```
pip install -e .
```

This installs an _editable_ development version of `mdbrtools` in your current Python environment.

## Usage

See the `./notebooks` directory for more detailed examples for schema parsing and workload generation.

### Schema Parsing

Schema parsing operates on a list of Python dictionaries.

```python
from mdbrtools.schema import parse_schema
from pprint import pprint

docs = [
    {"_id": 1, "mixed_field": "world", "missing_field": False},
    {"_id": 2, "mixed_field": 123},
    {"_id": 3, "mixed_field": False, "missing_field": True},
]

schema = parse_schema(docs)
pprint(dict(schema))
```

Converting the schema object to a dictionary will output some general information about the schema:

```
{'_id': [{'counter': 3, 'type': 'int'}],
 'missing_field': [{'counter': 2, 'type': 'bool'}],
 'mixed_field': [{'counter': 1, 'type': 'str'},
                 {'counter': 1, 'type': 'int'},
                 {'counter': 1, 'type': 'bool'}]}
```

For access to types, values and uniqueness information, see the examples in [`./notebooks/schema_parsing.ipynb`](./notebooks/schema_parsing.ipynb).

## Workload Generation

Workload generation takes either a list of Python dictionaries, or a `MongoCollection` object as input.

```python
from mdbrtools.workload import Workload

docs = [
    {"_id": 1, "mixed_field": "world", "missing_field": False},
    {"_id": 2, "mixed_field": 123},
    {"_id": 3, "mixed_field": False, "missing_field": True},
]

workload = Workload()
workload.generate(docs, num_queries=5)

for query in workload:
    print(query.to_mql())
```

The generated MQL queries are:

```python
{'missing_field': True}
{'missing_field': {'$exists': False}, '_id': {'$gte': 3}}
{'_id': {'$gt': 3}, 'mixed_field': False, 'missing_field': {'$exists': False}}
{'mixed_field': {'$gte': 'world'}, '_id': 3, 'missing_field': {'$ne': False}}
{'mixed_field': 'world'}
```

The workload generator supports a number of different constraints on the queries:

- min. and max. number of predicates per query
- allowing only certain fields
- which query operators are allowed for which data types
- control over the weights by which operators are randomly chosen
- min. and max. query selectivity constraints

See the notebook under [`./notebooks/workload_generation.ipynb`](./notebooks/workload_generation.ipynb) for examples.

## Tests

To execute the unit tests, run from the top-level directory:

```
python -m unittest discover ./tests
```

## License

MIT, see [LICENSE](./LICENSE).
