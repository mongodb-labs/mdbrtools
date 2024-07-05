from dataclasses import dataclass, field
from itertools import chain
from typing import Any, Iterable, Union

from tqdm.auto import tqdm


def generate_array_dive_combinations(field_path):
    """helper function for schema.get_prim_values() to dive one level into arrays."""
    # Split the field path into segments
    segments = field_path.split(".")
    num_segments = len(segments)
    combinations = []

    # There are 2 ** num_segments positions to insert ".[]"
    for mask in range(2**num_segments):
        result = []
        for i in range(num_segments):
            result.append(segments[i])
            # Check if the bit is set in the mask to append ".[]"
            if mask & (1 << i):
                result.append("[]")
        # Join the modified segments with dots
        combinations.append(".".join(result))

    return combinations


@dataclass(kw_only=True)
class Node:
    name: str
    path: str
    count: int = 0


@dataclass(kw_only=True)
class FieldContainer:
    fields: dict = field(default_factory=dict)


@dataclass(kw_only=True)
class TypeContainer:
    types: dict = field(default_factory=dict)

    @property
    def is_multi_type(self):
        return len(self.types) > 1

    def __getitem__(self, key: str):
        """overload the [] access method for type containers"""
        return self.types[key]

    def tolist(self):
        return list(t.todict() for t in self.types.values())


@dataclass(kw_only=True)
class Schema(FieldContainer):
    count: int = 0
    flat_fields: dict = field(default_factory=dict)

    def parse(self, docs: Iterable[dict]):
        for doc in docs:
            _parse_doc(doc, schema=self)
            self.count += 1
        _cleanup(node=self, schema=self)

    def __repr__(self):
        return f"Schema[{self.count} docs, {len(self.flat_fields)} fields]"

    def __getitem__(self, path: str):
        """overload the [] access method for schema paths"""
        return self.flat_fields[path]

    @property
    def leaf_fields(self) -> dict:
        """returns a dictionary containing only leaf fields."""
        return {
            path: field for path, field in self.flat_fields.items() if field.is_leaf
        }

    def get_prim_values(self, path: str, dive_into_arrays: bool = True) -> set:
        """returns the primitive values found for the given path. If the path is nested
        and dive_into_arrays is true, this will also return the values contained within
        arrays (of subdocuments) matching this path (but only one level deep), in line
        with MongoDB's query semantics.

            Example:
                documents = [ {a: [{b: 1}, {b: 2}]} ]

                get_prim_values("a.b", dive_into_arrays=False) returns {} because the
                    schema only knows of the path "a.[].b"

                get_prim_values("a.b", dive_into_arrays=True)  returns {1, 2} because
                    it inspects both "a.b" and "a.[].b" as possible paths.

            Example in the mongosh shell:
                test> db.dive.findOne({"a.b": 1})
                {
                _id: ObjectId('661e1cde1e0782fd140bb55d'),
                a: [ { b: [ 1 ] }, { b: [ [ 2 ] ] }, { b: 3 } ]
                }
                test> db.dive.findOne({"a.b": 2})
                null

        """

        result = set()
        paths = generate_array_dive_combinations(path) if dive_into_arrays else [path]

        for p in paths:
            if p in self.flat_fields:
                field = self.flat_fields[p]
                # collect values from all primitive types under this field
                vals = set(
                    chain.from_iterable(
                        t.values
                        for t in field.types.values()
                        if isinstance(t, PrimitiveType)
                    )
                )
                # update result set
                result.update(vals)

        return result

    def __iter__(self):
        for key, fld in self.flat_fields.items():
            yield key, list(fld.tolist())


@dataclass(kw_only=True)
class Field(TypeContainer, Node):
    has_missing: bool = False
    is_leaf: bool = True


@dataclass(kw_only=True)
class Type(Node):
    def todict(self):
        return {"type": self.name, "counter": self.count}


@dataclass(kw_only=True)
class PrimitiveType(Type):
    # initially this is a set, but in cleanup we replace with a sorted list
    values: Union[set, list] = field(default_factory=set)


@dataclass(kw_only=True)
class Array(Type, TypeContainer):
    array_count: int = 0
    is_leaf: bool = True

    @property
    def values(self) -> set:
        """Gathers and returns the set of all its primitive types' values."""
        pvals = set()
        for t in filter(lambda x: isinstance(x, PrimitiveType), self.types.values()):
            pvals |= set(t.values)
        return pvals


@dataclass(kw_only=True)
class Document(Type, FieldContainer):
    pass


def _get_type(val: Any):
    """Returns the type of a value, and the type class."""
    type_str = type(val).__name__

    if type_str == "dict":
        return "document", Document
    elif type_str == "list":
        return "array", Array
    else:
        return type_str, PrimitiveType


def _add_to_types(path: str, value: Any, type_container: TypeContainer):
    """Recursively adds types to the schema tree."""
    type_str, TypeClass = _get_type(value)

    # if the type is not in the container, add an empty one
    if type_str not in type_container.types.keys():
        type_container.types[type_str] = TypeClass(name=type_str, path=path)

    # grab the type from the container
    type_ = type_container.types[type_str]

    # increase counter
    type_.count += 1

    # if primitive type, add value
    if isinstance(type_, PrimitiveType):
        try:
            type_.values.add(value)
        except AttributeError:
            # so we can add values even after _cleanup was called
            type_.values = set(type_.values)
            type_.values.add(value)

    # if array type, add length and process elements
    if isinstance(type_, Array):
        for v in value:
            type_.array_count += 1
            _add_to_types(f"{path}.[]", v, type_)

    # if document type, parse doc recursively
    if isinstance(type_, Document):
        _parse_doc(value, type_, prefix_path=path)


def _add_to_fields(path: str, value: Any, field_container: FieldContainer):
    """Recursively adds fields to the schema tree."""
    name = path.split(".")[-1]
    if name not in field_container.fields.keys():
        # if the field is not in the container, add an empty one
        field_container.fields[name] = Field(name=name, path=path)
    # grab the field from the container
    field = field_container.fields[name]
    # increase counter
    field.count += 1
    # add type to the field
    _add_to_types(path, value, field)


def _cleanup(
    node: Any,
    schema: Schema,
    parent: Any = None,
):
    """Recursively cleans up the schema tree. It sets the `has_missing` flag for fields
    that have lower count than their parent field, and collects all fields in the flat_fields
    dictionary with their paths as keys. It also sorts all values for PrimitiveType and turns
    them back into a list"""

    if isinstance(node, Field):
        schema.flat_fields[node.path] = node
        if parent and parent.count > node.count:
            node.has_missing = True
    if isinstance(node, Array):
        path = f"{node.path}.[]"
        schema.flat_fields[path] = Field(
            name="[]", path=path, types=node.types, count=node.array_count
        )
    if isinstance(node, PrimitiveType):
        node.values = sorted(list(node.values))
    if isinstance(node, TypeContainer):
        for t in node.types.values():
            _cleanup(t, schema, parent)
    if isinstance(node, FieldContainer):
        for f in node.fields.values():
            _cleanup(f, schema, node)

    # mark non-leaf fields as such
    if isinstance(node, Node) and "." in node.path:
        inner_path = node.path.rsplit(".", 1)[0]
        if inner_path in schema.flat_fields:
            schema.flat_fields[inner_path].is_leaf = False


def _parse_doc(doc: dict, schema: Schema, prefix_path=""):
    """Parses a dictionary into a Schema object."""
    for key, value in doc.items():
        path = f"{prefix_path}.{key}".lstrip(".")
        _add_to_fields(path, value, schema)


def parse_schema(docs: Iterable[dict]):
    """Parses a list of dictionaries into a Schema object. This is the main entrypoint to
    parsing a schema."""
    schema = Schema()

    for doc in tqdm(docs, desc="Parsing schema"):
        _parse_doc(doc, schema)
        schema.count += 1

    _cleanup(node=schema, schema=schema)

    return schema
