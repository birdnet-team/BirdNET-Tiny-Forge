#   Copyright 2024 BirdNET-Team
#   Copyright 2024 fold ecosystemics
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

"""A few components of the pipeline are dynamically applied by looking at configuration. For example, one might
configure feature extraction to happen via FeatureExtractorX or FeatureExtractorY,
model training with ModelA, or ModelB, and so on.
For this reason, we keep a few registries around which can map from strings (used in the configuration yaml file)
to the concrete objects to be used in the pipeline. A pipeline node can then retrieve the needed object using
the string via the corresponding registry.
Here is a base class that can map classes and functions to names (and viceversa as needed).
Registers are singletons. Here we implement the singleton pattern by making each Register a class (rather than
a class instance).
As such, a metaclass RegisterMeta is used to make sure each Register class gets its own internal dicts
to perform the str->entity and entity->str mapping.
"""

from collections.abc import Callable
from contextlib import contextmanager
from inspect import isclass, isfunction
from typing import Generic, TypeVar

_D = TypeVar("_D")


class RegisterMeta(type):
    """Metaclass for RegistryBase, 'initializes' new empty internal dicts for each new register class."""

    def __new__(mcs, name, bases, namespace):
        namespace["_name_to_item"] = {}
        namespace["_item_to_name"] = {}
        return super().__new__(mcs, name, bases, namespace)

    def __repr__(cls):
        mapping = [f"\t{name}: {item}\n" for name, item in cls._name_to_item.items()]
        return f"<{cls.__name__}:\n{''.join(mapping)}>"


class RegistryBase(Generic[_D], metaclass=RegisterMeta):
    """Base class for all registries. Can map strings to classes/functions and vice-versa.
    a transient_registration context manager is also implemented to facilitate testing.
    """

    _name_to_item: dict[str, type[_D] | Callable[..., _D]] = {}
    _item_to_name: dict[type[_D] | Callable[..., _D], str] = {}

    @classmethod
    def add(cls, name: str, item: _D | type[_D] | Callable[..., _D]) -> None:
        """Register item with name"""
        # Handle both classes and functions
        registered_item = item if isclass(item) or isfunction(item) else item.__class__

        if registered_item in cls._item_to_name:
            raise ValueError(f"Duplicate item in {cls.__name__} register: {registered_item}")
        if name in cls._name_to_item:
            raise ValueError(f"Duplicate name in {cls.__name__} register: {name}")

        cls._name_to_item[name] = registered_item
        cls._item_to_name[registered_item] = name

    @classmethod
    def get_name(cls, item: _D | type[_D] | Callable[..., _D]) -> str:
        """Get name of item"""
        registered_item = item if isclass(item) or isfunction(item) else item.__class__
        return cls._item_to_name[registered_item]

    @classmethod
    def get_item(cls, name: str) -> type[_D] | Callable[..., _D]:
        """Get item by name"""
        return cls._name_to_item[name]

    @classmethod
    def contains(cls, item: str | _D | type[_D] | Callable[..., _D]) -> bool:
        """Return True if item is in registry, else False"""
        if isinstance(item, str):
            return item in cls._name_to_item
        if isclass(item) or isfunction(item):
            return item in cls._item_to_name
        return item.__class__ in cls._item_to_name

    @classmethod
    def remove(cls, item: str | _D | type[_D] | Callable[..., _D]) -> None:
        """Remove item from registry"""
        if isinstance(item, str):
            registered_item = cls._name_to_item.pop(item)
            cls._item_to_name.pop(registered_item)
            return

        registered_item = item if isclass(item) or isfunction(item) else item.__class__
        name = cls._item_to_name.pop(registered_item)
        cls._name_to_item.pop(name)

    @classmethod
    @contextmanager
    def transient_registration(cls, name: str, item: _D | type[_D] | Callable[..., _D]):
        """Utility context manager that temporarily registers an item under name,
        then undoes the registration at exit."""
        try:
            yield cls.add(name, item)
        finally:
            cls.remove(name)
