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

import pytest

from birdnet_tiny_forge.registries.base import RegistryBase


class DummyRegistry(RegistryBase):
    pass


class DummyClass:
    pass


def dummy_fn():
    pass


class AnotherDummyClass:
    pass


def another_dummy_fn():
    pass


class TestRegistryBase:
    @pytest.mark.parametrize("item", [DummyClass, dummy_fn])
    def test_add_and_get_item(self, item):
        DummyRegistry.add("test_item", item)
        assert DummyRegistry.get_item("test_item") is item
        DummyRegistry.remove("test_item")

    @pytest.mark.parametrize("item", [DummyClass, dummy_fn])
    def test_add_and_get_name(self, item):
        DummyRegistry.add("another_test_item", item)
        assert DummyRegistry.get_name(item) == "another_test_item"
        DummyRegistry.remove("another_test_item")

    @pytest.mark.parametrize("item", [DummyClass, dummy_fn])
    def test_contains(self, item):
        with DummyRegistry.transient_registration("test_item", item):
            assert DummyRegistry.contains("test_item")
            assert DummyRegistry.contains(item)
            assert not DummyRegistry.contains("non_existent_item")

    @pytest.mark.parametrize("item", [DummyClass, dummy_fn])
    def test_remove(self, item):
        DummyRegistry.add("test_item", item)
        DummyRegistry.remove("test_item")
        assert not DummyRegistry.contains("test_item")
        assert not DummyRegistry.contains(item)

    @pytest.mark.parametrize(
        "item, duplicate",
        [(DummyClass, AnotherDummyClass), (dummy_fn, another_dummy_fn)],
    )
    def test_duplicate_name_error(self, item, duplicate):
        with DummyRegistry.transient_registration("duplicate_name", item):
            with pytest.raises(ValueError, match="Duplicate name"):
                DummyRegistry.add("duplicate_name", duplicate)

    @pytest.mark.parametrize("item", [DummyClass, dummy_fn])
    def test_duplicate_class_error(self, item):
        with DummyRegistry.transient_registration("unique_name", item):
            with pytest.raises(ValueError, match="Duplicate item"):
                DummyRegistry.add("another_unique_name", item)

    @pytest.mark.parametrize("item", [DummyClass, dummy_fn])
    def test_transient_registration(self, item):
        with DummyRegistry.transient_registration("temp_item", item):
            assert DummyRegistry.contains("temp_item")
            assert DummyRegistry.contains(item)
        assert not DummyRegistry.contains("temp_item")
        assert not DummyRegistry.contains(item)
