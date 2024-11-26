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

import shutil
import tempfile
from pathlib import Path

import pytest

from birdnet_tiny_forge.datasets.tar_dataset import TarDataset


@pytest.fixture
def test_dir():
    """Fixture that creates a temporary directory with test files"""
    with tempfile.TemporaryDirectory() as source_dir:
        # Create test directory structure
        test_dir = Path(source_dir) / "test_dir"
        test_dir.mkdir()

        # Create test files
        (test_dir / "file1.txt").write_text("content1")
        (test_dir / "file2.txt").write_text("content2")

        # Create nested directory with file
        nested_dir = test_dir / "nested"
        nested_dir.mkdir()
        (nested_dir / "file3.txt").write_text("content3")

        yield test_dir


@pytest.fixture
def tar_path():
    """Fixture that provides a temporary tar file path"""
    with tempfile.NamedTemporaryFile(suffix=".tar.gz") as tmp_tar:
        tar_path = Path(tmp_tar.name)
        yield tar_path


class TestTarredDirectoryDataset:
    def test_tar_dataset_save(self, test_dir, tar_path):
        dataset = TarDataset(filepath=str(tar_path))
        dataset.save(test_dir)

        assert tar_path.exists()
        assert tar_path.stat().st_size > 0

    def test_tar_dataset_load(self, test_dir, tar_path):
        dataset = TarDataset(filepath=str(tar_path))
        dataset.save(test_dir)  # First save a tar

        # Load the directory
        extracted_path = dataset.load()

        try:
            # Verify the contents
            assert (extracted_path / "file1.txt").read_text() == "content1"
            assert (extracted_path / "file2.txt").read_text() == "content2"
            assert (extracted_path / "nested" / "file3.txt").read_text() == "content3"
        finally:
            # Clean up extracted files
            shutil.rmtree(extracted_path)
