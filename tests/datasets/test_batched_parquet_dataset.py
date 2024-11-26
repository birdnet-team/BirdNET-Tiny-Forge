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

import tempfile
from functools import partial
from pathlib import Path

import pandas as pd
import pytest
from kedro.io import DatasetError

from birdnet_tiny_forge.datasets.batched_parquet_dataset import BatchedParquetDataset


class TestBatchedParquetDataset:
    def test_save(self):
        batch_n_rows = 10

        def make_batch(filler):
            return pd.DataFrame({"val": [filler] * batch_n_rows})

        batches = {str(i): partial(make_batch, i) for i in range(5)}
        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = str(Path(tmp_dir) / "p.parquet")
            ds = BatchedParquetDataset(filepath=file_path)
            ds.save(batches)
            df = pd.read_parquet(file_path)
        assert len(df) == (len(batches) * batch_n_rows)

    def test_save_error(self):
        batch_n_rows = 10

        def make_batch(filler):
            return pd.DataFrame({"val": [filler] * batch_n_rows})

        batches = {str(i): make_batch(i) for i in range(5)}
        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = str(Path(tmp_dir) / "p.parquet")
            ds = BatchedParquetDataset(filepath=file_path)
            with pytest.raises(DatasetError, match="expects .* dict"):
                ds.save(batches)
