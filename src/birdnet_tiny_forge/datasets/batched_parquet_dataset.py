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

"""A save-only kedro dataset that, given a dict of callable loaders,
iterates through them appending to a parquet file.
Loading one batch at a time can help cap RAM usage for large datasets
when lazy loading techniques are used.
"""

from collections.abc import Callable
from io import BytesIO

import pyarrow as pa
from kedro.io.core import DatasetError, get_filepath_str
from kedro_datasets.pandas.parquet_dataset import ParquetDataset
from pyarrow import parquet as pq
from tqdm import tqdm


class BatchedParquetDataset(ParquetDataset):
    def _save(self, batch_loaders: dict[str, Callable]) -> None:
        """
        :param batch_loaders: a dict mapping strings to callables, each returning a pandas dataframe with same schema.
        """
        save_path = get_filepath_str(self._get_save_path(), self._protocol)
        if not all(callable(b) for b in batch_loaders.values()):
            raise DatasetError(
                f"{self.__class__.__name__} expects to receive a dict mapping strings to callable objects,"
                f"each returning a pandas dataframe"
            )
        writer = None
        bytes_buffer = BytesIO()
        with self._fs.open(save_path, mode="wb") as fs_file:
            for batch_loader in tqdm(batch_loaders.values()):
                batch = batch_loader()
                table = pa.Table.from_pandas(df=batch)
                if writer is None:
                    writer = pq.ParquetWriter(bytes_buffer, table.schema, use_dictionary=False)
                writer.write_table(table)
                fs_file.write(bytes_buffer.getvalue())
                bytes_buffer.seek(0)
                bytes_buffer.truncate()
            if writer is not None:
                writer.close()
                fs_file.write(bytes_buffer.getvalue())
        self._invalidate_cache()
