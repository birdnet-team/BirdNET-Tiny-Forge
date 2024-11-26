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

"""Simple dataset to load/save keras3 models"""

from copy import deepcopy
from pathlib import PurePosixPath
from typing import Any

import fsspec
import keras
from kedro.io import AbstractDataset
from kedro.io.core import get_filepath_str, get_protocol_and_path


class KerasModelDataset(AbstractDataset[keras.Model, keras.Model]):
    def __init__(
        self,
        filepath: str,
        credentials: dict[str, Any] = None,
        fs_args: dict[str, Any] = None,
    ):
        """Load / save Keras model data for given filepath.

        :param filepath: The location of the keras model to load / save.
        :param credentials: Credentials required to get access to the underlying filesystem.
            E.g. for ``GCSFileSystem`` it should look like `{"token": None}`.
        :param fs_args: Extra arguments to pass into underlying filesystem class.
               E.g. for ``GCSFileSystem`` class: `{"project": "my-project", ...}`.
        """
        protocol, path = get_protocol_and_path(filepath)
        _fs_args = deepcopy(fs_args) or {}
        _credentials = deepcopy(credentials) or {}

        if protocol == "file":
            _fs_args.setdefault("auto_mkdir", True)

        self._protocol = protocol
        self._filepath = PurePosixPath(path)
        self._fs = fsspec.filesystem(self._protocol, **_credentials, **_fs_args)

    def load(self) -> keras.Model:
        """Loads model

        Returns:
            a Keras model
        """
        load_path = get_filepath_str(self._filepath, self._protocol)
        return keras.models.load_model(load_path)

    def save(self, model: keras.Model) -> None:
        save_path = get_filepath_str(self._filepath, self._protocol)
        model.save(save_path)

    def _describe(self) -> dict[str, Any]:
        """Returns a dict that describes the attributes of the dataset."""
        return dict(
            filepath=self._filepath,
            protocol=self._protocol,
        )
