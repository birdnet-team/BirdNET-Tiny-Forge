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

"""Dataset to save/load wav data"""

import dataclasses
from copy import deepcopy
from pathlib import PurePosixPath
from typing import Any

import fsspec
import numpy as np
import soundfile as sf
from kedro.io import AbstractDataset
from kedro.io.core import get_filepath_str, get_protocol_and_path
from numpy import ndarray


@dataclasses.dataclass
class WavItem:
    data: ndarray
    sample_rate: int
    subtype: str | None  # See soundfile documentation on subtype


class AudioDataset(AbstractDataset[ndarray, ndarray]):
    def __init__(
        self,
        filepath: str,
        credentials: dict[str, Any] = None,
        fs_args: dict[str, Any] = None,
    ):
        """Load / save wav data for given filepath.

        :param filepath: The location of the audio file to load / save.
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

    def load(self) -> tuple[ndarray, int | float]:
        """Loads data from the audio file as [-1, 1] float32.

        Returns:
            a numpy array containing float32 data for the loaded audio file
        """
        load_path = get_filepath_str(self._filepath, self._protocol)
        with sf.SoundFile(load_path, mode="rb") as f:
            data = f.read(dtype=np.float32, always_2d=True)
            return data.T, f.samplerate

    def save(self, out: WavItem) -> None:
        save_path = get_filepath_str(self._filepath.with_suffix(".wav"), self._protocol)
        with self._fs.open(save_path, mode="wb") as f:
            sf.write(
                f,
                out.data,
                samplerate=out.sample_rate,
                subtype=out.subtype,
            )

    def _describe(self) -> dict[str, Any]:
        """Returns a dict that describes the attributes of the dataset."""
        return dict(
            filepath=self._filepath,
            protocol=self._protocol,
        )
