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

"""TarDataset saves a directory to a tar.gz file, or loads one by first extracting to a temporary directory,
and returning the path to the temp directory"""

import tarfile
import tempfile
from copy import deepcopy
from pathlib import Path, PurePosixPath
from typing import Any

import fsspec
from kedro.io import AbstractDataset
from kedro.io.core import get_filepath_str, get_protocol_and_path


class TarDataset(AbstractDataset[Path, Path]):
    def __init__(
        self,
        filepath: str,
        temp_extract_path: str = None,
        credentials: dict[str, Any] = None,
        fs_args: dict[str, Any] = None,
    ):
        """When loading tar.gz from a directory, it extracts it to a temp directory and returns its path.
        When saving directory to tar.gz, it compresses its contents, then moves them to filepath.

        :param filepath: The location of the directory to tar/untar.
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
        self._temp_extract_path = PurePosixPath(temp_extract_path) if temp_extract_path is not None else None
        self._fs = fsspec.filesystem(self._protocol, **_credentials, **_fs_args)

    def load(self) -> Path:
        """Extracts tar to temporary directory, returns the directory."""
        load_path = get_filepath_str(self._filepath, self._protocol)
        if self._temp_extract_path is None:
            extract_path = tempfile.mkdtemp()
        else:
            extract_path = get_filepath_str(self._temp_extract_path, self._protocol)
        with tarfile.open(load_path, "r:gz") as tar:
            tar.extractall(extract_path)
        return Path(extract_path)

    def save(self, dir: Path) -> None:
        """Tars contents of directory"""
        save_path = get_filepath_str(self._filepath, self._protocol)
        with tarfile.open(save_path, "w:gz") as tar:
            for item in dir.iterdir():
                tar.add(str(item), arcname=item.name)

    def _describe(self) -> dict[str, Any]:
        """Returns a dict that describes the attributes of the dataset."""
        return dict(
            filepath=self._filepath,
            protocol=self._protocol,
        )
