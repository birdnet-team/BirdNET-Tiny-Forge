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

"""Base class for Toolchain objects.
These encapsulate toolchain logic, including setup, compiling, flashing and teardown.
"""

from abc import ABC, abstractmethod
from pathlib import Path


class ToolchainBase(ABC):
    @abstractmethod
    def __init__(self, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def setup(self):
        raise NotImplementedError

    @abstractmethod
    def compile(self, code_path: Path, bin_path: Path):
        raise NotImplementedError

    @abstractmethod
    def flash(self, code_path: Path, bin_path: Path):
        raise NotImplementedError

    @abstractmethod
    def teardown(self, **kwargs):
        raise NotImplementedError

    def __enter__(self):
        self.setup()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.teardown()
