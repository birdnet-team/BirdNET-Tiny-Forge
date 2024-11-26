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
from pathlib import Path

import nox

WHEEL_NAME = "tflite_micro-0.dev20241031040012-py3-none-any.whl"


@nox.session(python=["3.10", "3.11"])
def tests(session):
    session.run("ls", "tflite-micro", external=True)
    with tempfile.TemporaryDirectory() as temp_dir:
        tmp_tflite_micro_path = Path(temp_dir) / "tflite-micro"
        wheel_path = tmp_tflite_micro_path / "bazel-bin" / "python" / "tflite_micro" / "whl_dist" / WHEEL_NAME
        session.run("./build_tflite.sh", tmp_tflite_micro_path, external=True)
        session.run("pip", "install", wheel_path)

        install_script_path = Path(temp_dir) / "poetry_install.py"
        session.run("curl", "-sSL", "-o", install_script_path, "https://install.python-poetry.org", external=True)
        session.run("python3", install_script_path, "--preview")
        # can't dynamically change path of tflite micro in pyproject.toml so rather installed it via pip,
        # and excluded it from installation with poetry
        session.run("poetry", "install", "--without", "tflite-micro", external=True)
    session.run("pytest", "tests/")
