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

"""Toolchain for ESP-IDF v5.2, uses esp-idf docker image to compile and flash esp targets"""

import logging
import os
import shutil
import sys
from pathlib import Path

import docker
from tqdm import tqdm

from birdnet_tiny_forge.toolchains.base import ToolchainBase

logger = logging.getLogger("birdnet_tiny_forge")


class ESP_IDF_v5_2(ToolchainBase):
    DOCKER_IMAGE_NAME = "espressif/idf:release-v5.2"

    def __init__(self, **kwargs):
        self.dc = docker.from_env()
        self.port = kwargs.get("port", "/dev/ttyUSB0")

    def setup(self):
        images = self.dc.images.list(filters={"reference": self.DOCKER_IMAGE_NAME})
        if images:
            return  # we already pulled it

        # TODO: this progress tracking is near useless
        #   but multiple tqdm bars make pycharm console go crazy
        for line in tqdm(
            self.dc.api.pull(f"docker.io/{self.DOCKER_IMAGE_NAME}", stream=True, decode=True),
            desc="Pulling image",
            leave=True,
            file=sys.stdout,
        ):
            pass

    def _create_container(self, volumes, command, mount_usb=False):
        return self.dc.containers.run(
            image=self.DOCKER_IMAGE_NAME,
            remove=True,
            volumes=volumes,
            working_dir="/project",
            user=os.getuid(),
            environment={"HOME": "/tmp"},
            command=command,
            devices=[f"{self.port}:/dev/ttyUSB0"] if mount_usb else None,
            detach=True,
        )

    def compile(self, code_path: Path, bin_path: Path):
        container = self._create_container(
            [f"{str(code_path)}:/project"],
            "idf.py build",
        )
        try:
            output = container.attach(stdout=True, stream=True, logs=True)
            for line in output:
                logger.info(line.decode("utf-8"))
            status = container.wait()
            exit_code = status["StatusCode"]
            if exit_code != 0:
                raise ValueError(f"Container exited with exit code {exit_code}")
            # copy both binaries and code, so we can still use idf tools in full
            shutil.copytree(code_path, bin_path, dirs_exist_ok=True, symlinks=True)
        except Exception:
            container.stop()

    def flash(self, code_path: Path, bin_path: Path):
        container = self._create_container(
            volumes=[f"{str(bin_path)}:/project"], command="idf.py flash", mount_usb=True
        )
        try:
            output = container.attach(stdout=True, stream=True, logs=True)
            for line in output:
                logger.info(line.decode("utf-8"))
        except Exception:
            container.stop()

    def teardown(self):
        pass
