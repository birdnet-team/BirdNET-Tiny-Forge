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

"""Nodes to be used in target code generation pipeline"""

import tempfile
from pathlib import Path

from birdnet_tiny_forge.registries import (
    BuildTargetsRegistry,
    FeatureExtractorsRegistry,
)
from birdnet_tiny_forge.targets.base import BuildTargetBase
from birdnet_tiny_forge.toolchains.base import ToolchainBase


def do_code_generation(
    build_target,
    feature_extractor,
    extractor_params,
    model,
    reference_dataset,
    labels_dict,
    all_params,
):
    bt_cls: type[BuildTargetBase] = BuildTargetsRegistry.get_item(build_target)
    extractor = FeatureExtractorsRegistry.get_item(feature_extractor)(extractor_params)
    tmpdir = Path(tempfile.mkdtemp()) / "bundle"
    tmpdir.mkdir(exist_ok=True)
    bt = bt_cls(
        feature_extractor=extractor,
        model=model,
        reference_dataset=reference_dataset,
        labels_dict=labels_dict,
        params=all_params,
    )
    bt.validate()
    bt.process_target_templates(tmpdir)
    return tmpdir


def compile_bin(build_target, src_dir, toolchain_params: dict):
    bt_cls: type[BuildTargetBase] = BuildTargetsRegistry.get_item(build_target)
    toolchain_cls: type[ToolchainBase] = bt_cls.get_toolchain()
    tmpdir = Path(tempfile.mkdtemp()) / "build"
    tmpdir.mkdir(exist_ok=True)
    with toolchain_cls(**toolchain_params) as tc:
        tc.compile(Path(src_dir), tmpdir)
    if len(list(tmpdir.iterdir())) == 0:
        raise RuntimeError(f"The build target {build_target} did not compile")
    return tmpdir


def flash_bin(build_target, src_dir, build_dir, toolchain_params: dict):
    bt_cls: type[BuildTargetBase] = BuildTargetsRegistry.get_item(build_target)
    toolchain_cls: type[ToolchainBase] = bt_cls.get_toolchain()
    with toolchain_cls(**toolchain_params) as tc:
        tc.flash(src_dir, build_dir)
