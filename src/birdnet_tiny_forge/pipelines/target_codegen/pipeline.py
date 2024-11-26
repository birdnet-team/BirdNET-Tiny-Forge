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

"""This pipeline is less of a data processing pipeline and more a way
to orchestrate code generation / compiling / flashing.

Note: Kedro is not really the right tool for it, but at the time of writing,
it's good enough and brings consistency with the rest of the codebase.
"""

from kedro.pipeline import Pipeline, node, pipeline

from birdnet_tiny_forge.pipelines.target_codegen.nodes import (
    compile_bin,
    do_code_generation,
    flash_bin,
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=do_code_generation,
                inputs={
                    "build_target": "params:target_codegen.target",
                    "feature_extractor": "params:feature_extraction.extractor",
                    "extractor_params": "params:feature_extraction.params",
                    "model": "model_trained",
                    "reference_dataset": "reference_dataset",
                    "labels_dict": "labels_dict",
                    "all_params": "parameters",
                },
                outputs="codegen_bundle",
                name="do_code_generation",
            ),
            node(
                func=compile_bin,
                inputs={
                    "build_target": "params:target_codegen.target",
                    "src_dir": "codegen_bundle",
                    "toolchain_params": "params:target_codegen.toolchain_params",
                },
                outputs="build_bundle",
                name="compile_bin",
            ),
            node(
                func=flash_bin,
                inputs={
                    "build_target": "params:target_codegen.target",
                    "src_dir": "codegen_bundle",
                    "build_dir": "build_bundle",
                    "toolchain_params": "params:target_codegen.toolchain_params",
                },
                outputs=None,
                name="flash_bin",
            ),
        ]
    )
