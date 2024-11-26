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

"""This pipeline extract features from pre-processed audio data."""

from kedro.pipeline import Pipeline, node, pipeline

from birdnet_tiny_forge.pipelines.feature_extraction.nodes import (
    collate,
    plot_features_sample,
    run_feature_extraction,
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=run_feature_extraction,
                inputs={
                    "audio_slices": "audio_slices",
                    "extractor_name": "params:feature_extraction.extractor",
                    "params": "params:feature_extraction.params",
                },
                outputs="features",
                name="run_feature_extraction",
            ),
            node(
                func=collate,
                inputs={"features": "features", "metadata": "audio_slices_metadata"},
                outputs="model_input",
                name="collate",
            ),
            node(
                func=plot_features_sample,
                inputs={
                    "audio_slices": "audio_slices",
                    "features": "model_input",
                    "sample_n": "params:feature_extraction.plot_n_features",
                },
                outputs="features_plot",
                name="plot_features_sample",
            ),
        ]
    )
