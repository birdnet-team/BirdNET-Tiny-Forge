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

"""This pipeline performs pre-processing on audio data,
as well as deciding early which audio data belongs to which train/test/validation split"""

from kedro.pipeline import Pipeline, node, pipeline

from birdnet_tiny_forge.pipelines.data_preprocessing.nodes import (
    decide_splits,
    extract_loudest_slice,
    extract_metadata,
    plot_slices_sample,
    plot_splits_info,
    save_labels_dict,
    slices_make_canonical, slices_filter_short,
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=extract_loudest_slice,
                inputs={
                    "audio_clips": "audio_clips",
                    "audio_slice_duration_ms": "params:data_preprocessing.slice_duration_ms",
                },
                outputs="audio_slices_noncanonical",
                name="extract_loudest_slice",
            ),
            node(
                func=slices_make_canonical,
                inputs={
                    "audio_slices": "audio_slices_noncanonical",
                    "sample_rate": "params:data_preprocessing.slice_sample_rate",
                    "subtype": "params:data_preprocessing.slice_subtype",
                },
                outputs="audio_slices_canonical",
                name="slices_make_canonical",
            ),
            node(
                func=slices_filter_short,
                inputs={
                    "audio_slices": "audio_slices_canonical",
                    "audio_slice_duration_ms": "params:data_preprocessing.slice_duration_ms",
                },
                outputs="audio_slices",
                name="slices_filter_short",
            ),
            node(
                func=plot_slices_sample,
                inputs=["audio_slices", "params:data_preprocessing.plot_n_slices"],
                outputs="audio_slices_plot",
                name="plot_slices_sample",
            ),
            node(
                func=extract_metadata,
                inputs="audio_slices",
                outputs="audio_slices_metadata_no_split",
                name="extract_metadata",
            ),
            node(
                func=decide_splits,
                inputs="audio_slices_metadata_no_split",
                outputs="audio_slices_metadata",
                name="decide_splits",
            ),
            node(
                func=plot_splits_info,
                inputs="audio_slices_metadata",
                outputs="splits_info_plot",
                name="plot_splits_info",
            ),
            node(
                func=save_labels_dict,
                inputs="audio_slices_metadata",
                outputs="labels_dict",
                name="save_labels_dict",
            ),
        ]
    )
