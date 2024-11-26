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

"""Nodes to be used in feature extraction pipeline"""

from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from birdnet_tiny_forge.features.base import FeatureExtractorBase
from birdnet_tiny_forge.registries import FeatureExtractorsRegistry

BatchedData = dict[Path, Callable[[], pd.DataFrame]]  # batch id -> data loader

MAX_SAMPLE_PLOTS = 20


def _run_feature_extraction_do_one(
    extractor: FeatureExtractorBase, slice_loader: Callable[[], tuple[np.ndarray, int]]
) -> np.ndarray:
    """Run feature extraction on slice

    :param extractor: feature extractor
    :param slice_loader: callable returning audio slice and its sampling rate
    :return: features
    """
    audio_slice, sr = slice_loader()
    return extractor.run(sr, audio_slice)


def run_feature_extraction(
    audio_slices: dict[str, Callable[[], tuple[np.ndarray, int]]], extractor_name: str, params: dict[str, Any]
) -> BatchedData:
    """Run feature extraction on audio slices. Extraction is done lazily, no data is returned here
    but rather a dictionary of callables, where each callable performs the actual feature extraction when called.

    :param audio_slices: a dictionary of callables returning audio slices
    :param extractor_name: which feature extractor to use
    :param params: parameters to be passed to the feature extractor
    :return: dictionary of callables returning features for each input audio slice
    """
    extractor_cls = FeatureExtractorsRegistry.get_item(extractor_name)
    extractor = extractor_cls(params)
    features = {}
    for batch_id, slice_loader in audio_slices.items():
        features[batch_id] = partial(_run_feature_extraction_do_one, extractor, slice_loader)
    return features


def _collate_one(feat_batch_loader, md):
    """Collate feature data and metadata"""
    data = feat_batch_loader()
    df_data = {
        "features": pd.Series([data.flatten()], dtype=object),
        "features_shape": pd.Series([data.shape], dtype=object),
        **md,
    }
    return pd.DataFrame(df_data)


def collate(features, metadata) -> BatchedData:
    """Collate features and metadata in a single dataframe. Collation is done lazily, no data is returned here,
     but rather a dictionary of callables which return the data when called.

    :param features: dictionary of callables mapping audio slice originating path to callable loading it
    :param metadata: pandas dataframe of metadata for each input audio slice
    :return out: dictionary of callables returning collated data
    """
    assert list(features.keys()) == list(metadata["path"])
    out = {}
    for (batch_id, feat_batch), (_, metadata_batch) in zip(features.items(), metadata.iterrows()):
        out[batch_id] = partial(_collate_one, feat_batch, metadata_batch)
    return out


def plot_features_sample(audio_slices, features: pd.DataFrame, sample_n):
    """Plot a few samples of features

    :param audio_slices: a dictionary of callables returning audio slices
    :param features: dictionary of callables returning features
    :param sample_n: number of slices features to plot
    """
    assert sample_n < MAX_SAMPLE_PLOTS, f"sample_n must be less than {MAX_SAMPLE_PLOTS}"
    fsample = features.sample(n=sample_n, random_state=42)
    fmin = features["features"].apply(lambda x: x.min()).min()
    fmax = features["features"].apply(lambda x: x.max()).max()

    fig = make_subplots(
        rows=sample_n + 1,
        cols=2,
        vertical_spacing=0.05,
        subplot_titles=[x for x in list(fsample["path"]) for _ in range(2)],
    )
    for i, (idx, row) in enumerate(fsample.iterrows()):
        path = row["path"]
        audio_data, sample_rate = audio_slices[path]()
        fig.add_trace(
            go.Heatmap(
                z=np.array(row["features"]).reshape(row["features_shape"]).T,
                zmin=fmin,
                zmax=fmax,
                name=path,
                showscale=False,
            ),
            row=i + 1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=list(range(audio_data.shape[1])),
                y=audio_data[0],
                mode="lines",
                name=path,
                showlegend=False,
            ),
            row=i + 1,
            col=2,
        )
        if i >= sample_n:
            break
    fig.update_layout(height=500 * sample_n, title_text="Features")
    return fig
