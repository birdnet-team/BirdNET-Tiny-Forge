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

"""Nodes to be used in data preprocessing pipeline"""

from functools import partial
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from birdnet_tiny_forge.datasets.wav_dataset import WavItem

MAX_PLOT_SLICES = 20  # just a sensible max so we don't try to plot thousands


def _extract_loudest_slice_do_one(loader, audio_slice_duration_ms):
    """Find the max of the audio, then return a slice of duration audio_slice_duration_ms centred on the max.
    Also deal with boundary conditions.

    :param loader: callable returning a numpy array of audio (first dimension being audio channel), and its sample rate
    :param audio_slice_duration_ms:
    :return: slice with duration audio_slice_duration_ms
    """
    audio_array, sample_rate = loader()
    slice_n_samples = int(audio_slice_duration_ms / 1000 * sample_rate)
    audio_n_samples = audio_array.shape[1]
    left_edge = slice_n_samples // 2
    right_edge = slice_n_samples - left_edge

    max_index = np.argmax(audio_array)
    start_index = max(max_index - left_edge, 0)
    end_index = min(max_index + right_edge, audio_n_samples)

    # Handle edge cases where the slice would go beyond bounds
    if end_index - start_index < slice_n_samples:
        if start_index == 0:  # Cut at left edge
            end_index = slice_n_samples
        else:  # Cut at right edge
            start_index = audio_n_samples - slice_n_samples
            end_index = audio_n_samples
    return WavItem(
        data=audio_array[:, start_index:end_index],
        sample_rate=sample_rate,
        subtype="FLOAT",
    )


def extract_loudest_slice(audio_clips, audio_slice_duration_ms):
    """This node uses Kedro's lazy loading idiom of passing a dictionary of callables which load the actual data.
    For each callable, it extracts a slice of audio_slice_duration_ms containing the max of the recording.
    It doesn't perform the operation straight away, but it creates a new callable
    (so we stay lazy and the processing is only done one *exactly* when it's time to save the data).

    :param audio_clips: a dictionary of callables returning audio, sample rate.
    :param audio_slice_duration_ms:
    :return: dictionary of callables returning sliced audio
    """
    out_dict = {}
    for path, loader in audio_clips.items():
        out_dict[path] = partial(
            _extract_loudest_slice_do_one,
            loader,
            audio_slice_duration_ms,
        )
    return out_dict


def _slices_make_canonical_do_one(loader, out_sample_rate, desired_subtype):
    """Load data from loader, then make it mono and resample it

    :param loader: callable returning a WavItem object
    :param out_sample_rate: sample rate to which the wav will be resampled
    :param desired_subtype: subtype used in audio saving by soundfile (see its documentation)
    """
    wav_item: WavItem = loader()
    mono = librosa.to_mono(wav_item.data) if wav_item.data.shape[0] != 1 else wav_item.data[0]
    resampled = librosa.resample(mono, orig_sr=wav_item.sample_rate, target_sr=out_sample_rate)
    return WavItem(data=resampled, sample_rate=out_sample_rate, subtype=desired_subtype)


def slices_make_canonical(audio_slices, sample_rate, subtype):
    """Make sure all slices have the same sample rate, number of channels, etc.
    This fn uses the kedro idiom of taking a dictionary of callables and returning a dictionary of callables
    to allow for lazy processing of data.

    :param audio_slices: a dictionary of callables returning sliced audio
    :param sample_rate:
    :param subtype: see soundfile's documentation for details
    :return: dictionary of callables returning canonical-ized sliced audio
    """
    out_dict = {}
    for path, loader in audio_slices.items():
        out_dict[path] = partial(_slices_make_canonical_do_one, loader, sample_rate, subtype)
    return out_dict


def plot_slices_sample(audio_slices, n_slices):
    """Plot a few slices of data as a plotly figure

    :param audio_slices: a dictionary of callables returning sliced audio
    :param n_slices: number of slices to plot
    :return: plotly figure
    """
    assert n_slices < MAX_PLOT_SLICES, f"n_slices must be less than {MAX_PLOT_SLICES}"
    fig = make_subplots(rows=n_slices + 1, cols=1, shared_xaxes=False, vertical_spacing=0.05)
    for i, (path, loader) in enumerate(audio_slices.items()):
        audio_data, sampling_rate = loader()
        fig.add_trace(
            go.Scatter(
                x=list(range(len(audio_data[0]))),
                y=audio_data[0],
                mode="lines",
                name=path,
            ),
            row=i + 1,
            col=1,
        )
        if i >= n_slices:
            break
    fig.update_layout(height=200 * n_slices, showlegend=True, title_text="Audio Waveforms")
    return fig


def extract_metadata(audio_slices):
    """Return a pandas dataframe of metadata for each audio slice. This includes its original path,
    and a label inferred from its path.

    :param audio_slices: a dictionary of callables returning sliced audio
    :return:
    """
    out = []
    for batch_id, slice_loader in audio_slices.items():
        path = Path(batch_id)
        out.extend([{"path": str(path), "label": path.parent.name}])
    return pd.DataFrame(out)


def decide_splits(df, validation_split=0.5):
    """Given dataset metadata (and specifically, each audio slice originating path),
    populate the split information. Part of the test split is further broken off into a validation split,
    the ratio being controlled by the validation_split parameter.

    :param df: metadata dataframe
    :param validation_split: which ratio of the test split to further break off into a validation split
    :return: metadata dataframe containing split info
    """
    df["split"] = df["path"].apply(lambda x: Path(x).parents[1].name)
    test_df = df[df["split"] == "test"]
    val_idx = test_df.sample(frac=validation_split).index
    df.loc[val_idx, "split"] = "validation"
    return df


def plot_splits_info(audio_slices_metadata: pd.DataFrame):
    """Plot split counts broken down by clip class, into a plotly figure

    :param audio_slices_metadata: metadata for the audio slices dataset
    :return: plotly figure
    """
    df = audio_slices_metadata[["split", "label"]].value_counts().reset_index()
    fig = go.Figure(
        data=[
            go.Bar(
                name=label,
                x=df[df["label"] == label]["split"],
                y=df[df["label"] == label]["count"],
            )
            for label in df["label"].unique()
        ]
    )
    fig.update_layout(
        title="Class distribution by split and label",
        xaxis_title="Split",
        yaxis_title="Count",
        barmode="group",
    )
    return fig


def save_labels_dict(audio_slices_metadata: pd.DataFrame):
    """Create dictionary mapping audio labels to sequential integers"""
    labels = sorted(audio_slices_metadata["label"].unique())
    return {lbl: i for i, lbl in enumerate(labels)}
