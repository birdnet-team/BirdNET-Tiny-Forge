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

from pathlib import Path

import pandas as pd
import pytest
import soundfile

from birdnet_tiny_forge.datasets.wav_dataset import WavItem
from birdnet_tiny_forge.pipelines.data_preprocessing.nodes import (
    extract_loudest_slice, extract_metadata, decide_splits, save_labels_dict
)

TEST_AUDIO = Path(__file__).parents[2] / 'datasets' / 'test_data' / 'CantinaBand3.wav'


@pytest.fixture
def dummy_audio_data():
    def load_data():
        data, sr = soundfile.read(str(TEST_AUDIO), always_2d=True, dtype='float32')
        return data.T, sr
    mock_data = {
        "train/class1/file.wav": load_data,
        "train/class2/file.wav": load_data,
        "test/class1/file.wav": load_data,
        "test/class2/file.wav": load_data,
    }
    return mock_data


class TestDataPreprocessingPipeline:

    def test_extract_loudest_slice(self, dummy_audio_data):
        data, sample_rate = dummy_audio_data['train/class1/file.wav']()
        max_val: float = data.max()
        audio_slice_duration_ms = 100
        slices = extract_loudest_slice(dummy_audio_data, audio_slice_duration_ms=audio_slice_duration_ms)
        audio_slice = slices["train/class1/file.wav"]()
        assert isinstance(audio_slice, WavItem)
        assert (audio_slice.data == max_val).any(), "slice doesn't contain max"
        assert (audio_slice.data.shape[1] / audio_slice.sample_rate) == (audio_slice_duration_ms / 1000), "wrong duration"

    def test_extract_metadata(self, dummy_audio_data):
        metadata_df = extract_metadata(dummy_audio_data)
        assert list(sorted(metadata_df["label"].unique())) == ["class1", "class2"]

    def test_decide_splits(self, dummy_audio_data):
        df = pd.DataFrame({"path": [x for x in dummy_audio_data.keys()]})
        df_with_splits = decide_splits(df, validation_split=0.5)
        assert set(sorted(df_with_splits["split"].unique())) == {"train", "test", "validation"}
        split_count = df_with_splits["split"].value_counts()
        assert split_count["test"] == split_count["validation"], "validation split at 0.5 not respected"

    def test_save_labels_dict(self, dummy_audio_data):
        df = pd.DataFrame({"label": ["C", "B", "A"]})
        labels_dict = save_labels_dict(df)
        assert labels_dict == {"A": 0, "B": 1, "C": 2}
