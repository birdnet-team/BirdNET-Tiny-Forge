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

import numpy as np
import pytest
import soundfile as sf

from birdnet_tiny_forge.datasets.wav_dataset import WavDataset, WavItem

TEST_FILE_PATH = Path(__file__).parent / "test_data" / "CantinaBand3.wav"


class TestWavDataset:
    def test_load(self):
        dataset = WavDataset(filepath=str(TEST_FILE_PATH))
        data, loaded_sr = dataset.load()
        stereo_n_channels = 2
        test_file_sample_rate = 22050
        assert len(data.shape) == stereo_n_channels
        assert loaded_sr == test_file_sample_rate

    @pytest.mark.parametrize("sr", [22000, 1000])
    @pytest.mark.parametrize("subtype", ["PCM_16", "PCM_32"])
    def test_save(self, sr, subtype):
        n_frames = 100
        with tempfile.TemporaryDirectory() as tmp_dir:
            dataset = WavDataset(
                filepath=str(Path(tmp_dir) / "audio.wav"),
            )
            data = np.ones(n_frames)
            item = WavItem(data=data, sample_rate=sr, subtype=subtype)
            dataset.save(item)

            audio = sf.SoundFile(str(Path(tmp_dir) / "audio.wav"))
            assert audio.samplerate == sr
            assert audio.subtype == subtype
            assert audio.frames == n_frames
