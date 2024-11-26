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

"""MicroSpeechExtractor uses tflite's microspeech example feature extraction chain"""

import numpy as np
import tensorflow as tf

from birdnet_tiny_forge.features.base import FeatureExtractorBase
from birdnet_tiny_forge.features.microspeech.tflite_micro_frontend import (
    AudioPreprocessor,
    FeatureParams,
)


class MicroSpeechExtractor(FeatureExtractorBase):
    def __init__(self, params):
        self.ap_params = FeatureParams(**params)
        self.win_samples = int((self.ap_params.sample_rate / 1000) * self.ap_params.window_size_ms)
        self.stride_samples = int((self.ap_params.sample_rate / 1000) * self.ap_params.window_stride_ms)
        self.audio_preprocessor = AudioPreprocessor(params=self.ap_params)

    def run(self, sample_rate, audio_slice):
        assert sample_rate == self.ap_params.sample_rate
        self.audio_preprocessor.reset_tflm()
        audio_slice = audio_slice[0]  # it's mono

        # embedded implementation expects audio input as int16
        single_audio = tf.cast(tf.multiply(audio_slice, tf.dtypes.int16.max), tf.int16)
        framed_audio = tf.signal.frame(single_audio, frame_length=self.win_samples, frame_step=self.stride_samples)
        features = []
        for frame in framed_audio:
            frame_feature = self.audio_preprocessor.generate_feature_using_tflm(tf.reshape(frame, (1, -1)))
            features.append(frame_feature.numpy())
        return np.array(features)
