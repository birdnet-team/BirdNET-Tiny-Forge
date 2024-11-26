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

"""Build target for korvo2_bird_logger template project.
The class will check that the operations from the input keras model are supported by tflite-micro.
It then converts the keras model to tflite, and prepares all the information needed by the project template
to generate the final code.
"""

import re
from pathlib import Path

import numpy as np
import tensorflow as tf
from keras import Model
from tensorflow.data import Dataset
from tensorflow.lite.tools import visualize as tflite_vis

from birdnet_tiny_forge.features.base import FeatureExtractorBase
from birdnet_tiny_forge.features.microspeech import MicroSpeechExtractor
from birdnet_tiny_forge.features.microspeech.tflite_micro_frontend import (
    AudioPreprocessor as MSAudioPreprocessor,
)
from birdnet_tiny_forge.targets.base import BuildTargetBase
from birdnet_tiny_forge.toolchains.esp_idf_v5_2 import ESP_IDF_v5_2


def tflite_to_byte_array(tflite_file: Path):
    with tflite_file.open("rb") as input_file:
        buffer = input_file.read()
    return buffer


def parse_op_str(op_str):
    """Converts a flatbuffer operator string to a format suitable for Micro
    Mutable Op Resolver. Example: CONV_2D --> AddConv2D.

    This fn is adapted from tensorflow lite micro tools scripts:
    (https://github.com/tensorflow/tflite-micro/tree/main/tensorflow/lite/micro/tools/gen_micro_mutable_op_resolver/generate_micro_mutable_op_resolver_from_model.py)
    """
    # Edge case for AddDetectionPostprocess().
    # The custom code is TFLite_Detection_PostProcess.
    op_str = op_str.replace("TFLite", "")
    word_split = re.split("[_-]", op_str)
    formatted_op_str = ""
    for part in word_split:
        if len(part) > 1:
            if part[0].isalpha():
                formatted_op_str += part[0].upper() + part[1:].lower()
            else:
                formatted_op_str += part.upper()
        else:
            formatted_op_str += part.upper()
    # Edge cases
    formatted_op_str = formatted_op_str.replace("Lstm", "LSTM")
    formatted_op_str = formatted_op_str.replace("BatchMatmul", "BatchMatMul")
    return formatted_op_str


def get_model_ops_and_acts(model_buf):
    """Extracts a set of operators from a tflite model.

    This fn is adapted from tensorflow lite micro tools scripts:
    (https://github.com/tensorflow/tflite-micro/tree/main/tensorflow/lite/micro/tools/gen_micro_mutable_op_resolver/generate_micro_mutable_op_resolver_from_model.py)
    """
    custom_op_found = False
    operators_and_activations = set()
    data = tflite_vis.CreateDictFromFlatbuffer(model_buf)
    for op_code in data["operator_codes"]:
        if op_code["custom_code"] is None:
            op_code["builtin_code"] = max(op_code["builtin_code"], op_code["deprecated_builtin_code"])
        else:
            custom_op_found = True
            operators_and_activations.add(tflite_vis.NameListToString(op_code["custom_code"]))
    for op_code in data["operator_codes"]:
        # Custom operator already added.
        if custom_op_found and tflite_vis.BuiltinCodeToName(op_code["builtin_code"]) == "CUSTOM":
            continue
        operators_and_activations.add(tflite_vis.BuiltinCodeToName(op_code["builtin_code"]))  # will be None if unknown
    return operators_and_activations


class Korvo2BirdLogger(BuildTargetBase):
    def __init__(
        self,
        feature_extractor: FeatureExtractorBase,
        model: Model,
        reference_dataset: Dataset,
        labels_dict: dict,
        params: dict,
    ):
        self._model_buf = self.get_model_buf(model, reference_dataset)
        self._model_ops = get_model_ops_and_acts(self._model_buf)
        super().__init__(feature_extractor, model, reference_dataset, labels_dict, params)

    @staticmethod
    def get_model_buf(model: Model, reference_dataset: Dataset):
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.inference_input_type = tf.dtypes.int8
        converter.inference_output_type = tf.dtypes.int8
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter._experimental_disable_per_channel_quantization_for_dense_layers = True

        def representative_dataset_gen():
            for example_spectrograms, example_spect_labels in reference_dataset.take(10):
                for X, _ in zip(example_spectrograms, example_spect_labels):
                    # Add a `batch` dimension, so that the spectrogram can be used
                    yield [X[tf.newaxis, ...]]

        converter.representative_dataset = representative_dataset_gen
        model_buf = converter.convert()
        return model_buf

    def do_check_model_compatible(self):
        if None in self._model_ops:
            raise ValueError(
                "Model contains op(s) that can't be converted to tflite micro. "
                f"Known ops: {self._model_ops.difference({None})}"
            )

    def do_check_feature_extractor_compatible(self):
        if not isinstance(self.feature_extractor, MicroSpeechExtractor):
            raise ValueError("Unknown feature extractor type")

    def extract_context(self) -> dict:
        if isinstance(self.feature_extractor, MicroSpeechExtractor):
            self.feature_extractor.ap_params.use_float_output = False  # preprocessor running on embedded will use int8
            fxtr = MSAudioPreprocessor(self.feature_extractor.ap_params)
            fxtr_path = fxtr.generate_tflite_file()
            extractor_buf = tflite_to_byte_array(fxtr_path)
            extractor_hex_vals = [hex(b) for b in extractor_buf]
        else:
            raise NotImplementedError

        model_hex_vals = [hex(b) for b in self._model_buf]

        op_list = [parse_op_str(op) for op in sorted(list(self._model_ops))]

        sr = self.feature_extractor.ap_params.sample_rate
        slice_duration_ms = self.params["data_preprocessing"]["slice_duration_ms"]
        n_windows = (
            int(sr / 1000 * slice_duration_ms) - self.feature_extractor.win_samples
        ) // self.feature_extractor.stride_samples + 1
        labels = sorted(self.labels_dict, key=self.labels_dict.get)
        tensor_arena_size = self.params["target_codegen"].get("params", {}).get("tensor_arena_size", 60000)
        # nearest power of 2 to the number of samples in a window
        max_audio_sample_size = 2 ** int(np.ceil(np.log2(self.feature_extractor.win_samples)))
        return {
            "max_audio_sample_size": max_audio_sample_size,
            "labels": labels,
            # no automatic way to get size of tensor arena. Current best way is trial and error and when value found,
            # update the param in the config
            "tensor_arena_size": tensor_arena_size,
            "extractor": {
                "params": self.feature_extractor.ap_params,
                "n_windows": n_windows,
                "hex_vals": extractor_hex_vals,
            },
            "model": {"hex_vals": model_hex_vals, "operators": op_list}
        }

    @classmethod
    def get_toolchain(cls):
        return ESP_IDF_v5_2
