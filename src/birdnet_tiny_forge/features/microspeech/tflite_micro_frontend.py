# Copyright 2023 The TensorFlow Authors.
# Copyright 2024 BirdNeT-Team
# Copyright 2024 fold ecosystemics
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# NOTICE: Modifications Copyright 2024 BirdNET-Team, fold ecosystemics
# - Removed absl flags, which are not in use in this project
# - Use python logging for debug prints
# - Remove _main() entrypoint as this file won't be called as script


from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import tensorflow as tf
from pydantic import BaseModel
from tflite_micro.python.tflite_micro import runtime
from tflite_micro.python.tflite_micro.signal.ops import (
    energy_op,
    fft_ops,
    filter_bank_ops,
    window_op,
)

logger = logging.getLogger("birdnet-tiny-forge")


class _GenerateFeature(tf.Module):
    """Generate feature tensor from audio window samples"""

    def __init__(self, name: str, params: FeatureParams, detail: str):
        super().__init__(name=name)
        self._params = params
        window_sample_count: int = int(params.window_size_ms * params.sample_rate / 1000)
        hann_window_weights = window_op.hann_window_weights(window_sample_count, params.window_scaling_bits)
        self._hann_window_weights_tensor = tf.constant(hann_window_weights, name="hann_window_weights")
        self._fft_size, self._fft_size_log2 = fft_ops.get_pow2_fft_length(window_sample_count)
        (
            self._filter_bank_index_start,
            self._filter_bank_index_end,
        ) = filter_bank_ops.calc_start_end_indices(
            self._fft_size,
            params.sample_rate,
            params.filter_bank_number_of_channels,
            params.filter_bank_lower_band_limit_hz,
            params.filter_bank_upper_band_limit_hz,
        )
        self._detail = detail

    def generate_feature_for_frame(self, audio_frame: tf.Tensor) -> tf.Tensor:  # noqa: PLR0915
        # Graph execution does not handle global variables.  Instead, capture the
        # global variable(s) within a closure (_debug_print_internal).
        def _debug_print_internal(*args):
            if tf.executing_eagerly():
                logger.debug(*args)

        _debug_print_internal("*** generate_feature_for_frame ***")
        params = self._params
        detail = self._detail

        # update filter_bank_ops constants
        filter_bank_ops.FILTER_BANK_WEIGHT_SCALING_BITS = params.filter_bank_scaling_bits
        filter_bank_ops.FILTER_BANK_ALIGNMENT = params.filter_bank_alignment
        filter_bank_ops.FILTER_BANK_CHANNEL_BLOCK_SIZE = params.filter_bank_channel_block_size

        _debug_print_internal(f"audio frame output [{detail}]: {audio_frame!r}")

        # apply window to audio frame
        weights = self._hann_window_weights_tensor
        _debug_print_internal(f"window weights output [{detail}]: {weights!r}")
        window_output: tf.Tensor = window_op.window(audio_frame, weights, params.window_scaling_bits)
        _debug_print_internal(f"window output [{detail}]: {window_output!r}")

        # pre-scale window output
        window_output = tf.reshape(window_output, [-1])
        window_scaled_output, scaling_shift = fft_ops.fft_auto_scale(window_output)
        _debug_print_internal(f"scaling shift [{detail}]: {scaling_shift!r}")

        # compute FFT on scaled window output
        _debug_print_internal(f"fft size, log2 [{detail}]: {self._fft_size}, {self._fft_size_log2}")
        fft_output: tf.Tensor = fft_ops.rfft(window_scaled_output, self._fft_size)
        _debug_print_internal(f"fft output [{detail}]: {fft_output!r}")

        index_start = self._filter_bank_index_start
        index_end = self._filter_bank_index_end
        # convert fft output complex numbers to energy values
        _debug_print_internal(f"index start, end [{detail}]: {index_start}, {index_end}")
        energy_output: tf.Tensor = energy_op.energy(fft_output, index_start, index_end)
        # Energy op does not zero indices outside [index_start,index_end).
        # The following operations to zero portions of the energy op output
        # could be much more efficiently performed inside the energy op C++
        # code.
        # Need to convert to tf.int32 or the TfLite converter will not use
        # the correct ops.
        energy_output = tf.cast(energy_output, tf.int32)  # type: ignore
        zeros_head = tf.zeros(index_start, dtype=tf.int32)
        number_of_elements = energy_output.shape.num_elements()
        zeros_tail = tf.zeros(
            number_of_elements - index_end,
            dtype=tf.int32,  # type: ignore
        )
        energy_slice = energy_output[index_start:index_end]
        energy_output = tf.concat([zeros_head, energy_slice, zeros_tail], 0)  # type: ignore
        energy_output = tf.cast(energy_output, dtype=tf.uint32)  # type: ignore
        _debug_print_internal(f"energy output [{detail}]: {energy_output!r}")

        # compress energy output into 40 channels
        filter_output: tf.Tensor = filter_bank_ops.filter_bank(
            energy_output,
            params.sample_rate,
            params.filter_bank_number_of_channels,
            params.filter_bank_lower_band_limit_hz,
            params.filter_bank_upper_band_limit_hz,
        )
        _debug_print_internal(f"filterbank output [{detail}]: {filter_output!r}")

        # scale down filter_output
        filter_scaled_output: tf.Tensor = filter_bank_ops.filter_bank_square_root(filter_output, scaling_shift)
        _debug_print_internal(f"scaled filterbank output [{detail}]: {filter_scaled_output!r}")
        #
        # # noise reduction
        # spectral_sub_bits: int = params.filter_bank_spectral_subtraction_bits
        # filter_noise_output: tf.Tensor
        # filter_noise_estimate: tf.Tensor
        # filter_noise_output, filter_noise_estimate = \
        #     filter_bank_ops.filter_bank_spectral_subtraction(
        #         filter_scaled_output,
        #         num_channels=params.filter_bank_number_of_channels,
        #         smoothing=params.filter_bank_even_smoothing,
        #         alternate_smoothing=params.filter_bank_odd_smoothing,
        #         smoothing_bits=params.filter_bank_smoothing_bits,
        #         min_signal_remaining=params.filter_bank_min_signal_remaining,
        #         clamping=params.filter_bank_clamping,
        #         spectral_subtraction_bits=spectral_sub_bits,
        #     )
        # _debug_print_internal(f'noise output [{detail}]: {filter_noise_output!r}')
        #
        # # automatic gain control (PCAN)
        correction_bits: int = self._fft_size_log2 - int(params.filter_bank_scaling_bits / 2)
        # filter_agc_output: tf.Tensor = pcan_op.pcan(
        #     filter_noise_output,
        #     filter_noise_estimate,
        #     strength=params.pcan_strength,
        #     offset=params.pcan_offset,
        #     gain_bits=params.pcan_gain_bits,
        #     smoothing_bits=params.pcan_smoothing_bits,
        #     input_correction_bits=correction_bits)
        # _debug_print_internal(
        #     f'AGC Noise output [{detail}]: {filter_agc_output!r}')

        # re-scale features from UINT32 to INT16
        feature_post_scale: int = 1 << params.filter_bank_post_scaling_bits
        feature_pre_scale_shift: int = correction_bits
        feature_rescaled_output: tf.Tensor = filter_bank_ops.filter_bank_log(
            filter_scaled_output,
            output_scale=feature_post_scale,
            input_correction_bits=feature_pre_scale_shift,
        )
        _debug_print_internal(f"scaled noise output [{detail}]: {feature_rescaled_output!r}")

        # These scaling values are derived from those used in input_data.py in the
        # training pipeline.
        # The feature pipeline outputs 16-bit signed integers in roughly a 0 to 670
        # range. In training, these are then arbitrarily divided by 25.6 to get
        # float values in the rough range of 0.0 to 26.0. This scaling is performed
        # for historical reasons, to match up with the output of other feature
        # generators.
        # The process is then further complicated when we quantize the model. This
        # means we have to scale the 0.0 to 26.0 real values to the -128 to 127
        # signed integer numbers.
        # All this means that to get matching values from our integer feature
        # output into the tensor input, we have to perform:
        # input = (((feature / 25.6) / 26.0) * 256) - 128
        # To simplify this and perform it in 32-bit integer math, we rearrange to:
        # input = (feature * 256) / (25.6 * 26.0) - 128
        # constexpr int32_t value_scale = 256;
        # constexpr int32_t value_div =
        #     static_cast<int32_t>((25.6f * 26.0f) + 0.5f);
        # int32_t value =
        #     ((frontend_output.values[i] * value_scale) + (value_div / 2)) /
        #     value_div;
        # value -= 128;
        # if (value < -128) {
        #   value = -128;
        # }
        # if (value > 127) {
        #   value = 127;
        # }
        # output[i] = value;

        feature_output: tf.Tensor
        if self._params.use_float_output:
            # feature_rescaled_output is INT16, cast to FLOAT32
            feature_output = tf.cast(feature_rescaled_output, tf.float32)  # type: ignore
            # feature_output will be FLOAT32
            feature_output /= self._params.legacy_output_scaling
        else:
            value_scale = tf.constant(256, dtype=tf.int32)
            value_div = tf.constant(int((25.6 * 26) + 0.5), dtype=tf.int32)
            feature_output = tf.cast(feature_rescaled_output, tf.int32)  # type: ignore
            feature_output = (feature_output * value_scale) + int(value_div / 2)
            feature_output = tf.truncatediv(feature_output, value_div)  # type: ignore
            feature_output += tf.constant(-128, dtype=tf.int32)
            feature_output = tf.clip_by_value(feature_output, clip_value_min=-128, clip_value_max=127)  # type: ignore
            feature_output = tf.cast(feature_output, tf.int8)  # type: ignore

        _debug_print_internal(f"feature output [{detail}]: {feature_output!r}")

        return feature_output


# @dataclass(kw_only=True, frozen=True)
class FeatureParams(BaseModel):
    """
    Feature generator parameters

    Defaults are configured to work with the micro_speech_quantized.tflite model
    """

    sample_rate: int = 16000
    """audio sample rate"""

    window_size_ms: int = 30
    """input window size in milliseconds"""

    window_stride_ms: int = 20
    """input window stride in milliseconds"""

    window_scaling_bits: int = 12
    """input window shaping: scaling bits"""

    filter_bank_number_of_channels: int = 40
    """filter bank channel count"""

    filter_bank_lower_band_limit_hz: float = 125.0
    """filter bank lower band limit"""

    filter_bank_upper_band_limit_hz: float = 7500.0
    """filter bank upper band limit"""

    filter_bank_scaling_bits: int = filter_bank_ops.FILTER_BANK_WEIGHT_SCALING_BITS
    """filter bank weight scaling bits, updates filter bank constant"""

    filter_bank_alignment: int = 4
    """filter bank alignment, updates filter bank constant"""

    filter_bank_channel_block_size: int = 4
    """filter bank channel block size, updates filter bank constant"""

    filter_bank_post_scaling_bits: int = 6
    """filter bank output log-scaling bits"""

    filter_bank_spectral_subtraction_bits: int = 14
    """filter bank noise reduction spectral subtration bits"""

    filter_bank_smoothing_bits: int = 10
    """filter bank noise reduction smoothing bits"""

    filter_bank_even_smoothing: float = 0.025
    """filter bank noise reduction even smoothing"""

    filter_bank_odd_smoothing: float = 0.06
    """filter bank noise reduction odd smoothing"""

    filter_bank_min_signal_remaining: float = 0.05
    """filter bank noise reduction minimum signal remaining"""

    filter_bank_clamping: bool = False
    """filter bank noise reduction clamping"""

    pcan_strength: float = 0.95
    """PCAN gain control strength"""

    pcan_offset: float = 80.0
    """PCAN gain control offset"""

    pcan_gain_bits: int = 21
    """PCAN gain control bits"""

    pcan_smoothing_bits: int = 10
    """PCAN gain control smoothing bits"""

    legacy_output_scaling: float = 25.6
    """Final output scaling, legacy from training"""

    use_float_output: bool = False
    """Use float output if True, otherwise int8 output"""


class AudioPreprocessor:
    """
    Audio Preprocessor

    Args:
      params: FeatureParams, an immutable object supplying parameters for
      the AudioPreprocessor instance
      detail: str, used for debug output (optional, for debugging only)
    """

    def __init__(self, params: FeatureParams, detail: str = "unknown"):
        self._detail = detail
        self._params = params
        self._samples_per_window = int(params.window_size_ms * params.sample_rate / 1000)
        self._feature_generator = _GenerateFeature(name="GenerateFeature", params=self._params, detail=self._detail)
        self._feature_generator_concrete_function = self._get_concrete_function(self._feature_generator)
        self._model = self._get_model(self._feature_generator_concrete_function, self._feature_generator)
        self._tflm_interpreter = runtime.Interpreter.from_bytes(self._model)

    def _get_concrete_function(self, feature_generator):
        shape = [1, self._samples_per_window]
        fg = feature_generator
        func = tf.function(func=fg.generate_feature_for_frame)
        feature_generator_concrete_function = func.get_concrete_function(tf.TensorSpec(shape=shape, dtype=tf.int16))  # type: ignore
        return feature_generator_concrete_function

    def _get_model(self, concrete_fn, feature_gen):
        converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_fn], feature_gen)
        converter.allow_custom_ops = True
        model = converter.convert()
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(tf.lite.experimental.Analyzer.analyze(model_content=model))
        return model

    def generate_feature_using_tflm(self, audio_frame: tf.Tensor) -> tf.Tensor:
        """
        Generate a single feature for a single audio frame.  Uses TensorFlow
        graph execution and the TensorFlow model converter to generate a
        TFLM compatible model.  This model is then used by the TFLM
        MicroInterpreter to execute a single inference operation.

        Args:
          audio_frame: tf.Tensor, a single audio frame (self.params.window_size_ms)
          with shape (1, audio_samples_count)

        Returns:
          tf.Tensor, a tensor containing a single audio feature with shape
          (self.params.filter_bank_number_of_channels,)
        """
        self._tflm_interpreter.set_input(audio_frame, 0)
        self._tflm_interpreter.invoke()
        result = self._tflm_interpreter.get_output(0)
        return tf.convert_to_tensor(result)

    def reset_tflm(self):
        """
        Reset TFLM interpreter state

        Re-initializes TFLM interpreter state and the internal state
        of all TFLM kernel operators.  Useful for resetting Signal
        library operator noise estimation and other internal state.
        """
        self._tflm_interpreter.reset()

    def generate_tflite_file(self) -> Path:
        """
        Create a .tflite model file

        The model output tensor type will depend on the
        'FeatureParams.use_float_output' parameter.

        Returns:
          Path object for the created model file
        """
        if self._params.use_float_output:
            type_name = "float"
        else:
            type_name = "int8"
        fname = Path(tempfile.gettempdir(), "audio_preprocessor_" + type_name + ".tflite")
        with open(fname, mode="wb") as file_handle:
            file_handle.write(self._model)
        return fname
