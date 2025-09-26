import keras
import tensorflow as tf
from spec_augment import SpecAugment


class GaussianNoise(keras.layers.Layer):
    def __init__(self, stddev=0.1, **kwargs):
        super().__init__(**kwargs)
        self.stddev = stddev

    def call(self, inputs, training=None):
        if training:
            noise = tf.random.normal(
                shape=tf.shape(inputs),
                mean=0.0,
                stddev=self.stddev,
                dtype=inputs.dtype
            )
            return inputs + noise
        return inputs

    def get_config(self):
        config = super().get_config()
        config.update({"stddev": self.stddev})
        return config


def make_augmentation_pipeline(
        mixup_alpha,
        noise_stddev,
        sa_freq_mask_max_width,
        sa_time_mask_max_width,
        sa_n_freq_mask,
        sa_n_time_mask,
):
    augm = [
        keras.layers.MixUp(mixup_alpha),
        GaussianNoise(noise_stddev),
        SpecAugment(sa_freq_mask_max_width, sa_time_mask_max_width, sa_n_freq_mask, sa_n_time_mask),
    ]
    return keras.layers.Pipeline(augm)
