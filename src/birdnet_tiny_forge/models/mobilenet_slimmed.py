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

"""Very simple MobileNet-inspired model"""

from keras import Model, layers
from keras.src.applications.mobilenet import _conv_block, _depthwise_conv_block

from birdnet_tiny_forge.models.base import ModelFactoryBase


class MobilenetSlimmed(ModelFactoryBase):
    @classmethod
    def create(cls, class_count, input_shape, n_filters_1=32, n_filters_2=64, dropout=0.02):
        inputs = layers.Input(shape=input_shape)
        x = _conv_block(inputs, filters=n_filters_1, alpha=1, kernel=(10, 4), strides=(5, 2))
        x = _depthwise_conv_block(x, pointwise_conv_filters=n_filters_1, alpha=1, block_id=1)
        x = _depthwise_conv_block(x, pointwise_conv_filters=n_filters_2, alpha=1, block_id=2)
        x = _depthwise_conv_block(x, pointwise_conv_filters=n_filters_1, alpha=1, block_id=3)
        x = layers.GlobalMaxPooling2D(keepdims=True)(x)
        x = layers.Dropout(dropout, name="dropout1")(x)
        x = layers.Flatten()(x)
        x = layers.Dense(class_count)(x)
        outputs = layers.Softmax()(x)
        return Model(inputs, outputs, name="mobilenet_slimmed")
