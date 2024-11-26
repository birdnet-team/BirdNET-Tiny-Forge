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

"""Hyperparameter search on a Mobilenet-like model"""

from tempfile import mkdtemp

import keras_tuner as kt

from birdnet_tiny_forge.hypersearch.base import HypersearchBase
from birdnet_tiny_forge.models import MobilenetSlimmed


class MobileNetSlimmedHS(HypersearchBase):
    def __init__(
        self,
        class_count,
        input_shape,
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=("accuracy",),
    ):
        super().__init__(class_count, input_shape, optimizer, loss, metrics)

    def build_hypermodel(self, hp):
        n_filters_1 = hp.Int("n_filters_1", min_value=16, max_value=64, step=16)
        n_filters_2 = hp.Int("n_filters_2", min_value=16, max_value=64, step=16)
        model = MobilenetSlimmed.create(
            class_count=self._class_count,
            input_shape=self._input_shape,
            n_filters_1=n_filters_1,
            n_filters_2=n_filters_2,
        )
        model.compile(optimizer=self._optimizer, loss=self._loss, metrics=self._metrics)
        return model

    def run_search(self, train_data, validation_data, **fit_kwargs):
        tdir = mkdtemp(prefix="grid_search")
        tuner = kt.Hyperband(
            self.build_hypermodel,
            objective="val_loss",
            max_epochs=100,
            factor=2,
            directory=tdir,
            project_name="birdnet_tiny",
        )
        tuner.search(train_data, validation_data=validation_data, **fit_kwargs)
        best_hps = tuner.get_best_hyperparameters()[0]
        return tuner.hypermodel.build(best_hps)
