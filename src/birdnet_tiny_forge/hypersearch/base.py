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

"""Abstract Base Class for Hypersearch object. This object is meant to perform a hyperparameter search or
architecture search, with a .run_search method returning a concrete Model architecture."""

from abc import ABC, abstractmethod

from keras import Model

# TODO: currently, configuration of hyperparameter types is not supported
#   need to decide on how it should look like in yaml


class HypersearchBase(ABC):
    @abstractmethod
    def __init__(self, class_count, input_shape, optimizer, loss, metrics, **kwargs):  # noqa: PLR0913
        self._class_count = class_count
        self._input_shape = input_shape
        self._optimizer = optimizer
        self._loss = loss
        self._metrics = metrics

    @abstractmethod
    def build_hypermodel(self, hp):
        raise NotImplementedError

    @abstractmethod
    def run_search(self, train_data, validation_data, **fit_kwargs) -> Model:
        raise NotImplementedError
