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

"""Abstract Base Class for a ModelFactory, a simple object that can create a model.
Meant to be registered with the model registry to make it easy to pick a model given runtime parameters.
"""
from abc import ABC, abstractmethod
from keras import Model


class ModelFactoryBase(ABC):
    @classmethod
    @abstractmethod
    def create(cls, class_count, input_size, **kwargs) -> Model:
        raise NotImplementedError
