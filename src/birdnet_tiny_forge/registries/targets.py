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

"""Registry for targets"""

from birdnet_tiny_forge.registries.base import RegistryBase
from birdnet_tiny_forge.targets.korvo2_bird_logger import Korvo2BirdLogger


class BuildTargetsRegistry(RegistryBase):
    pass


BuildTargetsRegistry.add("korvo2_bird_logger", Korvo2BirdLogger)
