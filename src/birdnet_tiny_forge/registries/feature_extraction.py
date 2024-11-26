"""Registry for feature extractors"""

from birdnet_tiny_forge.features.microspeech import MicroSpeechExtractor
from birdnet_tiny_forge.registries.base import RegistryBase


class FeatureExtractorsRegistry(RegistryBase):
    pass


FeatureExtractorsRegistry.add("microspeech_extractor", MicroSpeechExtractor)
