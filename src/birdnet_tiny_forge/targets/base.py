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

"""Abstract base class for all targets. It provides a base implementation for the
rendering of a project template given a context dict (that is populated by the inheriting child class).
"""

import os
import shutil
from abc import ABC, abstractmethod
from pathlib import Path

from jinja2 import Environment, FileSystemLoader
from keras import Model
from tensorflow.data import Dataset

from birdnet_tiny_forge.features.base import FeatureExtractorBase
from birdnet_tiny_forge.toolchains.base import ToolchainBase

TEMPLATES_PATH = Path(__file__).parents[1] / "templates"
TEMPLATE_EXTENSION = "jinja"


class BuildTargetBase(ABC):
    @abstractmethod
    def __init__(
        self,
        feature_extractor: FeatureExtractorBase,
        model: Model,
        reference_dataset: Dataset,
        labels_dict: dict,
        params: dict,
    ):
        self.feature_extractor = feature_extractor
        self.model = model
        self.reference_dataset = reference_dataset
        self.labels_dict = labels_dict
        self.params = params

    def validate(self):
        """Validate Target inputs, including the compatibility of feature extractor and model."""
        self.check_feature_extractor_compatible()
        self.check_model_compatible()
        self.check_labels(self.labels_dict)
        # TODO: dataset checks?
        # TODO: params checks?

    def check_model_compatible(self):
        """Check model compatibility with target.

        :raises: ValueError if model is not compatible with target.
        """
        return self.do_check_model_compatible()

    def check_feature_extractor_compatible(self) -> bool:
        """Check feature extractor compatibility with target.

        :raises: ValueError if extractor is not compatible with target.
        """
        from birdnet_tiny_forge.registries import FeatureExtractorsRegistry

        if not FeatureExtractorsRegistry.contains(self.feature_extractor.__class__):
            raise ValueError(f"{self.feature_extractor.__class__.__name__} not in feature extractors register.")
        return self.do_check_feature_extractor_compatible()

    @classmethod
    def check_labels(cls, labels_dict: dict):
        vals = sorted(labels_dict.values())
        if not all(isinstance(k, str) for k in labels_dict):
            raise ValueError("Labels dictionary keys must be strings")
        if not all(isinstance(v, int) for v in vals):
            raise ValueError("Labels dictionary values must be integers")
        if (vals[0] != 0) or (vals[-1] != (len(labels_dict) - 1)):
            raise ValueError("Labels dictionary values must be sequential integers starting at 0")

    @abstractmethod
    def do_check_feature_extractor_compatible(self):
        """
        :raises: ValueError if extractor is not compatible with target.
        """
        raise NotImplementedError

    @abstractmethod
    def do_check_model_compatible(self):
        """
        :raises: ValueError if model is not compatible with target.
        """
        raise NotImplementedError

    @classmethod
    def setup_template_environment(cls, template_dir):
        cls.env = Environment(loader=FileSystemLoader(template_dir))

    @classmethod
    def get_templates_dir(cls):
        """Return templates dir for this build target. Inheriting classes can override this if needed."""
        from birdnet_tiny_forge.registries import BuildTargetsRegistry

        target_templates_path = TEMPLATES_PATH / BuildTargetsRegistry.get_name(cls)
        return target_templates_path

    @abstractmethod
    def extract_context(self) -> dict:
        raise NotImplementedError

    def process_target_templates(self, outdir: Path) -> None:
        self.validate()

        # get available projects and their root template folders
        templates_dir = self.get_templates_dir()
        self.setup_template_environment(templates_dir)

        # extract context to be passed to jinja render
        context = self.extract_context()

        # Render and save each template
        if not outdir.exists():
            raise ValueError(f"{str(outdir)} does not exist, please create it.")

        # Process each file in the template directory
        for template_name in self.env.list_templates():
            template_path = Path(template_name)
            if template_path.suffix.lstrip(".") == TEMPLATE_EXTENSION:
                # Render and save the template file
                template = self.env.get_template(template_name)
                output_path = outdir / template_path.with_suffix("")
                with output_path.open("w") as f:
                    f.write(template.render(context))
            else:
                # Copy non-template files directly
                src_path = templates_dir / template_name
                dst_path = outdir / template_name
                os.makedirs(dst_path.parent, exist_ok=True)
                shutil.copyfile(src_path, dst_path)

    @classmethod
    @abstractmethod
    def get_toolchain(cls) -> type[ToolchainBase]:
        raise NotImplementedError
