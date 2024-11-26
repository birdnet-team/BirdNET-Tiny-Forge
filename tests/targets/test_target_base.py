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

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import pytest
from keras import Model

from birdnet_tiny_forge.features.base import FeatureExtractorBase
from birdnet_tiny_forge.models.base import ModelFactoryBase
from birdnet_tiny_forge.registries import FeatureExtractorsRegistry
from birdnet_tiny_forge.targets.base import TEMPLATE_EXTENSION, BuildTargetBase
from birdnet_tiny_forge.toolchains.base import ToolchainBase


class UnknownModel(Model):
    def __init__(self):
        super().__init__()
        pass


class CompatibleExtractor(FeatureExtractorBase):
    def run(self, sample_rate: int, data: np.ndarray) -> np.ndarray:
        pass


class IncompatibleExtractor(FeatureExtractorBase):
    def run(self, sample_rate: int, data: np.ndarray) -> np.ndarray:
        pass


class CompatibleModel(ModelFactoryBase):
    @classmethod
    def create(cls, class_count, input_size, **kwargs):
        return Model(name="compatible")


class IncompatibleModel(ModelFactoryBase):
    @classmethod
    def create(cls, class_count, input_size, **kwargs):
        return Model(name="incompatible")


# Dummy target class for testing
class DummyBuildTarget(BuildTargetBase):
    # noqa: PLR0913
    def __init__(self, feature_extractor=None, model=None, reference_dataset=None, labels_dict=None, params=None):
        super().__init__(feature_extractor, model, reference_dataset, labels_dict, params)

    def do_check_feature_extractor_compatible(self):
        if not isinstance(self.feature_extractor, CompatibleExtractor):
            raise ValueError

    def do_check_model_compatible(self) -> bool:
        if self.model.name == "incompatible":
            raise ValueError

    @classmethod
    def get_toolchain(cls) -> type[ToolchainBase]:
        return ToolchainBase

    def extract_context(self):
        return {"model_name": self.model.name, "params": self.params}


@pytest.fixture
def dummy_template_dir():
    with TemporaryDirectory() as tempdir:
        temp_path = Path(tempdir)
        (temp_path / f"dummy_template.txt.{TEMPLATE_EXTENSION}").write_text(
            "Model: {{ model_name }}, Params: {{ params }}"
        )
        (temp_path / "do_not_render.txt").write_text("{{ model_name }}")
        yield temp_path


GOOD_LABELS = {"a": 0, "b": 1}
BAD_LABELS = {"a": 0, "b": 2}


class TestTargetBase:
    @pytest.mark.parametrize("model_factory", [CompatibleModel, IncompatibleModel])
    @pytest.mark.parametrize("extractor", [CompatibleExtractor(), IncompatibleExtractor()])
    @pytest.mark.parametrize("labels_dict", [GOOD_LABELS, BAD_LABELS])
    def test_target_compatible(self, model_factory, extractor, labels_dict):
        should_fail = (
            (model_factory == IncompatibleModel)
            or isinstance(extractor, IncompatibleExtractor)
            or (labels_dict == BAD_LABELS)
        )
        model = model_factory.create(0, 0)
        with FeatureExtractorsRegistry.transient_registration("extr", extractor):
            if should_fail:
                with pytest.raises(ValueError):
                    DummyBuildTarget(model=model, feature_extractor=extractor, labels_dict=labels_dict).validate()
            else:
                DummyBuildTarget(model=model, feature_extractor=extractor, labels_dict=labels_dict).validate()

    def test_process_target_templates(self, dummy_template_dir):
        dummy_model_instance = CompatibleModel.create(0, 0)
        from birdnet_tiny_forge.registries import BuildTargetsRegistry

        with (
            patch.object(DummyBuildTarget, "get_templates_dir", return_value=dummy_template_dir),
            FeatureExtractorsRegistry.transient_registration("dummy_extractor", CompatibleExtractor()),
            BuildTargetsRegistry.transient_registration("dummy_target", DummyBuildTarget),
            TemporaryDirectory() as outdir,
        ):
            dummy_params = {"param1": "value1"}
            outdir_path = Path(outdir)
            target = DummyBuildTarget(
                model=dummy_model_instance,
                feature_extractor=CompatibleExtractor(),
                labels_dict=GOOD_LABELS,
                params=dummy_params,
            )
            target.process_target_templates(outdir_path)

            # Check that the output file was created
            output_file = outdir_path / "dummy_template.txt"
            assert output_file.exists()

            # Verify the content of the output file
            expected_content = f"Model: {dummy_model_instance.name}, Params: {str(dummy_params)}"
            assert output_file.read_text() == expected_content

            # Make sure that the non-jinja file is not getting template-substituted
            output_file = outdir_path / "do_not_render.txt"
            assert output_file.exists()
            expected_content = "{{ model_name }}"
            assert output_file.read_text() == expected_content
