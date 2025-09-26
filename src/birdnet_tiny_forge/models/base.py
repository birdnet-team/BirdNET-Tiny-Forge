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
import contextlib
import shutil
import uuid
from abc import ABC, abstractmethod
from pathlib import Path

import keras
import numpy as np
import pandas as pd
from keras import Model
from keras.src.callbacks import Callback
from birdnet_tiny_forge.constants import _CACHE_DIR
import logging


logger = logging.getLogger("tinyforge")


class ModelFactoryBase(ABC):
    @classmethod
    @abstractmethod
    def create(cls, class_count, input_size, **kwargs) -> Model:
        raise NotImplementedError


class TrainingSession(Callback):
    SELECT_FNS = {"min": np.argmin, "max": np.argmax}
    CHKP_NAME = "chkpt_{epoch:06d}.keras"

    def __init__(self, checkpoint_dir: Path):
        super().__init__()
        self._checkpoint_dir = checkpoint_dir
        self._metrics = []
        self._checkpoints = []

    def on_epoch_end(self, epoch, logs=None):
        self._log_checkpoint(self.model, epoch, logs)

    def _log_checkpoint(self, model: keras.Model, epoch: int, metrics: dict | None):
        if metrics is not None:
            for key, value in metrics.items():
                self._metrics.append({"metric": key, "value": value, "epoch": epoch})
        chkpt_path = self._checkpoint_dir / self.CHKP_NAME.format(epoch=epoch)
        keras.saving.save_model(
            model,
            filepath=chkpt_path
        )
        logger.info("Saved checkpoint at %s", chkpt_path)

    def _make_metrics_df(self):
        return pd.DataFrame.from_records(self._metrics)

    def get_best_checkpoint(self, metric="val_loss", mode="min") -> Model:
        """
        Get the best checkpoint based on the specified metric
        """
        if mode not in ["min", "max"]:
            raise ValueError(f"Unknown mode {mode}. Supported: min, max")

        metrics_df = self._make_metrics_df()
        if metrics_df.empty:
            raise ValueError("No checkpoints found")

        available_metrics = set(metrics_df["metric"].unique())
        if metric not in available_metrics:
            raise ValueError(
                f"Unknown metric {metric}. Available metrics for the run: {available_metrics}"
            )
        metric_df = metrics_df[metrics_df["metric"] == metric]
        best_row = metric_df["value"].idxmin() if mode == "min" else metric_df["value"].idxmax()
        best_epoch = metric_df.loc[best_row]["epoch"]
        chkpt_path = self._checkpoint_dir / self.CHKP_NAME.format(epoch=best_epoch)
        if not chkpt_path.exists():
            raise ValueError(f"Epoch {best_epoch} not found in checkpoints")
        logger.info("Loaded checkpoint at %s", chkpt_path)
        model = keras.saving.load_model(chkpt_path)
        return model

    def get_metrics(self):
        metrics_df = self._make_metrics_df()
        pivoted = metrics_df.pivot(index="epoch", columns=["metric"])
        pivoted.columns = [x[1] for x in pivoted.columns]
        return pivoted



@contextlib.contextmanager
def checkpoint_session(clean_up: bool = False):
    id_ = str(uuid.uuid4())
    checkpoint_dir = (_CACHE_DIR / id_)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Checkpoints saved at %s", checkpoint_dir)
    yield TrainingSession(checkpoint_dir)
    if clean_up:
        logger.info("Deleting checkpoint folder")
        shutil.rmtree(checkpoint_dir)
