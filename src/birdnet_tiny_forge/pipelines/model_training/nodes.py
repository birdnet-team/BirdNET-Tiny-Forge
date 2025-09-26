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

"""Nodes to be used in model training pipeline"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import tensorflow as tf
from keras import Model, losses, metrics
from keras.src.utils import to_categorical
from plotly.subplots import make_subplots
from sklearn.metrics import ConfusionMatrixDisplay

from birdnet_tiny_forge.models.base import checkpoint_session
from birdnet_tiny_forge.pipelines.model_training.augmentation import make_augmentation_pipeline
from birdnet_tiny_forge.registries import ModelsRegistry
import logging


logger = logging.getLogger("tinyforge")


def make_tf_datasets(
    data: pd.DataFrame,
    labels_dict: dict[str, int],
    augm_params,
    buffer_size=10000,
    seed=42,
    batch_size=32,
):
    datasets = {}
    features_shape = tuple(data["features_shape"][0])  # TODO: fix this ugliness
    assert data["features_shape"].apply(lambda x: tuple(x) == features_shape).all()
    augm_pipe = make_augmentation_pipeline(**augm_params)

    for group, group_data in data.groupby("split"):
        # shape (tf backend): batches, rows, cols, channels
        features = np.array(group_data["features"].to_list()).reshape((-1, features_shape[0], features_shape[1], 1))
        int_labels = group_data["label"].apply(lambda x: labels_dict[x]).values
        labels = to_categorical(int_labels)
        dataset = tf.data.Dataset.from_tensor_slices((features, labels))
        dataset = dataset.shuffle(buffer_size=buffer_size, seed=seed).batch(batch_size)
        if group == "train":
            dataset = dataset.map(
                lambda x, y: (augm_pipe(x, training=True), y),
                num_parallel_calls=tf.data.AUTOTUNE
            )
        datasets[group] = dataset

    # TODO: formalize the split names and presence
    reference_dataset = datasets["train"].shuffle(10000).take(100)
    for x, y in reference_dataset.take(1):
        features_shape = x.shape[1:]
    return datasets, reference_dataset, features_shape


def get_class_weight(data, labels_dict):
    train_labels = data[data["split"] == "train"]["label"]
    l_counts = dict(train_labels.value_counts())
    tot_counts = len(train_labels)
    if set(l_counts) != set(labels_dict):
        missing_classes = list(sorted(set(labels_dict).difference(l_counts)))
        raise ValueError(f"Training data does not contain all classes. Missing classes: {missing_classes}")
    class_weight = {v: tot_counts / l_counts[k] for k, v in labels_dict.items()}
    return class_weight


def get_model(model, input_shape, compile_params, params, labels_dict):
    loss_config = compile_params.pop("loss")
    loss = losses.get(loss_config)
    metrics_ = [metrics.get(x) for x in compile_params.pop('metrics', [])]
    model_factory = ModelsRegistry.get_item(model)
    model = model_factory.create(class_count=len(labels_dict), input_shape=input_shape, **params)
    model.compile(loss=loss, metrics=metrics_, **compile_params)
    return model


def train_model(model, datasets, n_epochs, class_weight):
    train_ds = datasets["train"].cache().shuffle(10000).prefetch(tf.data.AUTOTUNE)
    valid_ds = datasets["validation"].cache().prefetch(tf.data.AUTOTUNE)
    with checkpoint_session() as checkpointer:
        history = model.fit(
            train_ds,
            validation_data=valid_ds,
            epochs=n_epochs,
            class_weight=class_weight,
            callbacks=[checkpointer]
        )
        model = checkpointer.get_best_checkpoint()
    return model, history


def plot_training_metrics(training_history):
    metrics = training_history.history
    train_metrics = []
    val_metrics = []
    for metric in metrics:
        if metric.startswith("val_"):
            train_metric = metric.split("val_")[1]
            if train_metric in metrics:
                val_metrics.append(metric)
                train_metrics.append(train_metric)

    fig = make_subplots(
        rows=len(train_metrics), cols=2, subplot_titles=list([y for x in train_metrics for y in [x, x]]),
        shared_yaxes="rows", shared_xaxes="all"
    )
    for i, train_col in enumerate(train_metrics, start=1):
        fig.add_trace(
            go.Scatter(y=metrics[train_col], name=train_col),
            row=i,
            col=1,
        )
        fig.add_trace(
            go.Scatter(y=metrics[f"val_{train_col}"], name=f"val_{train_col}"),
            row=i,
            col=2,
        )
    fig.update_xaxes(title_text="Epoch", row=len(train_metrics), col=1)
    fig.update_xaxes(title_text="Epoch", row=len(train_metrics), col=2)
    fig.update_layout(showlegend=True, height=200 * len(train_metrics))
    return fig


def test_model(model: Model, datasets: dict[str, tf.data.Dataset]):
    test_ds = datasets["test"].cache().prefetch(tf.data.AUTOTUNE)
    y_true = np.argmax(np.concat(list(test_ds.map(lambda x, y: y).as_numpy_iterator())), axis=1)
    preds = model.predict(test_ds)
    y_pred = np.argmax(preds, axis=1)
    return y_true, y_pred


def get_confusion_matrix(y_true, y_pred, labels_dict: dict[str, int]):
    labels = [lbl for (lbl, i) in sorted(labels_dict.items(), key=lambda x: x[1])]
    return ConfusionMatrixDisplay.from_predictions(y_true, y_pred, display_labels=labels).figure_


def quantize_model(model):
    raise NotImplementedError
