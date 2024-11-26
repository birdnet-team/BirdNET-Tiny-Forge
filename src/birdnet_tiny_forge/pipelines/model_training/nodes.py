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
from keras import Model, losses
from plotly.subplots import make_subplots
from sklearn.metrics import ConfusionMatrixDisplay

from birdnet_tiny_forge.hypersearch.base import HypersearchBase
from birdnet_tiny_forge.registries import HypersearchRegistry, ModelsRegistry


def make_tf_datasets(
    data: pd.DataFrame,
    labels_dict: dict[str, int],
    buffer_size=10000,
    seed=42,
    tf_batch_size=32,
):
    datasets = {}
    features_shape = tuple(data["features_shape"][0])  # TODO: fix this ugliness
    assert data["features_shape"].apply(lambda x: tuple(x) == features_shape).all()

    for group, group_data in data.groupby("split"):
        # shape (tf backend): batches, rows, cols, channels
        features = np.array(group_data["features"].to_list()).reshape((-1, features_shape[0], features_shape[1], 1))
        int_labels = group_data["label"].apply(lambda x: labels_dict[x]).values
        labels = int_labels
        dataset = tf.data.Dataset.from_tensor_slices((features, labels))
        dataset = dataset.shuffle(buffer_size=buffer_size, seed=seed).batch(tf_batch_size)
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


def get_model(search_type, model, input_shape, compile_params, params, labels_dict):
    loss_config = compile_params.pop("loss")
    loss = losses.get(loss_config)

    if search_type == "predefined":
        model_factory = ModelsRegistry.get_item(model)
        model = model_factory.create(class_count=len(labels_dict), input_shape=input_shape, **params)
        model.compile(loss=loss, **compile_params)
        return model
    if search_type == "hypersearch":
        hs_cls: type[HypersearchBase] = HypersearchRegistry.get_item(model)
        return hs_cls(
            class_count=len(labels_dict),
            input_shape=input_shape,
            loss=loss,
            **compile_params,
            **params,
        )
    raise ValueError(f"Unknown search type {search_type}.")


def do_hyperparameter_search(model, datasets, class_weight):
    if isinstance(model, Model):  # normal model, just return it as-is
        return model
    assert isinstance(model, HypersearchBase)
    hypermodel: HypersearchBase = model
    train_ds = datasets["train"].cache().shuffle(10000).prefetch(tf.data.AUTOTUNE)
    valid_ds = datasets["test"].cache().prefetch(tf.data.AUTOTUNE)
    actual_model = hypermodel.run_search(train_ds, valid_ds, class_weight=class_weight)
    return actual_model


def train_model(model, datasets, n_epochs, class_weight):
    train_ds = datasets["train"].cache().shuffle(10000).prefetch(tf.data.AUTOTUNE)
    valid_ds = datasets["validation"].cache().prefetch(tf.data.AUTOTUNE)
    history = model.fit(train_ds, validation_data=valid_ds, epochs=n_epochs, class_weight=class_weight)
    return model, history


def plot_training_metrics(training_history):
    metrics = training_history.history
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Loss", "Accuracy"))
    fig.add_trace(
        go.Scatter(x=training_history.epoch, y=metrics["loss"], name="loss"),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=training_history.epoch, y=metrics["val_loss"], name="val_loss"),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=training_history.epoch,
            y=100 * np.array(metrics["accuracy"]),
            name="accuracy",
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=training_history.epoch,
            y=100 * np.array(metrics["val_accuracy"]),
            name="val_accuracy",
        ),
        row=1,
        col=2,
    )
    fig.update_layout(showlegend=True)
    fig.update_xaxes(title_text="Epoch", row=1, col=1)
    fig.update_xaxes(title_text="Epoch", row=1, col=2)
    fig.update_yaxes(title_text="Loss [CrossEntropy]", range=[0, None], row=1, col=1)
    fig.update_yaxes(title_text="Accuracy [%]", range=[0, 100], row=1, col=2)
    return fig


def test_model(model: Model, datasets: dict[str, tf.data.Dataset]):
    test_ds = datasets["test"].cache().prefetch(tf.data.AUTOTUNE)
    y_true = np.concat(list(test_ds.map(lambda x, y: y).as_numpy_iterator()))
    preds = model.predict(test_ds)
    y_pred = np.argmax(preds, axis=1)
    return y_true, y_pred


def get_confusion_matrix(y_true, y_pred, labels_dict: dict[str, int]):
    labels = [lbl for (lbl, i) in sorted(labels_dict.items(), key=lambda x: x[1])]
    return ConfusionMatrixDisplay.from_predictions(y_true, y_pred, display_labels=labels).figure_


def quantize_model(model):
    raise NotImplementedError
