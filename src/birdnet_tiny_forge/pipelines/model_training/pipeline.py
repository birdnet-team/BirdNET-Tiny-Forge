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

"""This pipeline performs a model hyperparameter / architecture search if so specified in configuration,
or loads a pre-defined architecture. It then trains and tests the model."""

from kedro.pipeline import Pipeline, node, pipeline

from birdnet_tiny_forge.pipelines.model_training.nodes import (
    do_hyperparameter_search,
    get_class_weight,
    get_confusion_matrix,
    get_model,
    make_tf_datasets,
    plot_training_metrics,
    test_model,
    train_model,
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=make_tf_datasets,
                inputs=["model_input", "labels_dict"],
                outputs=["tf_datasets", "reference_dataset", "features_shape"],
                name="make_tf_datasets",
            ),
            node(
                func=get_class_weight,
                inputs={"data": "model_input", "labels_dict": "labels_dict"},
                outputs="class_weight",
                name="get_class_weight",
            ),
            node(
                func=get_model,
                inputs={
                    "search_type": "params:model_training.model.type",
                    "model": "params:model_training.model.name",
                    "input_shape": "features_shape",
                    "params": "params:model_training.model.params",
                    "compile_params": "params:model_training.model.compile_params",
                    "labels_dict": "labels_dict",
                },
                outputs="model_pre_search",
                name="get_model",
            ),
            node(
                func=do_hyperparameter_search,
                inputs={
                    "model": "model_pre_search",
                    "datasets": "tf_datasets",
                    "class_weight": "class_weight",
                },
                outputs="model_post_search",
                name="do_hyperparameter_search",
            ),
            node(
                func=train_model,
                inputs={
                    "model": "model_post_search",
                    "datasets": "tf_datasets",
                    "class_weight": "class_weight",
                    "n_epochs": "params:model_training.n_epochs",
                },
                outputs=["model_trained", "training_history"],
                name="train_model",
            ),
            node(
                func=plot_training_metrics,
                inputs="training_history",
                outputs="training_metrics_plot",
                name="plot_training_metrics",
            ),
            node(
                func=test_model,
                inputs={
                    "model": "model_trained",
                    "datasets": "tf_datasets",
                },
                outputs=["y_true", "y_pred"],
                name="test_model",
            ),
            node(
                func=get_confusion_matrix,
                inputs=["y_true", "y_pred", "labels_dict"],
                outputs="confusion_matrix_plot",
                name="get_confusion_matrix",
            ),
        ]
    )
