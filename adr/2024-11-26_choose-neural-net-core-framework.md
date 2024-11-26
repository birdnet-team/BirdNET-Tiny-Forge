# Keras for neural network creation and training

## Status

Accepted

## Context

Within this project, we create specialized neural network models, train them, export them to an embedded target. There's a number of powerful frameworks to facilitate the model creation and training, such as PyTorch, Tensorflow, JAX.

While there is a degree of interoperability within these frameworks, it's not the case that any model created in one framework can run in the others.

Different embedded targets can import models from different frameworks. For example, ADI's own embedded ecosystem uses and extends PyTorch for model creation and training, then exports PyTorch models to the embedded engine counterpart. The ESP32S3, instead, supports TFLite, based on Tensorflow.

It would be in theory possible to run separate pipelines for different targets, with only the first stages of data preprocessing and possibly feature extraction in common. On the other hand, being able to train a model once, and deploy it to multiple targets, would be valuable. Reducing the number of pipelines to maintain would also be good.

Keras (from version 3 on) provides an abstraction layer over the three biggest frameworks, and is able to export the same model to different backends.

## Decision

Use Keras to provide a degree of independence from the underlying neural framework. Embedded counterparts that support one of the three major frameworks can then be integrated more easily.

## Consequences

Pros:
- Write one pipeline per model instead of N (where N = number of targets).
- One trained model can be exported to multiple targets.
- Models and their training code can still be coded in one of the three underlying frameworks.

Cons:
- Depending on the model graph, some operations might not be available in all the underlying frameworks. This means some model architectures might not be exportable to all targets.
- While one pipeline is better than N, there's overhead to make sure the one "framework-agnostic" pipeline actually does end up in a "target-agnostic" model. While we expect this burden to be less than that of maintaining N pipelines, it's still there and its extent not fully clear at the time of writing.
