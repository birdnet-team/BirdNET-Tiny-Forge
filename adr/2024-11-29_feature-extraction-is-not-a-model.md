# Feature extraction is not a Keras model

## Status

Accepted

## Context

The feature extraction step also needs to be converted to code that runs on the embedded side. If we take the tflite micro-speech example from the tflite repo, the feature extraction is *also* a tflite model.

For the trained model, the conversion is assumed to happen via a framework specialized in converting a model from one of the major frameworks running on desktop, to a specialized library that can run it on embedded.

So the question arises, should the feature extraction object also be a Keras model, such that conversion may happen via the same framework?

The feature extractor is more likely to rather see the definition of custom code, as opposed to common NN operations. This is because for this project's use case, feature extraction is most likely to be audio signal processing rather than a neural graph of some sort. Case in point, the micro-speech example in tflite-micro relies on a custom "signal" library extending tflite-micro by implementing some specific signal processing operations that make sense in the context of speech processing. In other words, to be able to write the micro-speech example, the authors had to anyway implement custom tensorflow ops on both sides (desktop / embedded). Importantly, this conversion to tensorflow ops was only done at a later time; initially, the feature extraction code wasn't done within the framework at all.

Feature extraction can be stateful, but is assumed to not need training on data. If it does need to be trained, it makes sense to move its operations in the actual model (creating an end-to-end model).

## Decision

Here we decide to keep the feature extractor object a generic one.

## Consequences

By keeping the feature extractor generic, we acknowledge the fact that feature extraction code is less likely to be mappable to existing ops in the NN frameworks, and needs custom code on both the desktop / embedded side. We leave the option open to write the custom code by extending the used framework with ops on both sides, like in tflite-micro's micro-speech example. But otherwise, users are also free to not do that and rather write custom code outside the framework. It's still their responsibility to make sure the two sides of the implementation are numerically equivalent or approximately so to the required level of approximation.

Different targets will also need to write their own conversion code, and embedded implementation, for each feature extractor.


