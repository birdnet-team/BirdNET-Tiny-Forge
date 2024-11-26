# Frequently Asked Questions

## How do I add my own data?
Place your data inside the `data/01_raw` folder, according to the following structure:
```
data
  01_raw
    my_awesome_dataset  # or however you want to name your dataset
      train
        class_1     # name of class for your data
          rec1.wav  # name of file doesn't matter, but must be .WAV
          ...
        class_2
        ...
        class_n
      test
        class_1
        class_2
        ...
        class_n
```

Then modify `conf/base/catalog.yml`:
```
audio_clips:
  type: partitions.PartitionedDataset
  path: "data/01_raw/my_awesome_dataset"  # point this path to your dataset
  dataset: birdnet_tiny_forge.datasets.WavDataset
  
# rest of the file  
```

Only this layout is supported at the moment of writing.

## How do I run the pipeline?
Run 
```tinyforge run```
to run the full pipeline, from data preprocessing to flashing of generated firmware to device.

# Unfrequently Asked Questions

## How do I add a new feature extractor?
- Create a child class of `birdnet_tiny_forge.features.FeatureExtractorBase`, e.g. `MyFeatureExtractor`
- Register the class with the `FeatureExtractorRegistry` in `birdnet_tiny_forge.registries`, e.g. `FeatureExtractorsRegistry.add("awesome_feature_extractor", MyFeatureExtractor)`

To use the extractor:
- in `/conf/base/parameters_feature_extraction.yml`, under `feature_extraction.extractor`, set "awesome_feature_extractor" (same name used to register the class in the registry) as the value, and place any params you need to pass it under `feature_extraction.params`. For example:
  ```yaml
  feature_extraction:        
    extractor: awesome_feature_extractor  # same name as used to register class in registry
    params:
      my_param: 1
    # rest of feature_extraction pipeline params
  ```
  
Keep in mind that any custom feature extraction implemented in the pipeline will need a corresponding and equivalent extraction step running on the embedded device, and a way to convert one to the other that can run in the target code generation step. 

## How do I add a new model?
- Create a child class of `birdnet_tiny_forge.models.base.ModelFactoryBase`, e.g. `MyModel`. This factory creates instances of `keras.Model`.
- Register the class with the `ModelsRegistry` in `birdnet_tiny_forge.registries`, e.g. `ModelsRegistry.add("my_model", MyModel)

To use the model:
- in `/conf/base/parameters_model_training.yml`, set `model_training.model.type` to `predefined`, and `model_training.model.name` to e.g. `my_model`. Then set the `compile_params` and `params` as needed. For example:
  ```yaml
  model_training:  
    model:
      type: predefined
      name: my_model  # same name as used to register the class in registry
      compile_params:
        optimizer: adam
        loss:
          class_name: SparseCategoricalCrossentropy
          config:
            from_logits: false
        metrics:
          - accuracy
      params:
        my_param: 0.1
  # rest of the params for model training pipeline
  ```
  
Depending on the target, your model might or might not be able to be converted / run. E.g. targets running tflite-micro only support a subset of operations supported by Keras/Tensorflow. ESP32S3 targets using the ESP port of tflite-micro only accelerate a subset of operations supported by tflite-micro (so a model might run, but much slower than expected).

## How do I add a new hypermodel?
- Create a child class of `birdnet_tiny_forge.hypersearch.base.HypersearchBase`, e.g. `MyHyperModel`
- Register the class with the `HypersearchRegistry` in `birdnet_tiny_forge.registries`, e.g. `HypersearchRegistry.add("awesome_hyper_model", MyHyperModel)

To use the hypermodel:
- in `/conf/base/parameters_model_training.yml`, set `model_training.model.type` to `hypersearch`, and `model_training.model.name` to e.g. `awesome_hyper_model`. Then set the `compile_params` and `params` as needed. For example:
  ```yaml
  model_training:  
    model:
      type: hypersearch  # same name as used to register the class in registry
      name: awesome_hyper_model
      compile_params:
        optimizer: adam
        loss:
          class_name: SparseCategoricalCrossentropy
          config:
            from_logits: false
        metrics:
          - accuracy
      params:
        my_param: 0.1
  # rest of the params for model training pipeline
  ```

## How do I add a new target?
A target needs three things:
- a template project inside the `birdnet-tiny-forge/templates` folder, containing normal code as well as files templated with Jinja2. See `birdnet-tiny-forge/templates/ESP32S3_bird_logger` as an example.
- a class inheriting from `TargetBase`, and registered with `birdnet_tiny_forge.registries.TargetsRegistry`. This class implements logic to take the active FeatureExtractorBase and trained Model instances, as well as any parameter used in the pipeline, and use this information to populate the templates in the template project. The class also specifies a `ToolchainBase` instance that is in charge of compiling its generated code and flash it to the device.
- a `ToolchainBase` child class implementing logic to compile code for the target device and flash it. Optimally, this makes use of a docker container to perform the actions, to guarantee reproducibility. 

The configuration in `conf/base/parameters_target_codegen.yml` might look like this:
```yaml
target_codegen:
  target: "name_of_your_target"
  cache_build: true  # if true, code is built in .cache folder, so it's not rebuilt from scratch every time
  toolchain_params:  # any parameter to be passed to the toolchain object at construction
    my_toolchain_param: 1
  params:  # any additional parameter you want the TargetBase child class to have access to during code gen     
    my_additional_param: []
```

## How do I check how my feature extractor / model is performing?

Make sure you installed the development dependencies of the project,  ```poetry install --with dev```.
Then run `kedro viz` at the root of the project. See (https://docs.kedro.org/projects/kedro-viz/en/stable/index.html)[Kedro viz]'s documentation on how to use the tool.
Selecting the model training pipeline in the visualization tool, click on the plotting nodes to check the produced plots (e.g. confusion matrix, training metrics). Selecting the feature extraction pipeline, you can check the plot nodes relating to feature extraction.

Selecting the experiment tracking tab in the sidebar, you can compare different runs of the pipeline, allowing you to e.g. compare metrics for different model architectures during exploration.

## Can I just run parts of the pipeline?
Yes. This project uses (https://docs.kedro.org/en/stable/)[Kedro] under the hood, so it is possible to run just parts of the full pipeline. 

The available (sub-)pipelines are:
`data_preprocessing`, `feature_extraction`, `model_training`, `target_codegen`, with each subsequent step relying on the result of the preceding ones. Each step persists its outputs to the `data` folder, such that the next steps don't require the previous one to be re-run everytime.

For example, if you wrote a new model but didn't change the input data and feature extraction, and you already ran the pipeline before, you can just do `tinyforge run --pipeline model_training` to run the `model_training` component exclusively. It will re-use the persisted output generated by `data_preprocessing` and `feature_extraction`.

There's more filtering features baked into Kedro, please consult its (https://docs.kedro.org/en/stable/)[documentation] to know more.