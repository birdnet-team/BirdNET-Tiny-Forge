# BirdNET-Tiny Forge
[![Powered by Kedro](https://img.shields.io/badge/powered_by-kedro-ffc900?logo=kedro)](https://kedro.org)

<p align="center">
<img src="https://raw.githubusercontent.com/birdnet-team/BirdNET-Tiny-Forge/refs/heads/master/img/birdnet-tiny-forge.png" width="128" alt="BirdNET-Tiny logo, an 8-bit image of a bird on an anvil">
</p>

BirdNET-Tiny Forge simplifies the training of BirdNET-Tiny models and their deployment on embedded devices.

BirdNET-Tiny Forge is a collaboration between
[BirdNET](https://birdnet.cornell.edu/)
and [fold ecosystemics](https://fold.eco).

<div align="center">
<a href="https://birdnet.cornell.edu/"><img src="https://github.com/birdnet-team/BirdNET-Analyzer/blob/main/docs/_static/logo_birdnet_big.png?raw=true" width="80px" alt="BirdNET logo"></a>
<a href="https://fold.eco"><img src="https://raw.githubusercontent.com/birdnet-team/BirdNET-Tiny-Forge/refs/heads/master/img/fold-ecosystemics.png" height="80px" alt="fold ecosystemics logo"></a>
</div>

## Documentation
Please find the documentation [here](https://birdnet-tiny-forge.readthedocs.io/en/latest/).

## Supported devices

### Active

- [ESP32-S3-Korvo-2 V3.0](https://docs.espressif.com/projects/esp-adf/en/latest/design-guide/dev-boards/user-guide-esp32-s3-korvo-2.html)

### Planned
- BirdWatcher PUC


## ⚠️ Development Status ⚠️

This project is in early development stage, and not recommended for production use:
- APIs may change without notice
- Features might be incomplete
- Documentation may be outdated

## System requirements
- Ubuntu 24.04 LTS. Other Debian-based OSs and Ubuntu versions will most likely work with minimal tweaks if any, but are not officially supported.
- Python 3.10 or 3.11 and corresponding `python-dev` library (`python3.10-dev` and `python3.11-dev` in Ubuntu apt repository)
- [bazel](https://bazel.build/install/bazelisk) (needed to build a patched python wheel of tflite-micro)
- [docker](https://docs.docker.com/engine/install/ubuntu/)


## Installation

```bash
# Download and build patched version of tflite (it makes custom signal ops available, and pins the tensorflow version to the one used in this repo)
./build_tflite.sh tflite-micro

# Project uses poetry for python dependency management
pip install poetry

# Finally, install all deps
poetry install
```

## Prepare your data
### Xenocanto
If using [xeno-canto](https://xeno-canto.com) data to train your network, please make sure to review its [terms](https://xeno-canto.org/about/terms) to check your usage is compatible with them.

- Please create an account on https://xeno-canto.com, which grants you the API key we will use to download bird recordings to train our network.

- Create a `species.txt` file, where each line contains the scientific name of a species you want to train on.

- Call `xc-download --api-key <your API key> --species-file <path to species file> --n-recs <number of recordings per species to download>`

### AudioSet

- Create a `audioset.txt` file, with each line being a class in AudioSet you wish to download data for.
- Run `audioset-download --labels-file <path to audioset labels> --limit <max number of files to download per class>`

### Custom data
If you have your own recordings you'd like to train on, first isolate the bird calls you're interested in, creating audio clips. Place your clips in `data/01_raw/audio_clips`, with the following structure:

```
audio_clips
├── <label, e.g. Apteryx mantelli>
│  ├── <recording, e.g. abc123.wav>
│  ├── ...
├── <label, e.g. Apteryx owenii>
├── ...
```

## Run the pipeline


## Development setup

```bash
poetry install --with dev
```

## License

See [LICENSE file](https://github.com/birdnet-team/BirdNET-Tiny-Forge/blob/master/LICENSE) at project root.

## Contributing

See our [contribution documentation](https://github.com/birdnet-team/BirdNET-Tiny-Forge/blob/master/CONTRIBUTING.MD).

## Funding

This project is supported by Jake Holshuh (Cornell class of ’69) and The Arthur Vining Davis Foundations. Our work in the K. Lisa Yang Center for Conservation Bioacoustics is made possible by the generosity of K. Lisa Yang to advance innovative conservation technologies to inspire and inform the conservation of wildlife and habitats.

The development of BirdNET is supported by the German Federal Ministry of Education and Research through the project “BirdNET+” (FKZ 01|S22072). The German Federal Ministry for the Environment, Nature Conservation and Nuclear Safety contributes through the “DeepBirdDetect” project (FKZ 67KI31040E). In addition, the Deutsche Bundesstiftung Umwelt supports BirdNET through the project “RangerSound” (project 39263/01).

## Partners

BirdNET is a joint effort of partners from academia and industry.
Without these partnerships, this project would not have been possible.
Thank you!

![Our partners](https://tuc.cloud/index.php/s/KSdWfX5CnSRpRgQ/download/box_logos.png)
