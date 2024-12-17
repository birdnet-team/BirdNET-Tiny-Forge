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
<a href="https://birdnet.cornell.edu/"><img src="https://raw.githubusercontent.com/kahst/BirdNET-Analyzer/refs/heads/main/docs/assets/img/birdnet_logo.png" width="80px" alt="BirdNET logo"></a>
<a href="https://fold.eco"><img src="https://raw.githubusercontent.com/birdnet-team/BirdNET-Tiny-Forge/refs/heads/master/img/fold-ecosystemics.png" height="80px" alt="fold ecosystemics logo"></a>
</div>

## Documentation
Please find the documentation [here](https://birdnet-tiny-forge.readthedocs.io/en/latest/).

## Supported devices

### Active

- [ESP32-S3-Korvo-2 V3.0](https://docs.espressif.com/projects/esp-adf/en/latest/design-guide/dev-boards/user-guide-esp32-s3-korvo-2.html)

### Planned
- ADI MAX78000
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
./build_tflite.sh

# Project uses poetry for python dependency management
pip install poetry

# Finally, install all deps
poetry install
```

## Development setup

```bash
poetry install --with dev
```

## License

See [LICENSE file](https://github.com/birdnet-team/BirdNET-Tiny-Forge/blob/master/LICENSE) at project root.

## Contributing

See our [contribution documentation](https://github.com/birdnet-team/BirdNET-Tiny-Forge/blob/master/CONTRIBUTING.MD).

## Funding

This project is supported by Jake Holshuh (Cornell class of '69) and The Arthur Vining Davis Foundations. Our work in the K. Lisa Yang Center for Conservation Bioacoustics is made possible by the generosity of K. Lisa Yang to advance innovative conservation technologies to inspire and inform the conservation of wildlife and habitats.

The German Federal Ministry of Education and Research is funding the development of BirdNET through the project "BirdNET+" (FKZ 01|S22072).
Additionally, the German Federal Ministry of Environment, Nature Conservation and Nuclear Safety is funding the development of BirdNET through the project "DeepBirdDetect" (FKZ 67KI31040E).

## Partners

BirdNET is a joint effort of partners from academia and industry.
Without these partnerships, this project would not have been possible.
Thank you!

![Our partners](https://tuc.cloud/index.php/s/KSdWfX5CnSRpRgQ/download/box_logos.png)
