# MicroTS: Microscopy Time Series Processor

[![License](https://img.shields.io/pypi/l/microts.svg?color=green)](https://github.com/yourusername/microts/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/microts.svg?color=green)](https://pypi.org/project/microts)
[![Python Version](https://img.shields.io/pypi/pyversions/microts.svg?color=green)](https://python.org)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/microts)](https://napari-hub.org/plugins/microts)

A napari plugin for processing microscopy time series data, designed to work alongside MicroSAM for efficient segmentation and tracking.

## Features

- Navigate through multiple TIF files in a directory
- Save segmentation masks and tracking data in compatible formats
- Keyboard shortcuts for efficient workflow
- Integration with MicroSAM segmentation workflow

## Usage

Start napari
Load the MicroTS plugin from the Plugins menu
Select your input directory containing TIF files
Set your output directory where masks and tracks will be saved
Use the navigation buttons to browse through your images
Save masks and tracks with the "Save Mask + Track" button

## Keyboard Shortcuts

- N - Next file
- P - Previous file
- W - Save mask and track

## Contributing
Contributions are very welcome. Please feel free to submit a Pull Request.

## License
Distributed under the terms of the [MIT] license.