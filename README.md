# Universal ONNX Converter

Tool to convert Darknet and PyTorch models into ONNX formats optimized for edge devices. It focuses on Opset 11 compatibility and uses static values to ensure stability on restricted hardware.

## Features

- Exports models with static value Opset 11 for maximum compatibility.
- Supports both raw tensor export and export with specific detect layers.
- Includes validation for all ONNX exports to ensure parity.
- Supports OpenCV inference for testing raw output ONNX models.
- Tested primarily on basic darknet tiny models.
- Can convert Darknet models to PyTorch format (.pt).
- Designed for edge devices to avoid dynamic operator issues.

## Environment

The use of a virtual environment is strongly recommended for stability.
Recommended Python version: 3.9.25

## Installation

pip install -r requirements.txt

## Usage

### Darknet to ONNX
python main.py --mode darknet --cfg model.cfg --weights model.weights --output model.onnx --shape 1 3 416 416 --validate

### PyTorch to ONNX
python main.py --mode pytorch --weights model.pt --output model.onnx --shape 1 3 640 640 --validate

### Darknet to PyTorch
python main.py --mode darknet --cfg model.cfg --weights model.weights --output-mode pytorch --output model.pt

## Arguments

--no-yolo-layer
Output raw tensors instead of the decoded detection layer.

--no-simplify
Skip the ONNX simplification pass.

--validate
Run validation suite and comparison between models.

--shape
Specify the input size (e.g. 1 3 416 416).

--help
Show available command line arguments and options.

--tutorial
Display detailed usage examples and a quick start guide.

## Status

This project is a prototype. It has been tested on basic darknet tiny models and is intended for development and testing purposes on edge hardware.
