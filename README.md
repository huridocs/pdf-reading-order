<h3 align="center">PDF Reading Order</h3>
<p align="center">This tool returns the reading order of a PDF</p>

## Quick Start
Create venv:

    make install_venv

Get the token types from a PDF:

    source venv/bin/activate
    python src/predict.py /path/to/pdf


## Train a new model

Get the labeled data tool from the GitHub repository:

    https://github.com/huridocs/pdf-labeled-data

Change the paths in src/config.py

LABELED_DATA_ROOT_PATH = /path/to/pdf-labeled-data/project
TRAINED_MODEL_PATH = /path/to/save/trained/model

Create venv:

    make install_venv

Train a new model:

    source venv/bin/activate
    python src/train.py

## Use a custom model
    
    python src/predict.py /path/to/pdf --model-path /path/to/model
    