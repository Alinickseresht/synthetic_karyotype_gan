#!/bin/bash
# First install kaggle-cli: pip install kaggle
# Then set your API key

kaggle datasets download -d your-dataset-id
unzip your-dataset-id.zip -d data/raw/
rm your-dataset-id.zip
