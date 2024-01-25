#!/bin/bash

# Install dependencies
pip install -r requirements.txt

# Run model training
python3 train_model.py

# Start the server
python3 server.py