#!/bin/bash

# Upgrade pip
pip install --upgrade pip

# Install dependencies with specific version constraints
pip install tensorflow==2.9.3
pip install keras==2.9.0
pip install -r requirements.txt
