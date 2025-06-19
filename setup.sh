#!/bin/bash
set -e

echo "Upgrading pip..."
pip install --upgrade pip

echo "Installing tensorflow-cpu and keras..."
pip install tensorflow-cpu==2.12.0
pip install keras==2.12.0

echo "Installing other dependencies..."
pip install -r requirements.txt
