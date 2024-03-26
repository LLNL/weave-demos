#!/bin/bash

python3 -m venv cz_tutorials_venv
source cz_tutorials_venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python3 -m ipykernel install --user --name cz-tutorials-demo

echo "Setup complete!"
