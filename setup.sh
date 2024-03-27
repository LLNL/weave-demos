#!/bin/bash

python3 -m venv weave_demos_venv
source weave_demos_venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python3 -m ipykernel install --user --name weave-demos

echo "Setup complete!"
