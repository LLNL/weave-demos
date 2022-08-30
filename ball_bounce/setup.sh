#!/bin/bash

python3 -m venv ball_bounce_demo_venv
source ball_bounce_demo_venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python -m ipykernel install --user --name ball-bounce-demo

echo "Setup complete!"
