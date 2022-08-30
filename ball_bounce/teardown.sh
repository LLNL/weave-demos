#!/bin/bash

jupyter kernelspec uninstall -f ball-bounce-demo
rm -rf ball_bounce_demo_venv
rm -rf output*

echo "Teardown complete! Virtual environment, kernel, and data removed."
