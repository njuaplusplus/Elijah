#!/bin/bash

# conda create --name elijah_villandiff3.8 python=3.8 anaconda
# conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch

cd diffusers
pip install -e .
cd ..

pip install -r my_requirements.txt
