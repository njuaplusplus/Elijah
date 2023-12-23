pip install pyarrow
# pip install accelerate comet-ml matplotlib datasets tqdm tensorboard tensorboardX torchvision tensorflow-datasets einops pytorch-fid joblib PyYAML kaggle wandb torchsummary torchinfo
pip install -r requirements.txt

cd diffusers
pip install -e .
cd ..

mkdir measure
mkdir datasets
mkdir measure
