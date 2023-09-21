# RAVT: Runtime Adaptive Video Transformer

## Installation
- install python=3.9
- setup virtual environment with conda or virtualenv
- install pytorch=2.0
```shell
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
```
- install pytorch-lightning
```shell
python -m pip install pytorch-lightning
```
- install image libs
```shell
python -m pip install opencv-python timm imgaug kornia kornia-rs
```
- install dataset server libs
```shell
python -m pip install flask
```
- install cli libs
```shell
python -m pip install fire
```
- install evaluation libs
```shell
python -m pip install pycocotools tensorboard sap_toolkit
```
- install typeguard libs
```shell
python -m pip install typeguard==3.0.2 jaxtyping
```


## TODO
- [ ] implement 3d attention
- [ ] fix phase init call multiple times 