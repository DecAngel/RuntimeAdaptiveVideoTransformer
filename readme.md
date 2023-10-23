# RAVT: Runtime Adaptive Video Transformer

## Installation
- install python=3.9
- setup virtual environment with conda or virtualenv
- install pytorch (examine your cuda version and modify [requirements_pytorch.txt](requirements_pytorch.txt))
```shell
python -m pip install -r requirements_pytorch.txt
```
- install other dependencies (modify [requirements_others.txt](requirements_others.txt))
```shell
python -m pip install -r requirements_others.txt
```
- create dataset link
```shell
cd <ravt dir>
ln -s <your dataset path> ./datasets
```
- use yolox cocoeval
```shell
cd <ravt dir>
chmod +x ./ravt/ninja
```

## TODO
- [x] implement 3d attention
- [x] fix phase init call multiple times 
- [x] fix random resize
- [x] fix ema saving & loading
- [x] fix kornia augmentation with different length input
