# MegReader
A project for research in text detection and recognition using PyTorch 1.2.

This project is originated from the research repo, which heavily relies on closed-source libraries, of CSG-Algorithm team of Megvii(https://megvii.com).
We are in ongoing progress to transfer models into this repo gradually, released implementations are listed in [Progress](#progress).

## Highlights

- Implementations of representative text detection and recognition methods.
- An effective framework for conducting experiments: We use yaml files to configure experiments, making it convenient to take experiments.
- Thorough logging features which make it easy to follow and analyze experimental results.
- CPU/GPU compatible for training and inference.
- Distributed training support.

## Install

### Requirements

`pip install -r requirements.txt`

- Python3.7
- PyTorch 1.2 and CUDA 10.0.
- gcc 5.5(Important for compiling)

### Compile cuda ops (If needed)
```
cd PATH_TO_OPS

python setup.py build_ext --inplace
```
ops may be used:
- DeformableConvV2 `assets/ops/dcn`
- CTC2DLoss `ops/ctc_2d`

### Configuration(optional)

Edit configurations in `config.py`.

## Training

See detailed options: `python3 train.py --help`

## Datasets
We provide data loading implementation with annotation packed with json for quick start.
Datasets used in our recognition experiments can be downloaded from [onedrive](https://megvii-my.sharepoint.cn/:f:/g/personal/wanzhaoyi_megvii_com/EjkcrpmiW6hJrUKY-0fEBRABvNMtYniUPfWLVptMmy9-6w?e=bJaYFo).

### Non-distributed

`python3 train.py PATH_TO_EXPERIMENT.yaml --validate --visualize --name NAME_OF_EXPERIMENT`

Following we provide some of configurations of the released recognition models:

- CRNN: `experiments/recognition/crnn.yaml`
- 2D CTC: `experiments/recognition/res50-ppm-2d-ctc.yaml`
- Attention Decoder: `experiments/recognition/fpn50-attention-decoder.yaml`

### Distributed(recommended for multi-gpu training)

`python3 -m torch.distributed.launch --nproc_per_node=NUM_GPUS train.py PATH_TO_EXPERIMENT.yaml -d --validate`

<!--
### Setup your own dataset
-->


## Evaluating

See detailed options: `python3 eval.py --help`.

Keeping ratio tesing is recommended: `python3 eval.py PATH_TO_EXPERIMENT.yaml --resize_mode keep_ratio`


### Model zoo
Trained models are comming soon.
<!--
Our trained model can be downloaded from xxx.
-->

## Progress
### Recognition Methods
- [x] 2D CTC
- [x] CRNN
- [x] Attention Decoder
- [ ] Rectification

### Detection Methods
- [x] Text Snake
- [x] EAST

### End-to-end
- [ ] Mask Text Spotter

## Contributing

[Contributing.md](CONTRIBUTING.md)
