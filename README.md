# There is More than Meets the Eye: Self-Supervised Multi-Object Detection and Tracking with Sound by Distilling Multimodal Knowledge

**MM-DistillNet** is a novel framework that is able to perform Multi-Object Detection and tracking using only ambient sound during inference time. The framework leverages on our new new MTA loss function that facilitates the distillation of information from multimodal teachers (RGB, thermal and depth) into an audio-only student network.


![Illustration of MM-DistillNet](/images/intro.png)

This repository contains the **PyTorch implementation** of our CVPR'2021 paper [There is More than Meets the Eye: Self-Supervised Multi-Object Detection and Tracking with Sound by Distilling Multimodal Knowledge](https://arxiv.org/abs/2103.01353). The repository builds on [PyTorch-YOLOv3 Metrics](https://github.com/eriklindernoren/PyTorch-YOLOv3) and [Yet-Another-EfficientDet-Pytorch](https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch) codebases.

If you find the code useful for your research, please consider citing our paper:
```
@article{riverahurtado2021mmdistillnet,
  title={There is More than Meets the Eye: Self-Supervised Multi-Object Detection and Tracking with Sound by Distilling Multimodal Knowledge},
  author={Rivera Valverde, Francisco and Valeria Hurtado, Juana and Valada, Abhinav},
  booktitle={Proceedings of the IEEE/CVF conference on computer vision and pattern recognition},
  year={2021}
}
```
## Demo
http://rl.uni-freiburg.de/research/multimodal-distill

## System Requirements
* Linux 
* Python 3.7
* PyTorch 1.3 
* CUDA 10.1

**IMPORTANT NOTE**: These requirements are not necessarily mandatory. However, we have only tested the code under the above settings and cannot provide support for other setups.

## Installation
a. Create a conda virtual environment.
```shell
git clone https://github.com/robot-learning-freiburg/MM-DistillNet.git
cd MM-DistillNet
conda create -n mmdistillnet_env
conda activate mmdistillnet_env
```
b. Install dependencies
```bash
pip install -r requirements.txt
```

## Prepare datasets and configure run
We also supply our large-scale multimodal dataset with over 113,000 time-synchronized frames of RGB,
depth, thermal, and audio modalities, available at http://multimodal-distill.cs.uni-freiburg.de/#dataset

Please make sure the data is available in the directory under the name `data`.

The binary download contains the expected folder format for our scripts to work. The path where the binary was extracted must be updated in the configuration files, in this case `configs/best.cfg`.

Our dataset download also contains pre-trained teached models that need to be available during training:
```bash
ln -sf data/trained_models .
```

Additionally, the file `configs/best.cfg` contains support for different parallelization strategies and GPU/CPU support (using PyTorch's [DataParallel](https://pytorch.org/docs/stable/generated/torch.nn.DataParallel.html)  and [DistributedDataParallel](https://pytorch.org/docs/master/generated/torch.nn.parallel.DistributedDataParallel.html))

Due to disk space constraints, we provide a mp3 version of the audio files. Librosa is known to be slow with mp3 files, so we also provide a mp3->pickle conversion utility. The idea is,
that before training we convert the audio files to a spectogram and store it to a pickle file.

```bash
mp3_to_pkl.py --dir <path to the dataset>
```

## Training and Evaluation
### Training Procedure
Edit the config file appropriately in configs folder. Our best recipe is found under `configs/best.cfg`.

```
python train.py --config <path to a config file>
```
To run the full dataset 
We our method using 4 GPUs with 2.4 Gb memory each (The expected runtime is 7 days). After training, the best model would be stored under `<exp_name>/best.pth.tar`. This file can be used to evaluate the performance of the model.

### Evaluation Procedure
Evaluate the performance of the model:
```
python evaluate.py --config <path to a config file> --checkpoint <path to checkpoint file to evaluate>
```
### Results
The evaluation results of our method, after bayesian optimization, are (more details can be found in the paper):

| Method  | KD | mAP@Avg | mAP@0.5 | mAP@0.75 | CDx | CDy |
  | :--- | ------------- |------------- | ------------- | ------------- | ------------- | ------------- |
  |StereoSoundNet[4] | RGB | 44.05 | 62.38 | 41.46 | 3.00 | 2.24 |
  | :--- | ------------- |------------- | ------------- | ------------- | ------------- | ------------- |
  |MM-DistillNet | RGB | 61.62 | 84.29 | 59.66 | 1.27 | 0.69 |

## Pre-Trained Models
Our best pre-trained model can be found on the dataset installation path.

## Acknowledgements
We have used utility functions from other open-source projects. We especially thank the authors of:
- [yTorch-YOLOv3](https://github.com/eriklindernoren/PyTorch-YOLOv3)
- [Yet Another EfficientDet](https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch)

## Contacts
* [Abhinav Valada](https://rl.uni-freiburg.de/people/valada)
* [Juana Valeria Hurtado](https://rl.uni-freiburg.de/people/hurtado)
* [Francisco Rivera](https://github.com/franchuterivera)

## License
For academic usage, the code is released under the [GPLv3](https://www.gnu.org/licenses/gpl-3.0.en.html) license. For any commercial purpose, please contact the authors.
