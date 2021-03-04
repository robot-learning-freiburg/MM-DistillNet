# MM-DistillNet
PyTorch code for training MM-DistillNet for multimodal knowledge distillation. http://rl.uni-freiburg.de/research/multimodal-distill

This is the code accompanying the paper "There is More than Meets the Eye: Self-Supervised Multi-Object Detection and Tracking with Sound by Distilling Multimodal Knowledge". 

It allows the user to train our network MM-DistillNet on 4 GPUs with 2.4 Gb memory each (The expected runtime is 7 days).


# Setup
The environment was saved using the command:
```
pip freeze > requirements.txt
```
One can load the required packages with
```
pip install -r requirements.txt
```

# Getting Started
To train a model we use the following command:
```
python train.py --config <path to a config file>
```
After that run is complete, we can evaluate how good it was with:
```
python evaluate.py --config <path to a config file> --checkpoint <path to checkpoint file to evaluate>
```
Notice that running and evaluation depends on a config file. Our recipe (the one that gave the best performance) is config/best.cfg and the best weights are in best/ folder

# Results
The evaluation results of our method, after bayesian optimization, are (more details can be found in the paper):

| Method  | KD | mAP@Avg | mAP@0.5 | mAP@0.75 | CDx | CDy |
  | :--- | ------------- |------------- | ------------- | ------------- | ------------- | ------------- |
  |StereoSoundNet[4] | RGB | 44.05 | 62.38 | 41.46 | 3.00 | 2.24 |
  | :--- | ------------- |------------- | ------------- | ------------- | ------------- | ------------- |
  |MM-DistillNet | RGB | 61.62 | 84.29 | 59.66 | 1.27 | 0.69 |


# Credits
The following repositories provided a network used in this work. Each file individually comments on this, but here also for reference:
- Metrics: https://github.com/eriklindernoren/PyTorch-YOLOv3
- Yet Another EfficientDet: https://github.com/zylo117/Yet-Another-EfficientDet-Pytorch

## License
For academic usage, the code is released under the [GPLv3](https://www.gnu.org/licenses/gpl-3.0.en.html) license. For any commercial purpose, please contact the authors.
