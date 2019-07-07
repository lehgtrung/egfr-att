Attention-based Multi-input Neural network
=============
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/attention-based-multi-input-deep-learning/drug-discovery-on-egfr-inh)](https://paperswithcode.com/sota/drug-discovery-on-egfr-inh?p=attention-based-multi-input-deep-learning)

<img src="https://i.imgur.com/4FBRFh6.jpg" width="700">

## How to install
This package requires:
* [RDkit](http://www.rdkit.org/docs/Install.html)
* Pytorch version 1.1.0

## Usage

#### To train with Train/Test scheme, use:
```bash
python simple_run.py --mode train
```
The original data will be splitted into training/test parts with ratio 8:2. 
When training completed, to evaluate on test data, use:
```bash
python simple_run.py --mode test --model_path <MODEL-IN-TRAINED_MODELS-FOLDER>
```
ROC curve plot for test data will be placed in egfr/vis folder.

#### To train with 5-fold cross validation scheme, use:
```bash
python cross_val.py --mode train
``` 
When training completed, to evaluate on test data, use:
```bash
python cross_val.py --mode test --model_path <MODEL-IN-TRAINED_MODELS-FOLDER>
```
ROC curve plot for test data will be placed in egfr/vis folder.

#### Attention weight visualization
To visualized attention weight of the model, use:
```bash
python weight_vis.py --dataset <PATH-TO-DATASET> --modelpath <PATH-TO-MODEL>
```
By default, all data will be used to to extract attention weights. Especially, 
only samples with prediction output over a threshold (0.2) are chosen.

