Attention-based Multi-input Neural network
=============
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/attention-based-multi-input-deep-learning/drug-discovery-on-egfr-inh)](https://paperswithcode.com/sota/drug-discovery-on-egfr-inh?p=attention-based-multi-input-deep-learning)

<img src="https://i.ibb.co/jg5kzd5/egfr-architecture-new.jpg" width="700">

## How to install

Using `conda`:
```bash
conda env create -n egfr -f environment.yml
conda activate egfr
```

## Usage

The working folder is `egfr-att/egfr` for the below instruction.

#### To train with Train/Test scheme, use:
```bash
python single_run.py --mode train
```
The original data will be splitted into training/test parts with ratio 8:2. 
When training completed, to evaluate on test data, use:
```bash
python single_run.py --mode test --model_path <MODEL-IN-TRAINED_MODELS-FOLDER>
# For example:
python single_run.py --mode test --model_path data/trained_models/model_TEST_BEST
```
ROC curve plot for test data will be placed in egfr/vis folder.

#### To train with 5-fold cross validation scheme, use:
```bash
python cross_val.py --mode train
``` 
When training completed, to evaluate on test data, use:
```bash
python cross_val.py --mode test --model_path <MODEL-IN-TRAINED_MODELS-FOLDER>
# For example:
python cross_val.py --mode test --model_path data/trained_models/model_TEST_BEST
```
ROC curve plot for test data will be placed in `egfr/vis/` folder.

#### Attention weight visualization
To visualized attention weight of the model, use:
```bash
python weight_vis.py --dataset <PATH-TO-DATASET> --modelpath <PATH-TO-MODEL>
# For example:
python weight_vis.py --dataset data/egfr_10_full_ft_pd_lines.json --modelpath data/trained_models/model_TEST_BEST
```
By default, all data will be used to to extract attention weights. However, 
only samples with prediction output over a threshold (0.2) are chosen.

