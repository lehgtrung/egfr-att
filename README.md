Attention-based Multi-input Neural network
=============

<img src="https://i.imgur.com/4FBRFh6.jpg" width="700">

## How to install
This package requires:
* [RDkit](http://www.rdkit.org/docs/Install.html)
* Pytorch version 1.1.0

## Usage

Clone the project and get the dataset:
```bash
git clone https://github.com/lehgtrung/egfr-att.git
cd egfr-att/egfr
mkdir data
wget --no-check-certificate \
'https://drive.google.com/uc?export=download&id=17kGQhgzs6qhNJ8wUQM7yJmdenMgw13B2'\
 -O data/data.zip
unzip data/data.zip
```

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

