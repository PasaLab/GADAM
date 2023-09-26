# GADAM
This is the code associated with the submission "Boosting Graph Anomaly Detection with Adaptive Message Passing".

### 1. Dependencies (with python >= 3.8):
```
conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.6 -c pytorch -c conda-forge
conda install -c dglteam/label/cu116 dgl
conda install scikit-learn
pip install pygod
```

### 2. Datasets

Unzip 'data/data.rar' and make directory structure as follows:
```
└─data
    │      Cora.bin
    │      Citeseer.bin
    │      ...
└─run.py
```
### 3. Anomaly detection
Run `run.py --data Cora` to perform anomaly detection.
