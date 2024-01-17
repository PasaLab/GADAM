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
We provide five benchmark datasets containing injected anomalies: `Cora, Citeseer, Pubmed, ACM` and `BlogCatalog`, as well as two real-world datasets containing organic anomalies: `books` and `reddit`, which can be found in `data/data.rar`. Anomalies are injected through the unified interface provided by the pygod library.
Two OGB datasets `ogbn-arxiv` and `ogbn-products` are not included due to memory limits.

#### 2.1 Preprocessed data
We recommend using preprocessed data for fair comparasion, unzip `data/data.rar` and make directory structure as follows:
```
└─data
    │      Cora.bin
    │      Citeseer.bin
    │      ...
└─run.py
```

#### 2.2 Customized data
For two OGB datasets or customized dataset, contextual and structural anomalies can be generated via 'pygod.generator'. See https://docs.pygod.org/en/latest/pygod.generator.html for details.

### 3. Anomaly detection
Run `python run.py --data Cora --local-lr 1e-3 --local-epochs 100 --global-lr 5e-4 --global-epochs 50` to perform anomaly detection.
