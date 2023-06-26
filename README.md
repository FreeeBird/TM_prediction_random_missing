
# Network traffic matrix prediction with incomplete data via masked matrix modeling

## Introduction
This repo is the implementation of "Network traffic matrix prediction with incomplete data via masked matrix modeling" (Information Sciences, Under review).  


## Dataset
Two publicly available datasets are utilized to validate the proposed prediction method, namely the Abilene and GÉANT datasets. They provide the statistical traffic volume data of the real network traffic trace from the American Research and Education Network (Abilene)  and the Europe Research and Education Network (GÉANT) .

| **Topology** | **Nodes** | **Flows** | **Links** | **Interval** | **Horizon** | **Records** |
| ------------ | --------- | --------- | --------- | ------------ | ----------- | ----------- |
| **Abilene**  | 12        | 144       | 15        | 5 min        | 6 months    | 48046       |
| **GÉANT**    | 23        | 529       | 38        | 15 min       | 4 months    | 10772       |

## Framework

1. Masked matrix modeling-based matrix completion
   - Masked matrix modeling (Mask generation, Pre-filling, and Reconstruction)
   - 3D-UNet module
3. Traffic matrix prediction
   - LSTM2D module
![image](https://github.com/FreeeBird/Network-traffic-matrix-prediction-with-incomplete-data-via-masked-matrix-modeling/assets/22734806/e98d2159-484c-4030-92e2-c3ef58932261)


## Results

Baselines:

![image](https://github.com/FreeeBird/Network-traffic-matrix-prediction-with-incomplete-data-via-masked-matrix-modeling/assets/22734806/034dc991-59e3-4fe0-9041-2abb2cdafe49)

![image](https://github.com/FreeeBird/Network-traffic-matrix-prediction-with-incomplete-data-via-masked-matrix-modeling/assets/22734806/f91e2ffe-50d8-4086-8a70-379a42ed1a59)


## Environment

python=3.7.9

torch==1.7.0

tsai==0.3.0

numpy==1.19.2

...

*more details can be found at pytorch-gpu.yml.

## Getting Started

### Config

```
config.py
config = Config(
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    gpu=0,
    cpu=os.cpu_count(),
    model = 'UNet3D_LSTM2D',
    bilinear = True,
    kernel_size = 5,
    in_chan = 1,
    dataset = 'abilene', # abilene or geant
    epochs=200,
    batch_size=32,
    learning_rate=0.0001,
    seq_len=26,  # previous timestamps to use
    pre_len=1,  # number of timestamps to predict
    dim_model = 64,
    rounds = 3,
    heads = 1,
    dim_ff = 512,
    train_rate = 0.6,
    test_rate=0.2,
    rnn_layers =3,
    encoder_layers =1,
    dropout = 0.2,
    missing_ratio = 0.4,
    std = 0.05,
    early_stop = 15,
    flow = 144, # for abilene
    # flow = 529, # for geant
    lw=0.5,
    test_during_training=True
)
```

### Training

```python
python prediction_with_md_train.py
```

...

## Citation

...
## Reference
[44] L. A. Mesquita, K. D. Assis, L. V. Pinho, R. C. Almeida Jr, E. F.Simas Filho, M. S. Alencar, Resource planning on elastic optical networks using traffic matrix prediction, AEU-International Journal of Electronics and Communications 134 (2021) 153615.  
[45] B. Yu, H. Yin, Z. Zhu, Spatio-temporal graph convolutional networks:a deep learning framework for traffic forecasting, in: Proceedings of the 27th International Joint Conference on Artificial Intelligence,2018, pp. 3634–3640.
