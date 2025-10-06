
# NAS-TAD: Neural Architecture Search for Temporal Action Detection

## Installation

To set up the environment, please follow these steps:

1. Install PyTorch=2.0.1, Python=3.10:
```
bash
conda create -n nastad python=3.10
conda activate nastad
conda install pytorch=2.0.1 torchvision=0.15.2 pytorch-cuda=11.8 -c pytorch -c nvidia

```
2. Install mmaction2:
```
bash
pip install openmim
mim install mmcv==2.0.1
mim install mmaction2==1.1.0
```
3. Install requirements:
```
bash
pip install -r requirements.txt
```
## Data Structure

The data should be organized in the following structure:

```

data/ 
├── thumos14/ 
│ ├── annotations/ 
│ ├── features/ 
│ └── classifier/ 
└── activitynet_1.3/  
│ ├── annotations/ 
│ ├── features/ 
│ └── classifier/ 
└── hacs/ 
│ ├── annotations/ 
│ ├── features/ 
│ └── classifier/ 
└── ...
```

## Training

`torchrun --nnodes={num_node} --nproc_per_node={num_gpu} --rdzv_backend=c10d --rdzv_endpoint=localhost:0 tools/train.py {config}`

- `num_node` is often set as 1 if all gpus are allocated in a single node. 
- `num_gpu` is the number of used GPU.
- `config` is the path of the config file.

For example:
- Training NAS-TAD on ActivityNet with 4 GPU.
```bash
torchrun \
    --nnodes=1 \
    --nproc_per_node=4 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=localhost:0 \
    tools/train.py configs/nastad/anet_internvideo2_nas_retrain.py
```


## Testing

`torchrun --nnodes={num_node} --nproc_per_node={num_gpu} --rdzv_backend=c10d --rdzv_endpoint=localhost:0 tools/test.py {config} --checkpoint {path}`

- `num_node` is often set as 1 if all gpus are allocated in a single node. 
- `num_gpu` is the number of used GPU.
- `config` is the path of the config file.
- `path` is the path of the checkpoint file.

For example:
- Testing NAS-TAD on ActivityNet with 4 GPU.
```bash
torchrun \
    --nnodes=1 \
    --nproc_per_node=4 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=localhost:0 \
    tools/test.py configs/nastad/anet_internvideo2_nas_retrain.py \
    --checkpoint work_dirs/nastad_internvideo2_6b_retrain/latest.pth
```
##

## Results and Models

**ActivityNet-1.3**

[config](configs/nastad/anet_internvideo2_nas_retrain.py)
[model](https://drive.google.com/file/d/1A6VKJAw1lBzdv6U4gifKDoBnFgoXyNOc/view?usp=sharing)

**HACS**

[config](configs/nastad/hacs_internvideo2_nas_retrain.py)
[model](https://drive.google.com/file/d/17_hDDfr0-YbxMtOzot9VUaopVuj4_Ok7/view?usp=sharing)

**THUMOS14**

[config](configs/nastad/thumos14_internvideo2_nas_retrain.py)
[model](https://drive.google.com/file/d/1O8Zc3QdnWAVF7AOs5AoUZ1moJgID5K5Q/view?usp=sharing)

**FineAction**

[config](configs/nastad/fineaction_internvideo2_nas_mixer_retrain.py)
[model](https://drive.google.com/file/d/1UCTHSQDXoQuSKO3iSc-WRKm6tujqUt_H/view?usp=sharing)
