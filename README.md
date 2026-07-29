
# Information Bottleneck Guided Hybrid Neural Architecture Search for Temporal Action Detection in Untrimmed Videos

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
- Training NAS-TAD on ActivityNet with 4 GPUs.
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
- Testing NAS-TAD on ActivityNet with 4 GPUs.
```bash
torchrun \
    --nnodes=1 \
    --nproc_per_node=4 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=localhost:0 \
    tools/test.py configs/nastad/anet_internvideo2_nas_retrain.py \
    --checkpoint work_dirs/nastad_internvideo2_6b_retrain/latest.pth
```
## NAS Supernet Training and Search

This project follows a two-stage NAS workflow before final retraining:

1. Train an IB-guided supernet over a candidate architecture range.
2. Run evolutionary search on the trained supernet to rank candidate architectures.
3. Decode the searched architecture code and write the selected stem/branch choices into the retraining config.
4. Retrain the selected architecture with `tools/train.py`.

The IB supernet uses `NASProjMixer` to sample hybrid Mamba/attention architectures. `VideoMambaSuiteNASIB` combines the detector loss with the input, spatial classification, and temporal regression information bottleneck losses implemented by `ActionFormerHeadIB`.

Install the Mamba and FlashAttention dependencies before training a supernet:

```bash
pip install mamba-ssm causal-conv1d
pip install flash-attn --no-build-isolation
```

Train the IB supernet with the configured architecture range:

```bash
torchrun \
    --nnodes=1 \
    --nproc_per_node=4 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=localhost:0 \
    tools/train_supernet_IB.py \
    configs/nastad/anet_internvideo2_nas_supernet_ib.py \
    --min_size=10 \
    --max_size=20 \
    --work_dir exps/anet/nastad_internvideo2_6b_supernet_ib
```

Search a trained IB supernet with evolutionary search. The search ranks candidates by information bottleneck score and writes `checkpoint-*.pth.tar` under `work_dir`.

```bash
torchrun \
    --nnodes=1 \
    --nproc_per_node=4 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=localhost:0 \
    tools/search_supernet_IB.py \
    configs/nastad/anet_internvideo2_nas_supernet_ib.py \
    --resume_supernet exps/anet/nastad_internvideo2_6b_supernet_ib/gpu4_id0/checkpoint/epoch_29.pth \
    --min_size=10 \
    --max_size=20 \
    --work_dir exps/anet/nastad_internvideo2_6b_ib_search
```

Candidate encoding is `[stem_length, *stem_choices, *branch_choices]`, where each choice is `0`, `1`, or `2`. Select a searched candidate, put its stem and branch choices into the relevant retraining configuration, then run `tools/train.py` for the final architecture.



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
