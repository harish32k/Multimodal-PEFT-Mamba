<div align="center">
<h1>Memo-RIS </h1>
<h3>Mamba-Enabled Minimal-parameter Optimization for
Referring Image Segmentation</h3>

Harish Akula, Tenzin Kunsang

Northeastern University

<div align="left">

## Abstract
Recent advancements in computer vision have extended semantic segmentation to
the challenging domain of Referring Image Segmentation (RIS), where the aim is to segment objects based on natural language descriptions. We propose MEMO-RIS, a parameter-efficient learning approach that adapts the Visual Mamba-based Multi-Modal architecture called the ReMamber, with minimal updates during training. Our work introduces the first parameter-efficient tuning framework specifically designed for the Visual Mamba based models in a multi-modal setting, employing Mamba-based adapters that enable multi-scale cross-modal feature interaction while preserving the computational benefits of state-space models. These adapters are inspired by DETRIS. MEMO-RIS incorporates various adapters connected in parallel to the model’s blocks while adding additional parameters constituting only 5.8% of the backbone parameters. We demonstrate the efficacy of our approach by training on the Ref-ZOM dataset, to enhance our model to handle complex linguistic scenarios like multi-object references and descriptions matching no objects—real-world challenges that traditional RIS models struggle with. Our experimental results show that our parameter-efficient approach achieves competitive performance to methods requiring full fine-tuning, establishing a promising direction for efficiently adapting state-space models to multi-modal understanding tasks.

## Preparation

We used the discovery cluster to create our environment. For the installation of VMamba dependencies, we suggest using a GPU instance with CUDA version of 11.3. Even for the inference we need the GPU. VMamba dependencies require GCC version > 9.2.0 to be installed successfully.

#### Prepare environment
- python 3.10.13: 
  - ``conda create -n remamber python=3.10.13``
- torch 2.1.1 + cu118: 
  - ``pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu118``
- install dependencies:
  - ``pip install -r requirements.txt``
- build kernel for VMamba dependencies: 
  - ``cd selective_scan && pip install .``
- vmamba imagenet checkpoint
  - download from [link](https://drive.google.com/file/d/1O9P6XLuWtUxFa70vwrYCRVedRAFutczV/view?usp=sharing) (This is the VMamba checkpoint pretrained on ImageNet. ). This is essential for the model to start.
  - Create a ``pretrain/`` folder and place the checkpoint in it


#### Model reference 
We have created four configurations - 2 using adapters and 2 that don’t use adapters which serve as a
baseline for comparison. We detail the configurations below.
- Adapters+Full Decoder (Lets refer to it as Half adaptation)
- Adapters for encoder and decoder (Lets call it Full adaptation) - Freeze the entire encoder
and part of the decoder, use adapters for the frozen parts (both encoder and decoder)
- Baseline with fully frozen encoder (Lets term this as Frozen Baseline)
- Baseline with partially frozen encoder (Lets term this as Partial Baseline)


#### Prepare dataset
Follow [ref_dataset/prepare_dataset.md](ref_dataset/prepare_dataset.md) to set up the data pipeline. Also check https://github.com/toggle1995/RIS-DMMI for downloading and installing the Ref-ZOM dataset.

## Training & Evaluation
This implementation only supports multi-gpu, DistributedDataParallel training.

We created batch scripts to run the training job in background on the Explorer SLURM cluster. You can find the scripts in the batch_scripts folder. To run them, use the command "sbatch batch_scripts/script_file.sh"

The training scripts for the models are 

- Half adaptation - batch_scripts/l1_backbone.sh
- Full adaptation - batch_scripts/l2_full_adapt.sh
- Partial Frozen baseline - batch_scripts/l3_baseline.sh
- Full Frozen baseline - batch_scripts/l5_deconly.sh

Other than these configurations, to train ReMamber using 2 GPUs, run:
``` bash
python -m torch.distributed.launch --nproc_per_node=2 \
                                   --use_env main.py \
                                   --model ReMamber_Conv \
                                   --output_dir your/logging/directory \
                                   --if_amp \
                                   --batch_size 14 \
```

Training logs get saved in the folder batch_scripts/run_logs. 
Our training logs for this project are present in that folder.
## Demo
Our demo is present in the file demo.ipynb. Ensure you have a GPU before running the notebook.

Our best checkpoints can be downloaded from [here](https://drive.google.com/drive/folders/1M4mGhf8JYSI6Bou4Mtg0nHAXZu0gsEh9?usp=sharing) 


Place the checkpoints in some folder and paste the path to the checkpoint in the notebook.

Checkpoint reference: 

- Full adaptation (Best performing, recommended) - checkpoint_peft_decoder_adapter.pth
- Half adaptation - checkpoint_peft_no_decoder_adapter.pth
- Partial baseline - half_frozen_baseline.pth
- Frozen baseline - full_frozen_baseline.pth

Place the checkpoints in some folder and paste the path to the checkpoint in the notebook.

## Acknowledgements
This project is based on [ReMamber](https://github.com/yyh-rain-song/ReMamber/tree/master), [DETRIS](https://github.com/jiaqihuang01/DETRIS/tree/main), [DMMI](https://github.com/toggle1995/RIS-DMMI). We are greatly thankful to their amazing works.