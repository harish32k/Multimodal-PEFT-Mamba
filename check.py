# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0" # NOTE: use this to select GPU
# import torch
# from timm.models import create_model

# import model.conv_segmenter
# import model.mamba_segmenter
# import utils as utils
# from ref_dataset import get_transform_seg


# DEVICE = torch.device("cuda")
# SEED = 42
# MODEL_NAME = "ReMamber_Mamba" # "ReMamber_Mamba" or "ReMamber_Conv"
# CKPT_PATH = "pretrain/vssm_base_0229_ckpt_epoch_237.pth"#"checkpoints/ReMamber_Mamba.pth"

# # prepare model and transforms
# demo_model, new_param = create_model(
#     MODEL_NAME,
#     img_size=480,
#     model_size="base",
# )

# checkpoint = torch.load(CKPT_PATH, map_location='cpu')
# demo_model.load_state_dict(checkpoint['model'])
# demo_model.to(DEVICE)
# demo_model.eval()
# transforms = get_transform_seg(size=(480,480), train=False)

from ref_dataset import build_dataset

config = {
    "batch_size": 2,
    "epochs": 50,
    "model": "ReMamber_Conv",
    "pretrain_path": "./pretrain",
    "input_size": 480,
    "drop": 0.0,
    "drop_path": 0.1,
    "model_ema": False,
    "model_ema_decay": 0.9999,
    "model_ema_force_cpu": False,
    "opt": "adamw",
    "opt_eps": 1e-08,
    "opt_betas": None,
    "clip_grad": None,
    "momentum": 0.9,
    "weight_decay": 0.0001,
    "sched": "cosine",
    "lr": 5e-05,
    "lr_decoder": 5e-05,
    "lr_backbone": 2.5e-05,
    "lr_vssm": 2.5e-05,
    "warmup_lr": 1e-06,
    "min_lr": 1e-06,
    "decay_epochs": 30,
    "warmup_epochs": 0,
    "cooldown_epochs": 10,
    "patience_epochs": 10,
    "decay_rate": 0.1,
    "train_mode": True,
    "finetune": "test_training/best_checkpoint.pth",
    "data_path": "./ref_dataset/data",
    "data_set": "refcoco",
    "output_dir": "./test_training",
    "device": "cuda",
    "seed": 0,
    "resume": "",
    "start_epoch": 0,
    "eval": False,
    "dist_eval": False,
    "num_workers": 8,
    "pin_mem": True,
    "distributed": True,
    "world_size": 1,
    "dist_url": "env://",
    "if_amp": True,
    "debug_mode": True,
    "local_rank": 0,
    "use_dense_aligner": True,
    "use_text_adapter": True,
    "da_layers": "1,3,5,7,9,11",
    "ta_layers": "1,3,5,7,9,11",
    "da_dim": 128,
    "ta_dim": 64,
    "rank": 0,
    "gpu": 0,
    "dist_backend": "nccl"
}

from types import SimpleNamespace

# Convert dictionary to namespace
args = config_obj = SimpleNamespace(**config)

print("build the dataset")
dataset_train = build_dataset(is_train=False, args=args, split="test")
