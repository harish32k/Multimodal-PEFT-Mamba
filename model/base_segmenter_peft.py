import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPTextModel, CLIPTokenizerFast
from peft import LoraConfig, get_peft_model, TaskType
import torch
from model.utils import dice_loss, sigmoid_focal_loss

def get_text_encoder_weights(
    checkpoint_path: str = "pretrain/ReMamber_Conv.pth"
) -> dict:
    """
    Load a BaseSegmenter checkpoint and extract only the text_encoder weights.

    Args:
        checkpoint_path (str): Path to the checkpoint file (.pth or .pt).
                               Defaults to "pretrain/ReMamber_Conv.pth".

    Returns:
        dict: A state_dict containing only the keys/values for the text_encoder,
              with the "text_encoder." prefix removed.
    """
    # Load the checkpoint (handles both plain state_dict and {"model": state_dict} formats)
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt

    # Extract text_encoder weights
    text_state = {
        k.replace("text_encoder.", ""): v
        for k, v in state_dict.items()
        if k.startswith("text_encoder.")
    }

    return text_state


class BaseSegmenter_PEFT(nn.Module):
    def __init__(self, backbone, **kwargs):
        super().__init__()
        self.backbone = backbone
        self.decoder = None

        self.tokenizer = CLIPTokenizerFast.from_pretrained(
            "openai/clip-vit-large-patch14"
        )

        # — load base CLIP text encoder —
        base_text = CLIPTextModel.from_pretrained(
            "openai/clip-vit-large-patch14"
        )

        text_state = get_text_encoder_weights()
        base_text.load_state_dict(text_state, strict=True)

        # — configure LoRA —
        lora_cfg = LoraConfig(
            inference_mode=False,          # we’re fine‐tuning
            r=8,                           # LoRA rank
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
        )

        # — wrap with LoRA adapters (freezes base_text) —
        self.text_encoder = get_peft_model(base_text, lora_cfg)

    def forward(self, x, text, mask=None, **kwargs):
            # Tokenize input text
            encoding = self.tokenizer(
                text,
                padding='max_length',
                truncation=True,
                max_length=20,
                return_tensors='pt'
            )
            input_ids = encoding['input_ids'].to(x.device, non_blocking=True)
            attention_mask = encoding['attention_mask'].to(x.device, non_blocking=True)

            # Encode text with LoRA-adapted encoder
            #ret = self.text_encoder.base_model(
            #    input_ids=input_ids,
            #    attention_mask=attention_mask,
            #    output_hidden_states=False,
            #    return_dict=True
            #)
            ret = self.text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=False,
                return_dict=True
            )
            l_feats = ret.last_hidden_state  # (B, N_l, C)
            l_feats = l_feats.permute(0, 2, 1)  # (B, C, N_l)
            l_mask = attention_mask.unsqueeze(-1)  # (B, N_l, 1)

            # Extract optional pooled output
            pooler_out = ret.pooler_output if hasattr(ret, "pooler_output") else None

            # Pass through backbone and decoder
            features = self.backbone(x, l_feats, l_mask, pooler_out=pooler_out)
            x_c1, x_c2, x_c3, x_c4 = features
            pred = self.decoder([x_c4, x_c3, x_c2, x_c1], l_feats, l_mask)
            pred = F.interpolate(pred, x.shape[-2:], mode='bilinear', align_corners=True)

            # Compute loss during training
            if self.training:
                loss = dice_loss(pred, mask) + sigmoid_focal_loss(pred, mask, alpha=-1, gamma=0)
                return pred.detach(), mask, loss
            else:
                return pred.detach()