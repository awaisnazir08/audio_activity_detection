import torch
import os

# Load the original checkpoint
checkpoint = torch.load('actionformer_thumos_i3d_epoch_34_8e5ac8fd.pth', map_location='cpu')

# Create a new checkpoint with regular structure
new_checkpoint = {
    "epoch": checkpoint.get("epoch", 34),  # Keep the epoch if available
    "state_dict": checkpoint["state_dict_ema"],  # Use EMA weights as regular weights
    "optimizer": {},  # Empty optimizer state (will be initialized during training)
    "scheduler": {},  # Empty scheduler state
    "state_dict_ema": checkpoint["state_dict_ema"]  # Keep the original EMA weights
}

# Save the converted checkpoint
new_checkpoint_path = 'converted_actionformer_thumos_i3d.pth'
torch.save(new_checkpoint, new_checkpoint_path)
print(f"Converted checkpoint saved to {new_checkpoint_path}")