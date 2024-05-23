import torch
TORCH_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SUPRESS_WARNINGS = True