import torch

c = torch.load('outputs/checkpoints/best.pt', weights_only=False)

print("Output proj layers:")
for k, v in c['model_state_dict'].items():
    if 'output_proj' in k:
        print(f"  {k}: {v.shape}")

print("\nAll keys containing 'multimodal':")
for k, v in c['model_state_dict'].items():
    if 'multimodal' in k:
        print(f"  {k}: {v.shape}")
