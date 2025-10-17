from habitat import Config
import torch

ckpt = torch.load("ckpt.iter100000.pth") # Input pre-trained the checkpoint from 3DFF model
ckpt = ckpt['state_dict']
new_ckpt = {}
for key in ckpt:
    if "net.module.feature_fields." in key:
        new_key = key[len("net.module.feature_fields."):]
        new_ckpt[new_key] = ckpt[key]
        key = new_key
    elif "net.feature_fields." in key:
        new_key = key[len("net.feature_fields."):]
        new_ckpt[new_key] = ckpt[key]
        key = new_key

    '''
    if "patch_to_instance_position_embedding" in key:
        new_key = key.replace("patch_to_instance_position_embedding","freezed_patch_to_instance_position_embedding")
        new_ckpt[new_key] = new_ckpt[key]
    if "aggregate_patch_to_instance_embedding" in key:
        new_key = key.replace("aggregate_patch_to_instance_embedding","freezed_aggregate_patch_to_instance_embedding")
        new_ckpt[new_key] = new_ckpt[key]
    if "aggregate_patch_to_instance_encoder" in key:
        new_key = key.replace("aggregate_patch_to_instance_encoder","freezed_aggregate_patch_to_instance_encoder")
        new_ckpt[new_key] = new_ckpt[key]
    '''

torch.save(new_ckpt,"dynam3d.pth") # Save the checkpoint for downstream tasks
