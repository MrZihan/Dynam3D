from copy import deepcopy
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from gym import Space
from habitat import Config
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.rl.ppo.policy import Net
import random
from vlnce_baselines.common.aux_losses import AuxLosses

from vlnce_baselines.models.encoders.resnet_encoders import (
    TorchVisionResNet50,
    VlnResnetDepthEncoder,
    CLIPEncoder,
)
from vlnce_baselines.models.policy import ILPolicy

from vlnce_baselines.waypoint_pred.TRM_net import BinaryDistPredictor_TRM
from vlnce_baselines.waypoint_pred.utils import nms
from vlnce_baselines.models.utils import (
    angle_feature_with_ele, dir_angle_feature_with_ele, angle_feature_torch, length2mask)
import math
from vlnce_baselines.models.feature_fields import Feature_Fields
from PIL import Image
import cv2
from transformers import AutoProcessor, LlavaForConditionalGeneration
from peft import LoraConfig, TaskType, get_peft_model


@baseline_registry.register_policy
class Policy_Dynam3D_VLN(ILPolicy):
    def __init__(
        self,
        observation_space: Space,
        action_space: Space,
        model_config: Config,
    ):
        super().__init__(
            Dynam3D_VLN(
                observation_space=observation_space,
                model_config=model_config,
                num_actions=action_space.n,
            ),
            action_space.n,
        )

    @classmethod
    def from_config(
        cls, config: Config, observation_space: Space, action_space: Space
    ):
        config.defrost()
        config.MODEL.TORCH_GPU_ID = config.TORCH_GPU_ID
        config.freeze()

        return cls(
            observation_space=observation_space,
            action_space=action_space,
            model_config=config.MODEL,
        )


class Dynam3D_VLN(Net):
    def __init__(
        self, observation_space: Space, model_config: Config, num_actions,
    ):
        super().__init__()

        device = (
            torch.device("cuda", model_config.TORCH_GPU_ID)
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        self.device = device
        print('\nInitalizing the Dynam3D_VLN model ...')
        self.feature_fields = Feature_Fields(batch_size=1, device=self.device)
        self.feature_fields.load_state_dict(torch.load("dynam3d.pth"),strict=True)
        self.feature_fields.eval()
        width = self.feature_fields.args.fts_dim
        self.patch_position_embedding =  nn.Sequential(
            nn.Linear(6, width*4),
            nn.LayerNorm(width*4),
            nn.GELU(),
            nn.Linear(width*4, width*4))

        self.instance_position_embedding =  nn.Sequential(
            nn.Linear(3, width),
            nn.LayerNorm(width),
            nn.GELU(),
            nn.Linear(width, width))

        self.zone_position_embedding =  nn.Sequential(
            nn.Linear(3, width),
            nn.LayerNorm(width),
            nn.GELU(),
            nn.Linear(width, width))

        self.instance_projector = nn.Sequential(
            nn.Linear(width*2, width*4),
            nn.LayerNorm(width*4),
            nn.GELU(),
            nn.Linear(width*4, width*4))

        self.zone_projector = nn.Sequential(
            nn.Linear(width*2, width*4),
            nn.LayerNorm(width*4),
            nn.GELU(),
            nn.Linear(width*4, width*4))

        model_id = "xtuner/llava-phi-3-mini-hf"
        # create LoRA configuration object
        #lora_config = LoraConfig(
        #    task_type=TaskType.CAUSAL_LM,
        #    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        #    inference_mode=False, 
        #    r=128,
        #    lora_alpha=256, # Lora alaph
        #    lora_dropout=0.1# Dropout
        #)
        self.llava = LlavaForConditionalGeneration.from_pretrained(
                    model_id, 
                    torch_dtype=torch.bfloat16,
                    low_cpu_mem_usage=False, 
                ).to(self.device)

        self.llava.gradient_checkpointing_enable()
        #self.llava.language_model = get_peft_model(self.llava.language_model, lora_config)
        self.llava_processor = AutoProcessor.from_pretrained(model_id)

        # Init the depth encoder
        assert model_config.DEPTH_ENCODER.cnn_type in [
            "VlnResnetDepthEncoder"
        ], "DEPTH_ENCODER.cnn_type must be VlnResnetDepthEncoder"
        self.depth_encoder = VlnResnetDepthEncoder(
            observation_space,
            output_size=model_config.DEPTH_ENCODER.output_size,
            checkpoint=model_config.DEPTH_ENCODER.ddppo_checkpoint,
            backbone=model_config.DEPTH_ENCODER.backbone,
            spatial_output=model_config.spatial_output,
        )
        self.space_pool_depth = nn.Sequential(nn.AdaptiveAvgPool2d((1,1)), nn.Flatten(start_dim=2))
        self.rgb_encoder = CLIPEncoder("ViT-L/14@336px",self.device)
    
        self.pano_img_idxes = np.arange(0, 12, dtype=np.int64)        # 逆时针
        pano_angle_rad_c = (1-self.pano_img_idxes/12) * 2 * math.pi   # 对应到逆时针
        self.pano_angle_fts = angle_feature_torch(torch.from_numpy(pano_angle_rad_c))

        # Avoid DDP bug
        for p in self.feature_fields.parameters():
            p.requires_grad_(False)
        for p in self.llava.vision_tower.parameters():
            p.requires_grad_(False)
        for p in self.llava.multi_modal_projector.parameters():
            p.requires_grad_(False)

    @property  # trivial argument, just for init with habitat
    def output_size(self):
        return 1

    @property
    def is_blind(self):
        return self.rgb_encoder.is_blind or self.depth_encoder.is_blind

    @property
    def num_recurrent_layers(self):
        return 1

    def preprocess_depth(self, depth, depth_scale=(0.,10.)):
        # depth - (B, H, W, 1) torch Tensor

        min_depth = depth_scale[0]  # !!!!!!!!!!! This is the setting for R2R
        max_depth = depth_scale[1] # !!!!!!!!!!! This is the setting for R2R

        # Column-wise post-processing
        depth = depth * 1.0
        H = depth.shape[1]
        depth_max, _ = depth.max(dim=1, keepdim=True)  # (B, H, W, 1)
        depth_max = depth_max.expand(-1, H, -1, -1)
        depth[depth == 0] = depth_max[depth == 0]

        depth = min_depth * 100.0 + depth * (max_depth - min_depth) * 100.0
        depth = depth / 100.
        return depth

    def get_candidate_waypoints(self, waypoint_predictor=None, observations=None):

        batch_size = observations['depth'].shape[0]
        ''' encoding rgb/depth at all directions ----------------------------- '''
        NUM_ANGLES = 120    # 120 angles 3 degrees each
        NUM_IMGS = 12
        NUM_CLASSES = 12    # 12 distances at each sector
        depth_batch = torch.zeros_like(observations['depth']).repeat(NUM_IMGS, 1, 1, 1)

        # reverse the order of input images to clockwise
        a_count = 0
        for i, (k, v) in enumerate(observations.items()):
            if 'depth' in k:  # You might need to double check the keys order
                for bi in range(v.size(0)):
                    ra_count = (NUM_IMGS - a_count) % NUM_IMGS
                    depth_batch[ra_count + bi*NUM_IMGS] = v[bi]
                a_count += 1
        obs_view12 = {}
        obs_view12['depth'] = depth_batch
        depth_embedding = self.depth_encoder(obs_view12)  # torch.Size([bs, 128, 4, 4])

        ''' waypoint prediction ----------------------------- '''
        waypoint_heatmap_logits = waypoint_predictor(
            None, depth_embedding)

        # reverse the order of images back to counter-clockwise
            
        depth_embed_reshape = depth_embedding.reshape(
            batch_size, NUM_IMGS, 128, 4, 4)
            
        depth_feats = torch.cat((
            depth_embed_reshape[:,0:1,:], 
            torch.flip(depth_embed_reshape[:,1:,:], [1]),
        ), dim=1)
        # way_feats = torch.cat((
        #     way_feats[:,0:1,:], 
        #     torch.flip(way_feats[:,1:,:], [1]),
        # ), dim=1)

        # from heatmap to points
        batch_x_norm = torch.softmax(
            waypoint_heatmap_logits.reshape(
                batch_size, NUM_ANGLES*NUM_CLASSES,
            ), dim=1
        )
        batch_x_norm = batch_x_norm.reshape(
            batch_size, NUM_ANGLES, NUM_CLASSES,
        )
        batch_x_norm_wrap = torch.cat((
            batch_x_norm[:,-1:,:], 
            batch_x_norm, 
            batch_x_norm[:,:1,:]), 
            dim=1)
        batch_output_map = nms(
            batch_x_norm_wrap.unsqueeze(1), 
            max_predictions=5,
            sigma=(7.0,5.0))

        # predicted waypoints before sampling
        batch_output_map = batch_output_map.squeeze(1)[:,1:-1,:]
            
        depth_feats = self.space_pool_depth(depth_feats)

        # for cand
        cand_depth = []
        cand_angle_fts = []
        cand_img_idxes = []
        cand_angles = []
        cand_distances = []
        for j in range(batch_size):

            angle_idxes = batch_output_map[j].nonzero()[:, 0]
            distance_idxes = batch_output_map[j].nonzero()[:, 1]
            # for angle & distance
            angle_rad_c = angle_idxes.cpu().float()/120*2*math.pi       # 顺时针
            angle_rad_cc = 2*math.pi-angle_idxes.float()/120*2*math.pi  # 逆时针
            cand_angle_fts.append( angle_feature_torch(angle_rad_c) )
            cand_angles.append(angle_rad_cc.tolist())
            cand_distances.append( ((distance_idxes + 1)*0.25).tolist() )
            # for img idxes
            img_idxes = 12 - (angle_idxes.cpu().numpy()+5) // 10        # 逆时针
            img_idxes[img_idxes==12] = 0
            cand_img_idxes.append(img_idxes)
            # for rgb & depth
            cand_depth.append(depth_feats[j, img_idxes, ...])
            
        # for pano
        pano_depth = depth_feats                        # B x 12 x 128
        pano_angle_fts = deepcopy(self.pano_angle_fts)  # 12 x 4
        pano_img_idxes = deepcopy(self.pano_img_idxes)  # 12

        # cand_angle_fts 顺时针
        # cand_angles 逆时针
        outputs = {
            'cand_depth': cand_depth,           # [K x 128]
            'cand_angle_fts': cand_angle_fts,   # [K x 4]
            'cand_img_idxes': cand_img_idxes,   # [K]
            'cand_angles': cand_angles,         # [K]
            'cand_distances': cand_distances,   # [K]

            'pano_depth': pano_depth,           # B x 12 x 128
            'pano_angle_fts': pano_angle_fts,   # 12 x 4
            'pano_img_idxes': pano_img_idxes,   # 12 
        }
        return outputs
            
    def get_gt_text(self, target_angles, target_distances, stop_actions):
        target_angles = [round(np.degrees(angle)) for angle in target_angles]
        target_distances = [distance for distance in target_distances]
        batch_gt_text = ["" for b in range(len(target_angles))]
        angle_per_step = 15
        distance_per_step = 0.25
        max_turn_steps = 4
        for b in range(len(target_angles)):
            if stop_actions[b] == True:
                batch_gt_text[b] = "stop.<|end|>"
            else:
                turn_angle = target_angles[b]
                move_distance = target_distances[b]
                turn_steps = round(turn_angle/angle_per_step)
                if max_turn_steps <= turn_steps and turn_steps < 360 // angle_per_step:
                    if max_turn_steps <= turn_steps  and turn_steps < 180 // angle_per_step:
                        batch_gt_text[b] = "turn left "+str(round(turn_angle/angle_per_step)) + " steps," + " move "+str(round(move_distance / distance_per_step)) + " steps.<|end|>"
                        self.feature_fields.keep_target_waypoint[b] = [ (np.radians(turn_angle-(max_turn_steps*angle_per_step)) + math.pi*2) % (math.pi*2), move_distance]
                    else:
                        batch_gt_text[b] = "turn right "+str(round((360-turn_angle)/angle_per_step)) + " steps," + " move "+str(round(move_distance / distance_per_step)) + " steps.<|end|>"
                        self.feature_fields.keep_target_waypoint[b] = [ (np.radians(turn_angle+(max_turn_steps*angle_per_step)) + math.pi*2) % (math.pi*2), move_distance]
                else:
                    if turn_steps < max_turn_steps:
                        batch_gt_text[b] = "turn left "+str(round(turn_angle/angle_per_step))+" steps," + " move "+str(round(move_distance / distance_per_step)) + " steps.<|end|>"
                    else:
                        batch_gt_text[b] = "turn right "+str(round((360-turn_angle)/angle_per_step))+" steps," + " move "+str(round(move_distance / distance_per_step)) + " steps.<|end|>"
                    self.feature_fields.keep_target_waypoint[b] = None
        
            # Avoid errors of data sample
            if self.feature_fields.history_actions[b][-2][:len("turn left 4 steps")] == batch_gt_text[b][:len("turn left 4 steps")] and self.feature_fields.history_actions[b][-4][:len("turn left 4 steps")] == batch_gt_text[b][:len("turn left 4 steps")] and self.feature_fields.history_actions[b][-3][:len("turn left 4 steps")] == batch_gt_text[-1][:len("turn left 4 steps")]:
                    batch_gt_text[b] = "error.<|end|>"

        return batch_gt_text


    def forward(self, observations, instructions, agent_positions, agent_heading_angles, depth_scale=(0.,10.), gt_text=None, delete_old_features=True, num_of_views=1, is_train=False):
        
        # Preprocess the observations
        batch_size = self.feature_fields.batch_size
        depth_height = self.feature_fields.args.input_height
        depth_width = self.feature_fields.args.input_width
        layer_width = self.feature_fields.args.fts_dim
        batch_depth_fts = torch.zeros((batch_size*num_of_views,depth_height,depth_width,1))
        for b in range(batch_size):
            for i in range(num_of_views):
                batch_depth_fts[b*num_of_views+i] = torch.tensor(cv2.resize(observations['depth'][b][i].cpu().numpy(), (depth_height, depth_width),  interpolation = cv2.INTER_NEAREST)).view(depth_height, depth_width,1)

        batch_depth_fts = self.preprocess_depth(batch_depth_fts).view(batch_size,num_of_views,depth_height*depth_width).numpy()

        with torch.no_grad():
            _, batch_grid_fts = self.rgb_encoder({"rgb":observations['rgb']})
            batch_grid_fts = batch_grid_fts.view(batch_size,num_of_views,depth_height*depth_width,layer_width).cpu().numpy()
            batch_rgb = observations['rgb'].view(batch_size,num_of_views,observations['rgb'].shape[-3],observations['rgb'].shape[-2],observations['rgb'].shape[-1]).cpu().numpy()

        # Delete the old features within the camera frustum
        if delete_old_features:
            batch_depth = self.preprocess_depth(observations['depth'], depth_scale).view(batch_size,num_of_views,observations['depth'].shape[-3], observations['depth'].shape[-2])
            self.feature_fields.delete_old_features_from_camera_frustum(batch_depth, agent_positions, agent_heading_angles, num_of_views=num_of_views)

        with torch.no_grad():
            self.feature_fields.update_feature_fields(batch_depth_fts, batch_grid_fts, batch_image=batch_rgb, batch_position=agent_positions, batch_heading=agent_heading_angles, num_of_views=num_of_views)
        

        # Get the 3D environment representations
        env_fts = self.feature_fields.get_environment_features(agent_positions, agent_heading_angles)
        batch_instance_fts = env_fts["batch_instance_fts"]
        batch_instance_relative_position = env_fts["batch_instance_relative_position"]
        batch_zone_fts = env_fts["batch_zone_fts"]
        batch_zone_relative_position = env_fts["batch_zone_relative_position"]
        batch_rel_x, batch_rel_y, batch_rel_z, batch_direction, batch_scale = self.feature_fields.get_patch_3d_info(batch_depth_fts.reshape(batch_size*num_of_views,-1))

        # Training mode
        if is_train:
            patch_3d_info = torch.cat([batch_rel_x, batch_rel_y, batch_rel_z, torch.sin(batch_direction), torch.cos(batch_direction), batch_scale],dim=-1)
            patch_position_fts = self.patch_position_embedding(patch_3d_info)
            batch_instance_fts = [self.instance_projector(torch.cat([batch_instance_fts[b],self.instance_position_embedding(batch_instance_relative_position[b])],dim=-1)) for b in range(batch_size)]
            batch_zone_fts = [self.zone_projector(torch.cat([batch_zone_fts[b], self.zone_position_embedding(batch_zone_relative_position[b])],dim=-1)) for b in range(batch_size)]
            inputs = ["<|user|>\n"+"<image>"*(depth_height*depth_width*num_of_views+len(batch_instance_fts[b])+len(batch_zone_fts[b]))+"\nInstruction:\n"+instructions[b]+"\nHistory actions:\n"+"".join(self.feature_fields.history_actions[b])+"<|end|>\n<|assistant|>\nNext action:\n"+gt_text[b] for b in range(batch_size)] # The patch_size is 24x24

            inputs = self.llava_processor(text=inputs, images=observations['rgb'], return_tensors='pt', padding=True).to(self.device, torch.float16)
            inputs_embeds = self.llava.get_input_embeddings()(inputs['input_ids']).to(self.device)

            vision_feature_layer = (
                        self.llava.config.vision_feature_layer
                    )
            vision_feature_select_strategy = (
                self.llava.config.vision_feature_select_strategy
            )

            with torch.no_grad():
                patch_features = self.llava.get_image_features(
                    pixel_values=inputs['pixel_values'],
                    vision_feature_layer=vision_feature_layer,
                    vision_feature_select_strategy=vision_feature_select_strategy
                )

            patch_features = patch_features + patch_position_fts # Add the patch position features

            # Combine the patch, instance, zone features
            inputs_embeds = [torch.cat([inputs_embeds[b:b+1,:2], patch_features[b:b+1], batch_instance_fts[b].unsqueeze(0), batch_zone_fts[b].unsqueeze(0), inputs_embeds[b:b+1,(depth_height*depth_width*num_of_views+len(batch_instance_fts[b])+len(batch_zone_fts[b]))+2:]],dim=1) for b in range(batch_size)]
            inputs_embeds = torch.cat(inputs_embeds,dim=0)
            inputs['inputs_embeds'] = inputs_embeds
            inputs['labels'] = inputs['input_ids']
            inputs['use_cache'] = False
            inputs.pop('input_ids')
            inputs.pop('pixel_values')

            output = self.llava(**inputs)

            gt_labels = self.llava_processor(text=gt_text, return_tensors='pt', padding=True).to(self.device, torch.float16)
            gt_labels['input_ids'] = gt_labels['input_ids'][:,1:]
            gt_labels['attention_mask'] = gt_labels['attention_mask'][:,1:] # Remove the start token

            prompt = ["<|user|>\n"+"<image>"*(depth_height*depth_width*num_of_views+len(batch_instance_fts[b])+len(batch_zone_fts[b]))+"\nInstruction:\n"+instructions[b]+"\nHistory actions:\n"+"".join(self.feature_fields.history_actions[b])+"<|end|>\n<|assistant|>\nNext action:\n" for b in range(batch_size)] # The patch_size is 24x24
            prompt = self.llava_processor(text=prompt, return_tensors='pt', padding=True).to(self.device, torch.float16)
            gt_labels_mask_count = gt_labels['attention_mask'].sum(-1)
            prompt_mask_count = prompt['attention_mask'].sum(-1) - 1 # -1, it's is very important for autoregression

            predicted_logits_index = [ [prompt_mask_count[b].item(),prompt_mask_count[b].item()+gt_labels_mask_count[b].item()] for b in range(batch_size) ]
            output_logits = [output['logits'][b,predicted_logits_index[b][0]:predicted_logits_index[b][1]] for b in range(batch_size)]
            loss = 0.
            for b in range(batch_size):
                loss += F.cross_entropy(output_logits[b],gt_labels['input_ids'][b][:gt_labels_mask_count[b].item()])
                if "stop" not in gt_text[b] and "error" not in gt_text[b]:
                    loss += F.cross_entropy(output_logits[b][1:2],gt_labels['input_ids'][b][1:2]) # Focus on the turn left/right prediction

                predicted_text = self.llava_processor.batch_decode([output_logits[b].argmax(-1)], skip_special_tokens=False)
                self.feature_fields.history_actions[b].pop(0)
                self.feature_fields.history_actions[b].append(gt_text[b].replace('<|end|>', '\n'))
                print(predicted_text,gt_text[b])


            loss = loss / batch_size
            return loss

        # Evaluation mode
        else:
            with torch.no_grad():
                patch_3d_info = torch.cat([batch_rel_x, batch_rel_y, batch_rel_z, torch.sin(batch_direction), torch.cos(batch_direction), batch_scale],dim=-1)
                patch_position_fts = self.patch_position_embedding(patch_3d_info)
                batch_instance_fts = [self.instance_projector(torch.cat([batch_instance_fts[b],self.instance_position_embedding(batch_instance_relative_position[b])],dim=-1)) for b in range(batch_size)]
                batch_zone_fts = [self.zone_projector(torch.cat([batch_zone_fts[b], self.zone_position_embedding(batch_zone_relative_position[b])],dim=-1)) for b in range(batch_size)]
                inputs = ["<|user|>\n"+"<image>"*(depth_height*depth_width*num_of_views+len(batch_instance_fts[b])+len(batch_zone_fts[b]))+"\nInstruction:\n"+instructions[b]+"\nHistory actions:\n"+"".join(self.feature_fields.history_actions[b])+"<|end|>\n<|assistant|>\nNext action:\n" for b in range(batch_size)] # The patch_size is 24x24

                inputs = self.llava_processor(text=inputs, images=observations['rgb'], return_tensors='pt', padding=True).to(self.device, torch.float16)
                inputs_embeds = self.llava.get_input_embeddings()(inputs['input_ids']).to(self.device)

                vision_feature_layer = (
                            self.llava.config.vision_feature_layer
                        )
                vision_feature_select_strategy = (
                    self.llava.config.vision_feature_select_strategy
                )

                patch_features = self.llava.get_image_features(
                    pixel_values=inputs['pixel_values'],
                    vision_feature_layer=vision_feature_layer,
                    vision_feature_select_strategy=vision_feature_select_strategy
                )
                patch_features = patch_features + patch_position_fts # Add the patch position features

                # Combine the patch, instance, zone features
                inputs_embeds = [torch.cat([inputs_embeds[b:b+1,:2], patch_features[b:b+1], batch_instance_fts[b].unsqueeze(0), batch_zone_fts[b].unsqueeze(0), inputs_embeds[b:b+1,(depth_height*depth_width*num_of_views+len(batch_instance_fts[b])+len(batch_zone_fts[b]))+2:]],dim=1) for b in range(batch_size)]
                inputs_embeds = torch.cat(inputs_embeds,dim=0)
                inputs['inputs_embeds'] = inputs_embeds
                inputs['labels'] = inputs['input_ids']
                inputs.pop('input_ids')
                inputs.pop('pixel_values')

                generated_ids = self.llava.generate(**inputs, max_new_tokens=20, do_sample=False)
                generated_text = self.llava_processor.batch_decode(generated_ids, skip_special_tokens=False)
                generated_text = [generated_text[b][:generated_text[b].find("<|end|>")] for b in range(batch_size)]
                for b in range(batch_size):
                    self.feature_fields.history_actions[b].pop(0)
                    self.feature_fields.history_actions[b].append(generated_text[b]+'\n')
                return generated_text
            

    def convert_text_to_action(self, generated_text):
        angle_per_step = 15
        distance_per_step = 0.25
        max_turn_steps = 4
        batch_actions = []
        for b in range(len(generated_text)): # Batch
            angle = distance = 0.
            if "stop" in generated_text[b] or "error" in generated_text[b]:
                batch_actions.append(-100)
                continue
            if "left" in generated_text[b]:
                start = generated_text[b].find("left") + len("left")
                end = generated_text[b].find("steps,")
                if end == -1:
                    batch_actions.append(-100) 
                    continue
                angle = math.radians(min(max_turn_steps,int(generated_text[b][start:end]))*angle_per_step)

            elif "right" in generated_text[b]:
                start = generated_text[b].find("right") + len("right")
                end = generated_text[b].find("steps,")
                if end == -1:
                    batch_actions.append(-100) 
                    continue
                angle = math.pi*2. - math.radians(min(max_turn_steps,int(generated_text[b][start:end]))*angle_per_step)

            if "move" in generated_text[b] and int(generated_text[b][start:end]) < max_turn_steps:
                start = generated_text[b].find("move") + len("move")
                end = generated_text[b].find("steps.")
                if end == -1:
                    distance = 0.
                distance = int(generated_text[b][start:end]) * distance_per_step

            batch_actions.append((angle,distance))
        return batch_actions