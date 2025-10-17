import os
import sys
import random
import warnings
from collections import defaultdict
from typing import Dict, List
import jsonlines
import numpy as np
import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import ReduceOp
import gc
import tqdm
from gym import Space
from habitat import Config, logger
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.environments import get_env_class
from habitat_baselines.common.obs_transformers import (
    apply_obs_transforms_batch,
    apply_obs_transforms_obs_space,
    get_active_obs_transforms,
)
from habitat_baselines.common.tensorboard_utils import TensorboardWriter
from habitat_baselines.utils.common import batch_obs

from src_3dff.common.base_il_trainer import BaseVLNCETrainer
from src_3dff.common.env_utils import construct_envs, construct_envs_for_rl, is_slurm_batch_job
from src_3dff.common.utils import extract_instruction_tokens
from src_3dff.utils import reduce_loss
from src_3dff.models.utils import get_angle_fts


from .utils import get_camera_orientations12
from .utils import (
    length2mask, dir_angle_feature_with_ele,
)
from src_3dff.common.utils import dis_to_con, gather_list_and_concat
from habitat_extensions.measures import NDTW, StepsTaken

import torch.distributed as distr
import gzip
import json
from copy import deepcopy
from torch.cuda.amp import autocast, GradScaler
from src_3dff.common.ops import pad_tensors_wgrad, gen_seq_masks
from torch.nn.utils.rnn import pad_sequence
import cv2
from PIL import Image
import clip
import open3d as o3d
import re
import matplotlib.pyplot as plt
import matplotlib

simulator_episodes = 0


@baseline_registry.register_trainer(name="SS-ETP")
class RLTrainer(BaseVLNCETrainer):
    def __init__(self, config=None):
        super().__init__(config)
        self.max_len = int(config.IL.max_traj_len) #  * 0.97 transfered gt path got 0.96 spl
        self.config = config

        # load the category embeddings
        category_data = torch.load("data/SceneVerse/category_embeddings.pth")
        self.category_dict, self.category_embeddings = category_data[0], category_data[1]
        self.category_embeddings = self.category_embeddings / torch.linalg.norm(self.category_embeddings, dim=-1, keepdim=True)

        # Load the data of hm3d
        self.hm3d_language_annotations = json.load(open("data/SceneVerse/HM3D/annotations/3dff_hm3d_annotations.json","r"))
        
        pcd_with_global_alignment_list = os.listdir("data/scene_datasets/hm3d/hm3d-train-semantic-annots-v0.2") # use the preprocessed pcd with labels, not of sceneverse
        self.hm3d_pcd_with_global_alignment = {}
        for file_name in pcd_with_global_alignment_list:
            if file_name[6:17] in self.hm3d_pcd_with_global_alignment:
                self.hm3d_pcd_with_global_alignment[file_name[6:17]].append("data/scene_datasets/hm3d/hm3d-train-semantic-annots-v0.2/"+file_name+"/"+file_name[6:17]+".semantic.pth")
            else:
                self.hm3d_pcd_with_global_alignment[file_name[6:17]] = ["data/scene_datasets/hm3d/hm3d-train-semantic-annots-v0.2/"+file_name+"/"+file_name[6:17]+".semantic.pth"]


        # Load the data of mp3d
        pcd_with_global_alignment_list = os.listdir("data/scene_datasets/mp3d/") # use the preprocessed pcd with labels, not of sceneverse
        #self.hm3d_pcd_with_global_alignment = {}
        for file_name in pcd_with_global_alignment_list:
            if file_name in self.hm3d_pcd_with_global_alignment:
                self.hm3d_pcd_with_global_alignment[file_name].append("data/scene_datasets/mp3d/"+file_name+"/"+file_name +"_semantic.pth")
            else:
                self.hm3d_pcd_with_global_alignment[file_name] = ["data/scene_datasets/mp3d/"+file_name+"/"+file_name +"_semantic.pth"]


        # Load the data of ScanNet
        self.scannet_3d_scenes = {}
        path = 'data/ScanNet/scannet_train_images/frames_square/'
        scenes = json.load(open('data/ScanNet/scannetv2_train_sort.json','r')) # only load train split of scannet
        for scene_id in scenes:    
            self.scannet_3d_scenes[scene_id] = path+scene_id

        instance_id_to_label_list = os.listdir("data/SceneVerse/ScanNet/scan_data/instance_id_to_label")
        self.ScanNet_instance_id_to_label = {}
        for file_name in instance_id_to_label_list:
            if file_name[:12] not in self.scannet_3d_scenes: # only load train split of scannet
                continue
            if file_name[:12] in self.ScanNet_instance_id_to_label:
                self.ScanNet_instance_id_to_label[file_name[:12]].append("data/SceneVerse/ScanNet/scan_data/instance_id_to_label/"+file_name)
            else:
                self.ScanNet_instance_id_to_label[file_name[:12]] = ["data/SceneVerse/ScanNet/scan_data/instance_id_to_label/"+file_name]

        pcd_with_global_alignment_list = os.listdir("data/SceneVerse/ScanNet/scan_data/pcd_with_global_alignment")
        self.ScanNet_pcd_with_global_alignment = {}
        for file_name in pcd_with_global_alignment_list:
            if file_name[:12] not in self.scannet_3d_scenes: # only load train split of scannet
                continue
            if file_name[:12] in self.ScanNet_pcd_with_global_alignment:
                self.ScanNet_pcd_with_global_alignment[file_name[:12]].append("data/SceneVerse/ScanNet/scan_data/pcd_with_global_alignment/"+file_name)
            else:
                self.ScanNet_pcd_with_global_alignment[file_name[:12]] = ["data/SceneVerse/ScanNet/scan_data/pcd_with_global_alignment/"+file_name]

        self.ScanNet_language_annotations = json.load(open("data/SceneVerse/ScanNet/annotations/3dff_scannet_annotations.json","r"))
        self.scannet_align_matrix = json.load(open("data/ScanNet/scannet_align_matrix.json","r"))


        # Load the data of 3RScan
        self.rscan_scenes = {}
        path = 'data/SceneVerse/3RScan/scan_data/instance_id_to_label/'
        scenes = os.listdir(path)
        for scene_id in scenes:    
            self.rscan_scenes[scene_id[:-4]] = "data/3RScan/"+scene_id[:-4]

        self.rscan_instance_id_to_label = {}
        for scene_id in self.rscan_scenes:
            if scene_id in self.rscan_instance_id_to_label:
                self.rscan_instance_id_to_label[scene_id].append("data/SceneVerse/3RScan/scan_data/instance_id_to_label/"+scene_id+'.pth')
            else:
                self.rscan_instance_id_to_label[scene_id] = ["data/SceneVerse/3RScan/scan_data/instance_id_to_label/"+scene_id+'.pth']

        pcd_with_global_alignment_list = os.listdir("data/SceneVerse/3RScan/scan_data/pcd_with_global_alignment")
        self.rscan_pcd_with_global_alignment = {}
        for scene_id in self.rscan_scenes:
            if scene_id in self.rscan_pcd_with_global_alignment:
                self.rscan_pcd_with_global_alignment[scene_id].append("data/SceneVerse/3RScan/scan_data/pcd_with_global_alignment/"+scene_id+'.pth')
            else:
                self.rscan_pcd_with_global_alignment[scene_id] = ["data/SceneVerse/3RScan/scan_data/pcd_with_global_alignment/"+scene_id+'.pth']

        self.rscan_language_annotations = json.load(open("data/SceneVerse/3RScan/annotations/3dff_3rscan_annotations.json","r"))



        # Load the data of ARKit
        self.arkit_scenes = {}
        path = 'data/ARKitScenes/3dod/Training/'
        scenes = os.listdir(path)
        for scene_id in scenes:    
            self.arkit_scenes[scene_id] = "data/ARKitScenes/3dod/Training/"+scene_id

        self.arkit_instance_id_to_label = {}
        for scene_id in self.arkit_scenes:
            if scene_id in self.arkit_instance_id_to_label:
                self.arkit_instance_id_to_label[scene_id].append("data/SceneVerse/ARKitScenes/scan_data/instance_id_to_label/"+scene_id+'.pth')
            else:
                self.arkit_instance_id_to_label[scene_id] = ["data/SceneVerse/ARKitScenes/scan_data/instance_id_to_label/"+scene_id+'.pth']

        pcd_with_global_alignment_list = os.listdir("data/SceneVerse/ARKitScenes/scan_data/pcd_with_global_alignment")
        self.arkit_pcd_with_global_alignment = {}
        for scene_id in self.arkit_scenes:
            if scene_id in self.arkit_pcd_with_global_alignment:
                self.arkit_pcd_with_global_alignment[scene_id].append("data/SceneVerse/ARKitScenes/scan_data/pcd_with_global_alignment/"+scene_id+'.pth')
            else:
                self.arkit_pcd_with_global_alignment[scene_id] = ["data/SceneVerse/ARKitScenes/scan_data/pcd_with_global_alignment/"+scene_id+'.pth']

        self.arkit_language_annotations = json.load(open("data/SceneVerse/ARKitScenes/annotations/3dff_arkit_annotations.json","r"))


        # Load the data of Structured3D
        self.structure_3d_scenes = {}
        for i in range(3500):
            if (1600 <= i and i <= 1799) or i in [1155,1816,1913,2034,3499]: # data was lost, see https://github.com/bertjiazheng/Structured3D/issues/30
                continue
            scene_path = 'data/Structured3D/scene_'+str(i).rjust(5, "0")+'/2D_rendering/' 
            self.structure_3d_scenes['scene_'+str(i).rjust(5, "0")] = scene_path

        '''
        instance_id_to_label_list = os.listdir("data/SceneVerse/Structured3D/scan_data/instance_id_to_label")
        self.Structured3D_instance_id_to_label = {}
        for file_name in instance_id_to_label_list:
            i = int(file_name[6:11])
            if (1600 <= i and i <= 1799) or i in [1155,1816,1913,2034,3499]: # data was lost, see https://github.com/bertjiazheng/Structured3D/issues/30
                continue
            if file_name[:11] in self.Structured3D_instance_id_to_label:
                self.Structured3D_instance_id_to_label[file_name[:11]].append("data/SceneVerse/Structured3D/scan_data/instance_id_to_label/"+file_name)
            else:
                self.Structured3D_instance_id_to_label[file_name[:11]] = ["data/SceneVerse/Structured3D/scan_data/instance_id_to_label/"+file_name]

        pcd_with_global_alignment_list = os.listdir("data/SceneVerse/Structured3D/scan_data/pcd_with_global_alignment")
        self.Structured3D_pcd_with_global_alignment = {}
        for file_name in pcd_with_global_alignment_list:
            i = int(file_name[6:11])
            if (1600 <= i and i <= 1799) or i in [1155,1816,1913,2034,3499]: # data was lost, see https://github.com/bertjiazheng/Structured3D/issues/30
                continue
            if file_name[:11] in self.Structured3D_pcd_with_global_alignment:
                self.Structured3D_pcd_with_global_alignment[file_name[:11]].append("data/SceneVerse/Structured3D/scan_data/pcd_with_global_alignment/"+file_name)
            else:
                self.Structured3D_pcd_with_global_alignment[file_name[:11]] = ["data/SceneVerse/Structured3D/scan_data/pcd_with_global_alignment/"+file_name]

        self.Structured3D_language_annotations = json.load(open("data/SceneVerse/Structured3D/annotations/3dff_structured3d_annotations.json","r"))
        '''


    def _make_dirs(self):
        if self.config.local_rank == 0:
            self._make_ckpt_dir()
            if self.config.EVAL.SAVE_RESULTS:
                self._make_results_dir()

    def save_checkpoint(self, iteration: int):
        torch.save(
            obj={
                "state_dict": self.policy.state_dict(),
                "config": self.config,
                "optim_state": self.optimizer.state_dict(),
                "iteration": iteration,
            },
            f=os.path.join(self.config.CHECKPOINT_FOLDER, f"ckpt.iter{iteration}.pth"),
        )

    def _set_config(self):
        self.split = self.config.TASK_CONFIG.DATASET.SPLIT
        self.config.defrost()
        self.config.TASK_CONFIG.TASK.NDTW.SPLIT = self.split
        self.config.TASK_CONFIG.TASK.SDTW.SPLIT = self.split
        self.config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_STEPS = -1
        self.config.SIMULATOR_GPU_IDS = self.config.SIMULATOR_GPU_IDS[self.config.local_rank]
        self.config.use_pbar = not is_slurm_batch_job()
        ''' if choosing image '''
        resize_config = self.config.RL.POLICY.OBS_TRANSFORMS.RESIZER_PER_SENSOR.SIZES
        crop_config = self.config.RL.POLICY.OBS_TRANSFORMS.CENTER_CROPPER_PER_SENSOR.SENSOR_CROPS
        task_config = self.config.TASK_CONFIG
        camera_orientations = get_camera_orientations12()
        for sensor_type in ["RGB", "DEPTH"]:
            resizer_size = dict(resize_config)[sensor_type.lower()]
            cropper_size = dict(crop_config)[sensor_type.lower()]
            sensor = getattr(task_config.SIMULATOR, f"{sensor_type}_SENSOR")
            for action, orient in camera_orientations.items():
                camera_template = f"{sensor_type}_{action}"
                camera_config = deepcopy(sensor)
                camera_config.ORIENTATION = camera_orientations[action]
                camera_config.UUID = camera_template.lower()
                setattr(task_config.SIMULATOR, camera_template, camera_config)
                task_config.SIMULATOR.AGENT_0.SENSORS.append(camera_template)
                resize_config.append((camera_template.lower(), resizer_size))
                crop_config.append((camera_template.lower(), cropper_size))
        self.config.RL.POLICY.OBS_TRANSFORMS.RESIZER_PER_SENSOR.SIZES = resize_config
        self.config.RL.POLICY.OBS_TRANSFORMS.CENTER_CROPPER_PER_SENSOR.SENSOR_CROPS = crop_config
        self.config.TASK_CONFIG = task_config
        self.config.SENSORS = task_config.SIMULATOR.AGENT_0.SENSORS
        if self.config.VIDEO_OPTION:
            self.config.TASK_CONFIG.TASK.MEASUREMENTS.append("TOP_DOWN_MAP_VLNCE")
            self.config.TASK_CONFIG.TASK.MEASUREMENTS.append("DISTANCE_TO_GOAL")
            self.config.TASK_CONFIG.TASK.MEASUREMENTS.append("SUCCESS")
            self.config.TASK_CONFIG.TASK.MEASUREMENTS.append("SPL")
            os.makedirs(self.config.VIDEO_DIR, exist_ok=True)
            shift = 0.
            orient_dict = {
                'Back': [0, math.pi + shift, 0],            # Back
                'Down': [-math.pi / 2, 0 + shift, 0],       # Down
                'Front':[0, 0 + shift, 0],                  # Front
                'Right':[0, math.pi / 2 + shift, 0],        # Right
                'Left': [0, 3 / 2 * math.pi + shift, 0],    # Left
                'Up':   [math.pi / 2, 0 + shift, 0],        # Up
            }
            sensor_uuids = []
            #H = 224
            for sensor_type in ["RGB"]:
                sensor = getattr(self.config.TASK_CONFIG.SIMULATOR, f"{sensor_type}_SENSOR")
                for camera_id, orient in orient_dict.items():
                    camera_template = f"{sensor_type}{camera_id}"
                    camera_config = deepcopy(sensor)
                    #camera_config.WIDTH = H
                    #camera_config.HEIGHT = H
                    camera_config.ORIENTATION = orient
                    camera_config.UUID = camera_template.lower()
                    #camera_config.HFOV = 90
                    sensor_uuids.append(camera_config.UUID)
                    setattr(self.config.TASK_CONFIG.SIMULATOR, camera_template, camera_config)
                    self.config.TASK_CONFIG.SIMULATOR.AGENT_0.SENSORS.append(camera_template)
        self.config.freeze()

        self.world_size = self.config.GPU_NUMBERS
        self.local_rank = self.config.local_rank
        self.batch_size = self.config.IL.batch_size

        if self.world_size > 1:
            distr.init_process_group(backend='nccl', init_method='env://')
            self.device = self.config.TORCH_GPU_IDS[self.local_rank]
            self.config.defrost()
            self.config.TORCH_GPU_ID = self.config.TORCH_GPU_IDS[self.local_rank]
            self.config.freeze()
            torch.cuda.set_device(self.device)

    def _init_envs(self):
        # for DDP to load different data
        self.config.defrost()
        self.config.TASK_CONFIG.SEED = self.config.TASK_CONFIG.SEED + self.local_rank
        self.config.freeze()

        self.envs = construct_envs(
            self.config, 
            get_env_class(self.config.ENV_NAME),
            auto_reset_done=False
        )
        env_num = self.envs.num_envs

        self.batch_size = env_num # !!!!!!!!!!!!

        dataset_len = sum(self.envs.number_of_episodes)
        logger.info(f'LOCAL RANK: {self.local_rank}, ENV NUM: {env_num}, DATASET LEN: {dataset_len}')
        observation_space = self.envs.observation_spaces[0]
        action_space = self.envs.action_spaces[0]
        self.obs_transforms = get_active_obs_transforms(self.config)
        observation_space = apply_obs_transforms_obs_space(
            observation_space, self.obs_transforms
        )

        return observation_space, action_space

    def _initialize_policy(
        self,
        config: Config,
        load_from_ckpt: bool,
        observation_space: Space,
        action_space: Space,
    ):
        start_iter = 0
        policy = baseline_registry.get_policy(self.config.MODEL.policy_name)
        self.policy = policy.from_config(
            config=config,
            observation_space=observation_space,
            action_space=action_space,
        )
        ''' initialize the waypoint predictor here '''
        from src_3dff.waypoint_pred.TRM_net import BinaryDistPredictor_TRM
        self.waypoint_predictor = BinaryDistPredictor_TRM(device=self.device)
        cwp_fn = 'data/wp_pred/check_cwp_bestdist_hfov79' if self.config.MODEL.task_type == 'rxr' else 'data/wp_pred/check_cwp_bestdist_hfov90'
        self.waypoint_predictor.load_state_dict(torch.load(cwp_fn, map_location = torch.device('cpu'))['predictor']['state_dict'],strict=False)
        for param in self.waypoint_predictor.parameters():
            param.requires_grad_(False)

        self.policy.to(self.device)
        self.waypoint_predictor.to(self.device)

        if self.config.GPU_NUMBERS > 1:
            print('Using', self.config.GPU_NUMBERS,'GPU!')
            # find_unused_parameters=False fix ddp bug
            self.policy.net = DDP(self.policy.net.to(self.device), device_ids=[self.device],
                output_device=self.device, find_unused_parameters=False, broadcast_buffers=False)
        self.optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.policy.parameters()), lr=self.config.IL.lr, eps=1e-5)

        if load_from_ckpt:
            if config.IL.is_requeue:
                import glob
                ckpt_list = list(filter(os.path.isfile, glob.glob(config.CHECKPOINT_FOLDER + "/*")) )
                ckpt_list.sort(key=os.path.getmtime)
                ckpt_path = ckpt_list[-1]
            else:
                ckpt_path = config.IL.ckpt_to_load
            ckpt_dict = self.load_checkpoint(ckpt_path, map_location="cpu")
            start_iter = ckpt_dict["iteration"]

            if 'module' in list(ckpt_dict['state_dict'].keys())[0] and self.config.GPU_NUMBERS == 1:
                self.policy.net = torch.nn.DataParallel(self.policy.net.to(self.device),
                    device_ids=[self.device], output_device=self.device)
                self.policy.load_state_dict(ckpt_dict["state_dict"],strict=False)
                self.policy.net = self.policy.net.module
                self.waypoint_predictor = torch.nn.DataParallel(self.waypoint_predictor.to(self.device),
                    device_ids=[self.device], output_device=self.device)
            else:
                self.policy.load_state_dict(ckpt_dict["state_dict"],strict=False)
            if config.IL.is_requeue:
                try:
                    self.optimizer.load_state_dict(ckpt_dict["optim_state"])
                except:
                    print("Optim_state is not loaded")

            logger.info(f"Loaded weights from checkpoint: {ckpt_path}, iteration: {start_iter}")

        params = sum(param.numel() for param in self.policy.parameters())
        params_t = sum(
            p.numel() for p in self.policy.parameters() if p.requires_grad
        )
        logger.info(f"Agent parameters: {params/1e6:.2f} MB. Trainable: {params_t/1e6:.2f} MB.")
        logger.info("Finished setting up policy.")

        self.config.defrost()
        self.config.TASK_CONFIG.SEED = self.config.TASK_CONFIG.SEED + start_iter
        self.config.freeze()
        random.seed(self.config.TASK_CONFIG.SEED)
        np.random.seed(self.config.TASK_CONFIG.SEED)
        torch.manual_seed(self.config.TASK_CONFIG.SEED)
        return start_iter


    def _teacher_action(self, batch_angles, batch_distances):
        cand_dists_to_goal = [[] for _ in range(len(batch_angles))]
        oracle_cand_idx = []
        for j in range(len(batch_angles)):
            for k in range(len(batch_angles[j])):
                angle_k = batch_angles[j][k]
                forward_k = batch_distances[j][k]
                dist_k = self.envs.call_at(j, "cand_line_dist_to_goal", {"angle": angle_k, "forward": forward_k})
                cand_dists_to_goal[j].append(dist_k)
            curr_dist_to_goal = self.envs.call_at(j, "current_line_dist_to_goal")
        # if within target range (which def as 3.0)
            if curr_dist_to_goal < 1.5:
                oracle_cand_idx.append(-1)
            else:
                oracle_cand_idx.append(np.argmin(cand_dists_to_goal[j]))

        return oracle_cand_idx


    @staticmethod
    def _pause_envs(envs, batch, envs_to_pause):
        if len(envs_to_pause) > 0:
            state_index = list(range(envs.num_envs))
            for idx in reversed(envs_to_pause):
                state_index.pop(idx)
                envs.pause_at(idx)
            
            for k, v in batch.items():
                batch[k] = v[state_index]

        return envs, batch

    def train(self):
        self._set_config()
        if self.config.MODEL.task_type == 'rxr':
            self.gt_data = {}
            for role in self.config.TASK_CONFIG.DATASET.ROLES:
                with gzip.open(
                    self.config.TASK_CONFIG.TASK.NDTW.GT_PATH.format(
                        split=self.split, role=role
                    ), "rt") as f:
                    self.gt_data.update(json.load(f))

        observation_space, action_space = self._init_envs()
        start_iter = self._initialize_policy(
            self.config,
            self.config.IL.load_from_ckpt,
            observation_space=observation_space,
            action_space=action_space,
        )

        total_iter = self.config.IL.iters
        log_every  = self.config.IL.log_every
        writer     = TensorboardWriter(self.config.TENSORBOARD_DIR if self.local_rank < 1 else None)

        logger.info('Training Starts... GOOD LUCK!')
        for idx in range(start_iter, total_iter, log_every):
            interval = min(log_every, max(total_iter-idx, 0))
            cur_iter = idx + interval
    
            logs = self._train_interval(interval, self.config.IL.ml_weight)

            if self.local_rank < 1:
                loss_str = f'iter {cur_iter}: '
                for k, v in logs.items():
                    logs[k] = np.mean(v)
                    loss_str += f'{k}: {logs[k]:.3f}, '
                    writer.add_scalar(f'loss/{k}', logs[k], cur_iter)
                logger.info(loss_str)
                self.save_checkpoint(cur_iter)

        
    def _train_interval(self, interval, ml_weight):
        self.policy.train()
        if self.world_size > 1:
            self.policy.net.module.rgb_encoder.eval()
            self.policy.net.module.depth_encoder.eval()
        else:
            self.policy.net.rgb_encoder.eval()
            self.policy.net.depth_encoder.eval()

        self.waypoint_predictor.eval()
        self.category_embeddings = self.category_embeddings.to(self.device)

        if self.local_rank < 1:
            pbar = tqdm.trange(interval, leave=False, dynamic_ncols=True)
        else:
            pbar = range(interval)
        self.logs = defaultdict(list)

        for idx in pbar:
            self.optimizer.zero_grad()
            self.loss = 0.

            with autocast():
                self.rollout('train', ml_weight)
       
            if self.loss != 0.:
                loss_value = self.loss.detach().clone()
                if self.world_size > 1: # sync the loss value for all gpus
                    distr.all_reduce(loss_value, op=ReduceOp.SUM)
                if torch.any(torch.isnan(loss_value)):
                    self.optimizer.zero_grad()
                    print("Backward error, skip...")
                else:
                    self.loss.backward()
                    for parms in self.policy.net.parameters():
                        if parms.grad != None and torch.any(torch.isnan(parms.grad)):
                            parms.grad[torch.isnan(parms.grad)] = 0

                    torch.nn.utils.clip_grad_value_(self.policy.net.parameters(), clip_value=10.)

                    self.optimizer.step()
            
            gc.collect()
            torch.cuda.empty_cache()
            if self.local_rank < 1:
                pbar.set_postfix({'iter': f'{idx+1}/{interval}'})
            
        return deepcopy(self.logs)
           

    @torch.no_grad()
    def _eval_checkpoint(
        self,
        checkpoint_path: str,
        writer: TensorboardWriter,
        checkpoint_index: int = 0,
    ):
        
        if self.local_rank < 1:
            logger.info(f"checkpoint_path: {checkpoint_path}")
        self.config.defrost()
        self.config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.SHUFFLE = True
        self.config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_STEPS = -1
        self.config.IL.ckpt_to_load = checkpoint_path
        if self.config.VIDEO_OPTION:
            self.config.TASK_CONFIG.TASK.MEASUREMENTS.append("TOP_DOWN_MAP_VLNCE")
            self.config.TASK_CONFIG.TASK.MEASUREMENTS.append("DISTANCE_TO_GOAL")
            self.config.TASK_CONFIG.TASK.MEASUREMENTS.append("SUCCESS")
            self.config.TASK_CONFIG.TASK.MEASUREMENTS.append("SPL")
            os.makedirs(self.config.VIDEO_DIR, exist_ok=True)
            shift = 0.
            orient_dict = {
                'Back': [0, math.pi + shift, 0],            # Back
                'Down': [-math.pi / 2, 0 + shift, 0],       # Down
                'Front':[0, 0 + shift, 0],                  # Front
                'Right':[0, math.pi / 2 + shift, 0],        # Right
                'Left': [0, 3 / 2 * math.pi + shift, 0],    # Left
                'Up':   [math.pi / 2, 0 + shift, 0],        # Up
            }
            sensor_uuids = []
            #H = 224
            for sensor_type in ["RGB"]:
                sensor = getattr(self.config.TASK_CONFIG.SIMULATOR, f"{sensor_type}_SENSOR")
                for camera_id, orient in orient_dict.items():
                    camera_template = f"{sensor_type}{camera_id}"
                    camera_config = deepcopy(sensor)
                    #camera_config.WIDTH = H
                    #camera_config.HEIGHT = H
                    camera_config.ORIENTATION = orient
                    camera_config.UUID = camera_template.lower()
                    #camera_config.HFOV = 90
                    sensor_uuids.append(camera_config.UUID)
                    setattr(self.config.TASK_CONFIG.SIMULATOR, camera_template, camera_config)
                    self.config.TASK_CONFIG.SIMULATOR.AGENT_0.SENSORS.append(camera_template)
        self.config.freeze()

        if self.config.EVAL.SAVE_RESULTS:
            fname = os.path.join(
                self.config.RESULTS_DIR,
                f"stats_ckpt_{checkpoint_index}_{self.config.TASK_CONFIG.DATASET.SPLIT}.json",
            )
            if os.path.exists(fname) and not os.path.isfile(self.config.EVAL.CKPT_PATH_DIR):
                print("skipping -- evaluation exists.")
                return
        self.envs = construct_envs(
            self.config, 
            get_env_class(self.config.ENV_NAME),
            episodes_allowed=self.traj[::5] if self.config.EVAL.fast_eval else self.traj,
            auto_reset_done=False, # unseen: 11006 
        )
        dataset_length = sum(self.envs.number_of_episodes)
        print('local rank:', self.local_rank, '|', 'dataset length:', dataset_length)

        obs_transforms = get_active_obs_transforms(self.config)
        observation_space = apply_obs_transforms_obs_space(
            self.envs.observation_spaces[0], obs_transforms
        )
        self._initialize_policy(
            self.config,
            load_from_ckpt=True,
            observation_space=observation_space,
            action_space=self.envs.action_spaces[0],
        )
        self.policy.eval()
        self.waypoint_predictor.eval()

        if self.config.EVAL.EPISODE_COUNT == -1:
            eps_to_eval = sum(self.envs.number_of_episodes)
        else:
            eps_to_eval = min(self.config.EVAL.EPISODE_COUNT, sum(self.envs.number_of_episodes))
        self.stat_eps = {}
        self.pbar = tqdm.tqdm(total=eps_to_eval) if self.config.use_pbar else None

        while len(self.stat_eps) < eps_to_eval:
            self.rollout('eval')
        self.envs.close()

        if self.world_size > 1:
            distr.barrier()


    def focal_loss(self, inputs, targets, focal_rate=0.1):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        focal_num = max(int(focal_rate * targets.shape[-1]),1)
        focal_loss = ce_loss.mean() + torch.topk(ce_loss.view(-1),focal_num)[0].mean()
        return focal_loss

    def sim_matrix_cross_entropy(self, sim_matrix):
        logpt = F.log_softmax(sim_matrix, dim=-1)
        logpt = torch.diag(logpt)
        nce_loss = -logpt
        sim_loss = nce_loss.mean()
        return sim_loss

    def contrastive_loss(self, fts_1, fts_2, logit_scale=10.):
        sim_matrix = logit_scale * torch.matmul(fts_1, fts_2.t())
        sim_loss1 = self.sim_matrix_cross_entropy(sim_matrix)
        sim_loss2 = self.sim_matrix_cross_entropy(sim_matrix.T)
        sim_loss = (sim_loss1 + sim_loss2)
        return sim_loss


    def fine_grained_contrastive_loss(self, batch_fts_1, batch_fts_2, logit_scale=10.):
        batch_fts_1 = batch_fts_1 / (torch.linalg.norm(batch_fts_1, dim=-1, keepdim=True) + 1e-5)
        batch_sim_score = []
        for batch_id in range(len(batch_fts_2)):
            fts_2 = batch_fts_2[batch_id]
            fts_2 = fts_2[torch.abs(fts_2).sum(-1) != 0]
            fts_2_length = fts_2.shape[0]
            if fts_2_length == 0:
                batch_sim_score.append(torch.zeros((1,len(batch_fts_1)),device=batch_fts_1.device,dtype=batch_fts_1.dtype))
                continue
            fts_2 = fts_2 / (torch.linalg.norm(fts_2, dim=-1, keepdim=True) + 1e-5)
            sim_matrix = logit_scale * torch.matmul(batch_fts_1, fts_2.t())
            sim_matrix = sim_matrix.view(batch_fts_1.shape[0],-1)
            sim_score =  torch.topk(sim_matrix,fts_2_length, dim=-1)[0].mean(dim=-1).view(1,-1)
            batch_sim_score.append(sim_score)

        batch_sim_score = torch.cat(batch_sim_score,dim=0).to(torch.float32)
        sim_loss1 = self.sim_matrix_cross_entropy(batch_sim_score)
        sim_loss2 = self.sim_matrix_cross_entropy(batch_sim_score.T)
        sim_loss = (sim_loss1 + sim_loss2)
        return sim_loss


    def parse_camera_info(self, camera_info, height, width):
        """ extract intrinsic and extrinsic matrix
        """
        lookat = camera_info[3:6] / np.linalg.norm(camera_info[3:6])
        up = camera_info[6:9] / np.linalg.norm(camera_info[6:9])

        W = lookat
        U = np.cross(W, up)
        V = np.cross(W, U)

        rot = np.vstack((U, V, W))
        trans = camera_info[:3] / 1000.

        xfov = camera_info[9]
        yfov = camera_info[10]

        K = np.diag([1, 1, 1])

        K[0, 2] = width / 2
        K[1, 2] = height / 2

        K[0, 0] = K[0, 2] / np.tan(xfov)
        K[1, 1] = K[1, 2] / np.tan(yfov)

        return rot, trans, K
    

    # from ARKitScene code base
    def convert_angle_axis_to_matrix3(self,angle_axis):
        """Return a Matrix3 for the angle axis.
        Arguments:
            angle_axis {Point3} -- a rotation in angle axis form.
        """
        matrix, jacobian = cv2.Rodrigues(angle_axis)
        return matrix

    # from ARKit Scene, some with modifications
    def TrajStringToMatrix(self, traj_str):
        """ convert traj_str into translation and rotation matrices
        Args:
            traj_str: A space-delimited file where each line represents a camera position at a particular timestamp.
            The file has seven columns:
            * Column 1: timestamp
            * Columns 2-4: rotation (axis-angle representation in radians)
            * Columns 5-7: translation (usually in meters)

        Returns:
            Rt: rotation matrix, translation matrix
        """

        tokens = traj_str.split()
        assert len(tokens) == 7
        # Rotation in angle axis
        angle_axis = [float(tokens[1]), float(tokens[2]), float(tokens[3])]
        r_w_to_p = self.convert_angle_axis_to_matrix3(np.asarray(angle_axis))
        # Translation
        t_w_to_p = np.asarray([float(tokens[4]), float(tokens[5]), float(tokens[6])])
        extrinsics = np.eye(4, 4)
        extrinsics[:3, :3] = r_w_to_p
        extrinsics[:3, -1] = t_w_to_p
        Rt = np.linalg.inv(extrinsics)
        return Rt


    def st2_camera_intrinsics(self, filename):
        w, h, fx, fy, hw, hh = np.loadtxt(filename)
        return np.asarray([[fx, 0, hw], [0, fy, hh], [0, 0, 1]])


    def run_on_hm3d(self, mode):
        global simulator_episodes
        simulator_episodes += 1
        if simulator_episodes % 50 == 0:
            self.envs.close()
            self.envs = construct_envs(
                self.config, 
                get_env_class(self.config.ENV_NAME),
                auto_reset_done=False
            )

        loss = 0.
        total_actions = 0.
        self.envs.resume_all()
        observations = self.envs.reset()
        instr_max_len = self.config.IL.max_text_len # r2r 80, rxr 200
        instr_pad_id = 1 if self.config.MODEL.task_type == 'rxr' else 0

        observations = extract_instruction_tokens(observations, self.config.TASK_CONFIG.TASK.INSTRUCTION_SENSOR_UUID,
                                                    max_length=instr_max_len, pad_id=instr_pad_id)
        batch = batch_obs(observations, self.device)
        batch = apply_obs_transforms_batch(batch, self.obs_transforms)
        
        if mode == 'eval':
            env_to_pause = [i for i, ep in enumerate(self.envs.current_episodes()) 
                            if ep.episode_id in self.stat_eps]    
            self.envs, batch = self._pause_envs(self.envs, batch, env_to_pause)
            if self.envs.num_envs == 0: return
            
        batch_size = len(observations)

        # Get the instance label
        batch_scene_id = []
        batch_pcd_xyz = []
        batch_pcd_label = []
        #batch_instance_id_to_label = []
        batch_instance_id_to_object_type = []

        for b in range(batch_size):
            pcd_xyz = []
            pcd_label = []
            scene_id = self.envs.current_episodes()[b].scene_id
            if 'mp3d' in scene_id:
                scene_id = scene_id.split('/')[-1][:-4]
            elif 'hm3d' in scene_id:
                scene_id = scene_id.split('/')[-1][:-10]
                
            batch_scene_id.append(scene_id)
            if scene_id in self.hm3d_pcd_with_global_alignment:
                pcd_file_list = self.hm3d_pcd_with_global_alignment[scene_id]
                for pcd_file in pcd_file_list:
                    pcd_file = torch.load(pcd_file)
                    pcd_xyz.append(torch.tensor(pcd_file[0]))
                    pcd_label.append(torch.tensor(pcd_file[-1]))

                batch_instance_id_to_object_type.append(pcd_file[1])
                batch_pcd_xyz.append(torch.cat(pcd_xyz,0))
                batch_pcd_label.append(torch.cat(pcd_label,0))

                #instance_id_to_label_list = self.hm3d_instance_id_to_label[scene_id]
                #label_dict = {}
                #for label_file in instance_id_to_label_list:
                #    label_file = torch.load(label_file)
                #    label_dict.update(label_file)
                #batch_instance_id_to_label.append(label_dict)
            else:
                batch_pcd_xyz.append(None)
                batch_pcd_label.append(None)
                #batch_instance_id_to_label.append(None)
                batch_instance_id_to_object_type.append(None)


        gt_instance_text_fts = []
        gt_instance_category_id = []
        gt_zone_text_fts = []
        gt_zone_fine_grained_text_fts = []
        
        predicted_instance_fts = []
        predicted_zone_fts = []
        predicted_fine_grained_zone_fts = []
        
        sampled_gt_patch_fts, sampled_predicted_patch_fts = [], []
        sampled_gt_patch_language_label, sampled_predicted_patch_language_fts = [], []

        for stepk in range(self.max_len): 

            total_actions += 1

            positions = []; headings = []
            for ob_i in range(len(observations)):
                agent_state_i = self.envs.call_at(ob_i,
                        "get_agent_info", {})
                positions.append(agent_state_i['position'])
                headings.append(agent_state_i['heading'])

            policy_net = self.policy.net
            if hasattr(self.policy.net, 'module'):
                policy_net = self.policy.net.module

            if stepk == 0:
                policy_net.feature_fields.reset(batch_size, mode='habitat', batch_gt_pcd_xyz=batch_pcd_xyz, batch_gt_pcd_label=batch_pcd_label) # Reset the settings of 3D feature fields

            batch_size = len(observations)

            policy_net.positions = positions
            policy_net.headings = [(heading+2*math.pi)%(2*math.pi) for heading in headings]

            # cand waypoint prediction
            wp_outputs, sim_loss, segm_loss, batch_gt_3d_instance_id, batch_predicted_3d_instancs_fts, \
            batch_gt_3d_instance_ids_in_zone, batch_predicted_3d_zone_fts = self.policy.net(
                waypoint_predictor = self.waypoint_predictor,
                observations = batch,
                in_train = (mode == 'train' and self.config.IL.waypoint_aug),
            )
            loss += sim_loss + segm_loss

            # For Generalizable Feature Fields Pre-training
            sampled_view_num = 4 # The number of sampled novel views for each batch_id
            batch_angles, batch_distances = wp_outputs['cand_angles'], wp_outputs['cand_distances']

            for sampled_id in range(sampled_view_num):
                batch_selected_heading_angle = []
                batch_selected_position = []
                batch_selected_view_rgb = []
                #batch_selected_view_depth = []
                for b in range(batch_size):                
                    selected_nodes = random.choices(list(range(len(batch_angles[b]))), k=1)
                    node_id = selected_nodes[0]

                    selected_position = self.envs.call_at(b, "get_cand_real_pos", {"angle": batch_angles[b][node_id], "forward": batch_distances[b][node_id]})
                    selected_heading_angle = random.uniform(-math.pi,math.pi)
                    q1 = math.cos(selected_heading_angle/2)
                    q2 = math.sin(selected_heading_angle/2)
                    selected_rotation = np.quaternion(q1,0,q2,0)

                    camera_obs = self.envs.call_at(b, "get_observation",{"source_position":selected_position,"source_rotation":selected_rotation})

                    view_rgb = torch.tensor(camera_obs['rgb']).to(self.device).unsqueeze(0)
                    #view_depth = torch.tensor(camera_obs['depth']).to(self.device).unsqueeze(0)

                    batch_selected_heading_angle.append(selected_heading_angle)
                    batch_selected_position.append(selected_position)
                    batch_selected_view_rgb.append(view_rgb)
                    #batch_selected_view_depth.append(view_depth)


                # Get ground truth novel view CLIP features
                rgb_input = {}
                rgb_input['rgb'] = torch.cat(batch_selected_view_rgb,dim=0)
                with torch.no_grad():
                    clip_fts, patch_fts = policy_net.rgb_encoder(rgb_input)   
                    patch_fts = patch_fts.view(patch_fts.shape[0], 24,24,-1).permute(0,3,1,2)
                    avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)
                    patch_fts = avg_pool(patch_fts).permute(0,2,3,1)

                sampled_gt_patch_fts.append(patch_fts.view(batch_size,-1,patch_fts.shape[-1]))

                # Predict the novel view features and the instance id within this view
                batch_rendered_patch_fts, batch_rendered_patch_positions, gt_label = policy_net.feature_fields.render_view_3d_patch(batch_selected_position, batch_selected_heading_angle,visualization=False)
                sampled_predicted_patch_fts.append(batch_rendered_patch_fts.view(batch_size,-1,batch_rendered_patch_fts.shape[-1]))


                # Get language supervision of patch features

                for batch_id in range(batch_size):
                    if gt_label[batch_id] != None and batch_instance_id_to_object_type[batch_id] != None:
                        #instance_id_to_label = batch_instance_id_to_label[batch_id]
                        selected_mask = (gt_label[batch_id]>0)#.view(-1)
                        selected_gt_label = gt_label[batch_id][selected_mask]
                        if selected_gt_label.shape[0] == 0:
                            continue

                        selected_patch_fts = batch_rendered_patch_fts[batch_id][selected_mask]

                        #count = 0
                        #output_str = ''
                        for ray_id in range(selected_gt_label.shape[0]):
                            #count += 1
                            instance_id = int(selected_gt_label[ray_id].item())
                            category_id = batch_instance_id_to_object_type[batch_id][instance_id][1] #instance_id_to_label[instance_id]
                            #output_str += " "+category_id + " "
                            category_id = category_id.replace("_"," ").replace("|"," ").replace("/"," ")
                            if category_id in self.category_dict:
                                category_embedding_index = self.category_dict[category_id]
                                sampled_gt_patch_language_label.append(category_embedding_index)
                                sampled_predicted_patch_language_fts.append(selected_patch_fts[ray_id:ray_id+1])



            # Calculate the text alignment loss
            for b in range(batch_size):
                scene_id = batch_scene_id[b]
                instance_ids = batch_gt_3d_instance_id[b].cpu().numpy().tolist()
                zones = batch_gt_3d_instance_ids_in_zone[b]

                # Store the predicted features
                predicted_instance_fts.append(batch_predicted_3d_instancs_fts[b])
                predicted_zone_fts.extend([tmp_fts[0:1] for tmp_fts in batch_predicted_3d_zone_fts[b]])
                predicted_fine_grained_zone_fts.extend([tmp_fts[1:] for tmp_fts in batch_predicted_3d_zone_fts[b]])

                # Instance alignment
                for instance_id in instance_ids:
                    if batch_instance_id_to_object_type[b] == None or instance_id <= 0:
                        gt_instance_text_fts.append(torch.zeros((1,768),device=self.device))
                        gt_instance_category_id.append(-1)
                        continue
                    instance_id = batch_instance_id_to_object_type[b][instance_id][0] # Convert HM3D_id to Sceneverse_id
                    if str(instance_id) not in self.hm3d_language_annotations[scene_id]:
                        if instance_id in batch_instance_id_to_object_type[b]:
                            category_id = batch_instance_id_to_object_type[b][instance_id][1].replace("_"," ").replace("|"," ").replace("/"," ")
                            if category_id in self.category_dict:
                                category_embedding_index = self.category_dict[category_id]
                                gt_instance_text_fts.append(self.category_embeddings[category_embedding_index:category_embedding_index+1])
                                gt_instance_category_id.append(category_embedding_index)
                            else:
                                gt_instance_text_fts.append(torch.zeros((1,768),device=self.device))
                                gt_instance_category_id.append(-1)
                        else:
                            gt_instance_text_fts.append(torch.zeros((1,768),device=self.device))
                            gt_instance_category_id.append(-1)
                    else:
                        object_category, object_text = random.choice(self.hm3d_language_annotations[scene_id][str(instance_id)])
                        with torch.no_grad():
                            text_ids = policy_net.rgb_encoder.tokenize(object_text).to(policy_net.rgb_encoder.device)
                            text_fts = policy_net.rgb_encoder.model.encode_text(text_ids)
                            gt_instance_text_fts.append(text_fts)

                            category_id = object_category.replace("_"," ").replace("|"," ").replace("/"," ")
                            if category_id in self.category_dict:
                                category_embedding_index = self.category_dict[category_id]
                                gt_instance_category_id.append(category_embedding_index)
                            else:
                                gt_instance_category_id.append(-1)

                # Zone alignment
                for instance_ids_in_zone in zones:
                    if len(instance_ids_in_zone) > 0:
                        instance_ids_in_zone = list(set(instance_ids_in_zone.cpu().numpy().tolist()))
                        random.shuffle(instance_ids_in_zone)
                        for index, instance_id_in_zone in enumerate(instance_ids_in_zone):
                            if instance_id_in_zone <= 0 or batch_instance_id_to_object_type[b] == None or instance_id_in_zone not in batch_instance_id_to_object_type[b]:
                                gt_zone_text_fts.append(torch.zeros((1,768),device=self.device))
                                gt_zone_fine_grained_text_fts.append(torch.zeros((1,768),device=self.device))
                                break
                            instance_id_in_zone = batch_instance_id_to_object_type[b][instance_id_in_zone][0] # Convert HM3D_id to Sceneverse_id
                            if scene_id in self.hm3d_language_annotations and str(instance_id_in_zone) in self.hm3d_language_annotations[scene_id]:
                                object_category, object_text = random.choice(self.hm3d_language_annotations[scene_id][str(instance_id_in_zone)])
                                with torch.no_grad():
                                    text_ids = policy_net.rgb_encoder.tokenize(object_text).to(policy_net.rgb_encoder.device)
                                    text_fts, sep_fts = policy_net.rgb_encoder.model.encode_all_text(text_ids)
                                    gt_zone_text_fts.append(sep_fts)
                                    gt_zone_fine_grained_text_fts.append(text_fts)
                                break
                            
                            if index == len(instance_ids_in_zone) - 1:
                                if batch_instance_id_to_object_type[b] != None and instance_id_in_zone in batch_instance_id_to_object_type[b]:
                                    category_id = batch_instance_id_to_object_type[b][instance_id_in_zone][1].replace("_"," ").replace("|"," ").replace("/"," ")
                                    if category_id in self.category_dict:
                                        category_embedding_index = self.category_dict[category_id]
                                        gt_zone_text_fts.append(self.category_embeddings[category_embedding_index:category_embedding_index+1])
                                        gt_zone_fine_grained_text_fts.append(self.category_embeddings[category_embedding_index:category_embedding_index+1])
                                    else:
                                        gt_zone_text_fts.append(torch.zeros((1,768),device=self.device))
                                        gt_zone_fine_grained_text_fts.append(torch.zeros((1,768),device=self.device))
                                else:
                                    gt_zone_text_fts.append(torch.zeros((1,768),device=self.device))
                                    gt_zone_fine_grained_text_fts.append(torch.zeros((1,768),device=self.device))

                    else:
                        gt_zone_text_fts.append(torch.zeros((1,768),device=self.device))
                        gt_zone_fine_grained_text_fts.append(torch.zeros((1,768),device=self.device))

            
            # Get next waypoint to move
            batch_angles, batch_distances = wp_outputs['cand_angles'], wp_outputs['cand_distances']
            if random.choice([0,1]) == 0:
                teacher_actions = self._teacher_action(batch_angles, batch_distances)
            else:
                teacher_actions = np.array([random.choice([j for j in range(len(batch_angles[i]))]) for i in range(batch_size)])

            env_actions = []
            for i in range(batch_size):
                if teacher_actions[i] == -1 or stepk == self.max_len-1:
                    env_actions.append({'action':
                        {'action': 0, 'action_args':{}}})
                else:
                    env_actions.append({'action':
                        {'action': 4,  # HIGHTOLOW
                        'action_args':{
                            'angle': batch_angles[i][teacher_actions[i].item()], 
                            'distance': batch_distances[i][teacher_actions[i].item()],
                        }}})

            outputs = self.envs.step(env_actions)
            observations, _, dones, infos = [list(x) for x in
                                                zip(*outputs)]
            

            # pause env
            if sum(dones) > 0:
                for i in reversed(list(range(len(dones)))):
                    if dones[i]:
                        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                        self.envs.pause_at(i)
                        observations.pop(i)
                        policy_net.feature_fields.pop(i) # Very important, for some navigation processes finished in habitat simulator
                        batch_scene_id.pop(i)
                        batch_pcd_xyz.pop(i)
                        batch_pcd_label.pop(i)
                        #batch_instance_id_to_label.pop(i)
                        batch_instance_id_to_object_type.pop(i)
                        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            if self.envs.num_envs == 0:
                break

            # obs for next step
            observations = extract_instruction_tokens(observations,self.config.TASK_CONFIG.TASK.INSTRUCTION_SENSOR_UUID)
            batch = batch_obs(observations, self.device)
            batch = apply_obs_transforms_batch(batch, self.obs_transforms)

        loss = loss / total_actions

        # Feature Fields Pre-training
        sampled_gt_patch_fts = torch.cat(sampled_gt_patch_fts,dim=0).to(torch.float32)
        sampled_predicted_patch_fts = torch.cat(sampled_predicted_patch_fts,dim=0).to(torch.float32)

        subspace_sampled_gt_patch_fts = sampled_gt_patch_fts - sampled_gt_patch_fts.mean(1, keepdim=True)
        subspace_sampled_predicted_patch_fts = sampled_predicted_patch_fts - sampled_predicted_patch_fts.mean(1, keepdim=True)

        subspace_sampled_predicted_patch_fts = subspace_sampled_predicted_patch_fts / (torch.linalg.norm(subspace_sampled_predicted_patch_fts, dim=-1, keepdim=True) + 1e-5)
        subspace_sampled_gt_patch_fts = subspace_sampled_gt_patch_fts / (torch.linalg.norm(subspace_sampled_gt_patch_fts, dim=-1, keepdim=True) + 1e-5)
        loss += (1. - (subspace_sampled_predicted_patch_fts * subspace_sampled_gt_patch_fts).sum(-1)).mean() * 2.

        sampled_predicted_patch_fts = sampled_predicted_patch_fts / (torch.linalg.norm(sampled_predicted_patch_fts, dim=-1, keepdim=True) + 1e-5)
        sampled_gt_patch_fts = sampled_gt_patch_fts / (torch.linalg.norm(sampled_gt_patch_fts, dim=-1, keepdim=True) + 1e-5)

        sampled_predicted_patch_fts = sampled_predicted_patch_fts.view(-1,sampled_predicted_patch_fts.shape[-1])
        sampled_gt_patch_fts = sampled_gt_patch_fts.view(-1,sampled_gt_patch_fts.shape[-1])


        loss += (1. - (sampled_predicted_patch_fts * sampled_gt_patch_fts).sum(-1)).mean() * 5.

        loss += self.contrastive_loss(sampled_predicted_patch_fts, sampled_gt_patch_fts, logit_scale=10.) / 5.

        if len(gt_instance_text_fts) != 0 and torch.cat(gt_instance_text_fts,dim=0).sum().cpu().numpy().item() != 0:
            gt_instance_text_fts = torch.cat(gt_instance_text_fts,dim=0).to(torch.float32)
            gt_zone_text_fts = torch.cat(gt_zone_text_fts,dim=0).to(torch.float32)
            predicted_instance_fts = torch.cat(predicted_instance_fts,dim=0).to(torch.float32)
            predicted_zone_fts = torch.cat(predicted_zone_fts,dim=0).to(torch.float32)

            predicted_instance_fts = predicted_instance_fts / (torch.linalg.norm(predicted_instance_fts, dim=-1, keepdim=True) + 1e-5)
            gt_instance_text_fts = gt_instance_text_fts / (torch.linalg.norm(gt_instance_text_fts, dim=-1, keepdim=True) + 1e-5)
            
            target = torch.tensor(gt_instance_category_id, device=self.device)
            logits = predicted_instance_fts[target!=-1] @ self.category_embeddings.T * 10.
            target = target[target!=-1]
            loss +=  F.cross_entropy(logits, target) / 10.

            predicted_instance_fts = predicted_instance_fts[gt_instance_text_fts.sum(-1)!=0]
            gt_instance_text_fts = gt_instance_text_fts[gt_instance_text_fts.sum(-1)!=0]
            loss += self.contrastive_loss(predicted_instance_fts, gt_instance_text_fts, logit_scale=10.) / 5.

            predicted_zone_fts = predicted_zone_fts[gt_zone_text_fts.sum(-1)!=0]
            gt_zone_text_fts = gt_zone_text_fts[gt_zone_text_fts.sum(-1)!=0]
            
            predicted_zone_fts = predicted_zone_fts / (torch.linalg.norm(predicted_zone_fts, dim=-1, keepdim=True) + 1e-5)
            gt_zone_text_fts = gt_zone_text_fts / (torch.linalg.norm(gt_zone_text_fts, dim=-1, keepdim=True) + 1e-5)
            loss += self.contrastive_loss(predicted_zone_fts, gt_zone_text_fts, logit_scale=10.) / 5.

            #predicted_fine_grained_zone_fts = pad_sequence(predicted_fine_grained_zone_fts).permute(1,0,2)
            #loss += self.fine_grained_contrastive_loss(predicted_fine_grained_zone_fts,gt_zone_fine_grained_text_fts) / 5.

        # Feature Fields Pre-training
        if len(sampled_gt_patch_language_label) > 0:
            sampled_predicted_patch_language_fts = torch.cat(sampled_predicted_patch_language_fts,dim=0).to(torch.float32)
            sampled_predicted_patch_language_fts = sampled_predicted_patch_language_fts / torch.linalg.norm(sampled_predicted_patch_language_fts, dim=-1, keepdim=True)
            logits = sampled_predicted_patch_language_fts @ self.category_embeddings.T * 10.
            target = torch.tensor(sampled_gt_patch_language_label, device=self.device)
            loss += self.focal_loss(logits,target) / 10.

        policy_net.feature_fields.delete_feature_fields()

        return loss


    def run_on_scannet(self, mode):
        loss = 0.
        policy_net = self.policy.net
        if hasattr(self.policy.net, 'module'):
            policy_net = self.policy.net.module

        batch_size = self.batch_size # Larger batch size is better, so different from batch_size of hm3d is also ok

        num_of_sampled_images = 16 # Number of sampled images to construct the feature fields
            
        batch_camera_intrinsic = []
        batch_grid_ft = []
        batch_image_ft = []
        batch_depth = []
        batch_rot = []
        batch_trans = []
        batch_extrinsic = []

        # Get the instance label
        batch_pcd_xyz = []
        batch_pcd_label = []
        batch_instance_id_to_label = []
        batch_scene_id = []
        batch_image = []

        gt_instance_text_fts = []
        gt_instance_category_id = []
        gt_zone_text_fts = []
        gt_zone_fine_grained_text_fts = []
        
        predicted_instance_fts = []
        predicted_zone_fts = []
        predicted_fine_grained_zone_fts = []

        sampled_gt_patch_fts, sampled_predicted_patch_fts = [], []
        sampled_gt_patch_language_label, sampled_predicted_patch_language_fts = [], []

        for batch_id in range(batch_size):

            scene_id = random.choice(list(set(self.ScanNet_pcd_with_global_alignment.keys()) & set(self.ScanNet_language_annotations.keys())))
            file_path = self.scannet_3d_scenes[scene_id]
            batch_scene_id.append(scene_id)

            # Get the instance label
            pcd_xyz = []
            pcd_label = []
            if scene_id in self.ScanNet_pcd_with_global_alignment:
                    
                pcd_file_list = self.ScanNet_pcd_with_global_alignment[scene_id]
                for pcd_file in pcd_file_list:
                    pcd_file = torch.load(pcd_file)
                    align_matrix = torch.tensor(np.linalg.inv(self.scannet_align_matrix[scene_id])).to(self.device).to(torch.float32) # Align the coordinate using align_matrix
                    pts = np.ones((pcd_file[0].shape[0], 4), dtype=np.float32)
                    pts[:, 0:3] = pcd_file[0]
                    aligned_xyz = (torch.tensor(pts).to(self.device) @ align_matrix.T)[:, :3].cpu()
                    pcd_xyz.append(aligned_xyz)
                    pcd_label.append(torch.tensor(pcd_file[3])) # 3 not 2, different from hm3d and structured
                batch_pcd_xyz.append(torch.cat(pcd_xyz,0))
                batch_pcd_label.append(torch.cat(pcd_label,0))

                instance_id_to_label_list = self.ScanNet_instance_id_to_label[scene_id]
                label_dict = {}
                for label_file in instance_id_to_label_list:
                    label_file = torch.load(label_file)
                    label_dict.update(label_file)
                batch_instance_id_to_label.append(label_dict)
            else:
                batch_pcd_xyz.append(None)
                batch_pcd_label.append(None)
                batch_instance_id_to_label.append(None)

            image_list = os.listdir(file_path + '/color/')
            for image_id in range(len(image_list)):  
                image_list[image_id] = image_list[image_id][:-4]

            random.shuffle(image_list)
            image_list = image_list[:num_of_sampled_images]

            while len(image_list) < num_of_sampled_images:
                image_list += image_list[:num_of_sampled_images-len(image_list)] # Full number for num_of_sampled_images

            intrinsic_list = []
            extrinsic_list = []
            R_list = []
            T_list = []
            rgb_list = []
            depth_list = []
            for image_id in image_list:
                intrinsic = np.eye(4)
                with open(file_path + '/intrinsic_depth.txt', 'r') as file:  
                    intrinsic_raw = [line.strip() for line in file]
                for i in range(4):  
                    for j in range(4): 
                        intrinsic[i][j] = float(intrinsic_raw[i].split()[j])

                # divide 2 is necessary for camera intrinsics of scannet dataset due to the changed resolution
                intrinsic[0][0] =  intrinsic[0][0] / 2.
                intrinsic[1][1] =  intrinsic[1][1] / 2.
                intrinsic[0][2] =  intrinsic[0][2] / 2.
                intrinsic[1][2] =  intrinsic[1][2] / 2.
                intrinsic_list.append(intrinsic)
                extrinsic = np.eye(4)

                with open(file_path + '/pose/' + image_id + '.txt', 'r') as file:  
                    extrinsic_raw = [line.strip() for line in file]
                for i in range(4):  
                    for j in range(4): 
                        extrinsic[i][j] = float(extrinsic_raw[i].split()[j])
                R = extrinsic[:3,:3]
                T = extrinsic[:3,3:4]

                rgb_image = np.asarray(Image.open(file_path + '/color/' + image_id + ".jpg"))
                depth_image = np.asarray(Image.open(file_path + '/depth/' + image_id + ".png"))

                R_list.append(R)
                T_list.append(T)
                extrinsic_list.append(np.linalg.inv(extrinsic))
                rgb_list.append(torch.tensor(rgb_image).unsqueeze(0))
                depth_list.append(torch.tensor(depth_image).unsqueeze(0))

            rgb_list = torch.cat(rgb_list,dim=0)
            depth_list = torch.cat(depth_list,dim=0)
            with torch.no_grad():
                img_fts, grid_fts = policy_net.rgb_encoder({'rgb':rgb_list})
                patch_fts = grid_fts.view(grid_fts.shape[0], 24,24,-1).permute(0,3,1,2)
                avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)
                patch_fts = avg_pool(patch_fts).permute(0,2,3,1)
                sampled_gt_patch_fts.append(patch_fts.view(num_of_sampled_images,-1,patch_fts.shape[-1]))


            batch_camera_intrinsic.append(np.stack(intrinsic_list,axis=0))
            batch_rot.append(np.stack(R_list,axis=0))
            batch_trans.append(np.stack(T_list,axis=0))
            batch_extrinsic.append(np.stack(extrinsic_list,axis=0))
            batch_grid_ft.append(grid_fts.cpu().numpy())
            batch_image.append(rgb_list)
            batch_image_ft.append(img_fts)
            batch_depth.append(depth_list)

        batch_camera_intrinsic = np.stack(batch_camera_intrinsic,axis=0)
        batch_rot = np.stack(batch_rot,axis=0)
        batch_trans = np.stack(batch_trans,axis=0)
        batch_extrinsic = np.stack(batch_extrinsic,axis=0)
        batch_image = torch.stack(batch_image,0).numpy()

        policy_net.feature_fields.reset(batch_size=batch_size, mode="scannet", batch_gt_pcd_xyz=batch_pcd_xyz, batch_gt_pcd_label=batch_pcd_label)  # Reset the settings of 3D feature fields
        # Do not change the order of the following two lines !!!!!
        policy_net.feature_fields.delete_old_features_from_camera_frustum(batch_depth=batch_depth, batch_camera_intrinsic=batch_camera_intrinsic, batch_extrinsic=batch_extrinsic)
        
        sim_loss, segm_loss, batch_gt_3d_instance_id,batch_predicted_3d_instancs_fts,batch_gt_3d_instance_ids_in_zone,batch_predicted_3d_zone_fts = policy_net.feature_fields.update_feature_fields(batch_depth, batch_grid_ft, batch_image, batch_image_ft=batch_image_ft, batch_camera_intrinsic=batch_camera_intrinsic, batch_rot=batch_rot, batch_trans=batch_trans, depth_scale=1000., depth_trunc=1000.)
        loss += sim_loss + segm_loss


        # Predict the novel view features and the instance id within this view
        for image_idx in range(num_of_sampled_images):
            batch_rendered_patch_fts, batch_rendered_patch_positions, gt_label = policy_net.feature_fields.render_view_3d_patch(batch_camera_intrinsic=batch_camera_intrinsic[:,image_idx], batch_rot=batch_rot[:,image_idx], batch_trans=batch_trans[:,image_idx], visualization=False)
            sampled_predicted_patch_fts.append(batch_rendered_patch_fts.view(batch_size,1, -1,batch_rendered_patch_fts.shape[-1]))

            # Get language supervision of patch features
            for batch_id in range(batch_size):
                if gt_label[batch_id] != None and batch_instance_id_to_label[batch_id] != None:
                    instance_id_to_label = batch_instance_id_to_label[batch_id]
                    selected_mask = (gt_label[batch_id]>0)#.view(-1)
                    selected_gt_label = gt_label[batch_id][selected_mask]
                    if selected_gt_label.shape[0] == 0:
                        continue

                    selected_patch_fts = batch_rendered_patch_fts[batch_id][selected_mask]

                    for ray_id in range(selected_gt_label.shape[0]):
                        instance_id = int(selected_gt_label[ray_id].item())
                        category_id = instance_id_to_label[instance_id]
                        category_id = category_id.replace("_"," ").replace("|"," ").replace("/"," ")
                        if category_id in self.category_dict:
                            category_embedding_index = self.category_dict[category_id]
                            sampled_gt_patch_language_label.append(category_embedding_index)
                            sampled_predicted_patch_language_fts.append(selected_patch_fts[ray_id:ray_id+1])

        sampled_predicted_patch_fts = [torch.cat(sampled_predicted_patch_fts,dim=1).view(-1,sampled_predicted_patch_fts[0].shape[-2],sampled_predicted_patch_fts[0].shape[-1])] # Keep the correct order of patch features

        # Calculate the text alignment loss
        for b in range(batch_size):
            scene_id = batch_scene_id[b]
            instance_ids = batch_gt_3d_instance_id[b].cpu().numpy().tolist()
            zones = batch_gt_3d_instance_ids_in_zone[b]

            # Store the predicted features
            predicted_instance_fts.append(batch_predicted_3d_instancs_fts[b])

            predicted_zone_fts.extend([tmp_fts[0:1] for tmp_fts in batch_predicted_3d_zone_fts[b]])
            predicted_fine_grained_zone_fts.extend([tmp_fts[1:] for tmp_fts in batch_predicted_3d_zone_fts[b]])

            if scene_id in self.ScanNet_language_annotations:
                # Instance alignment
                for instance_id in instance_ids:
                    if str(instance_id) not in self.ScanNet_language_annotations[scene_id]:
                        if instance_id in batch_instance_id_to_label[b]:
                            category_id = batch_instance_id_to_label[b][instance_id].replace("_"," ").replace("|"," ").replace("/"," ")
                            category_embedding_index = self.category_dict[category_id]
                            gt_instance_text_fts.append(self.category_embeddings[category_embedding_index:category_embedding_index+1])
                            gt_instance_category_id.append(category_embedding_index)
                        else:
                            gt_instance_text_fts.append(torch.zeros((1,768),device=self.device))
                            gt_instance_category_id.append(-1)
                    else:
                        object_category, object_text = random.choice(self.ScanNet_language_annotations[scene_id][str(instance_id)])
                        with torch.no_grad():
                            text_ids = policy_net.rgb_encoder.tokenize(object_text).to(policy_net.rgb_encoder.device)
                            text_fts = policy_net.rgb_encoder.model.encode_text(text_ids)
                            gt_instance_text_fts.append(text_fts)

                            category_id = object_category.replace("_"," ").replace("|"," ").replace("/"," ")
                            if category_id in self.category_dict:
                                category_embedding_index = self.category_dict[category_id]
                                gt_instance_category_id.append(category_embedding_index)
                            else:
                                gt_instance_category_id.append(-1)

                # Zone alignment
                for instance_ids_in_zone in zones:
                    if len(instance_ids_in_zone) > 0:
                        instance_ids_in_zone = list(set(instance_ids_in_zone.cpu().numpy().tolist()))
                        random.shuffle(instance_ids_in_zone)
                        for index, instance_id_in_zone in enumerate(instance_ids_in_zone):
                            if str(instance_id_in_zone) in self.ScanNet_language_annotations[scene_id]:
                                object_category, object_text = random.choice(self.ScanNet_language_annotations[scene_id][str(instance_id_in_zone)])
                                with torch.no_grad():
                                    text_ids = policy_net.rgb_encoder.tokenize(object_text).to(policy_net.rgb_encoder.device)
                                    text_fts, sep_fts = policy_net.rgb_encoder.model.encode_all_text(text_ids)
                                    gt_zone_text_fts.append(sep_fts)
                                    gt_zone_fine_grained_text_fts.append(text_fts)
                                break
                            
                            if index == len(instance_ids_in_zone) - 1:
                                if instance_id_in_zone in batch_instance_id_to_label[b]:
                                    category_id = batch_instance_id_to_label[b][instance_id_in_zone].replace("_"," ").replace("|"," ").replace("/"," ")
                                    category_embedding_index = self.category_dict[category_id]
                                    gt_zone_text_fts.append(self.category_embeddings[category_embedding_index:category_embedding_index+1])
                                    gt_zone_fine_grained_text_fts.append(self.category_embeddings[category_embedding_index:category_embedding_index+1])
                                else:
                                    gt_zone_text_fts.append(torch.zeros((1,768),device=self.device))
                                    gt_zone_fine_grained_text_fts.append(torch.zeros((1,768),device=self.device))
                    else:
                        gt_zone_text_fts.append(torch.zeros((1,768),device=self.device))
                        gt_zone_fine_grained_text_fts.append(torch.zeros((1,768),device=self.device))


        # Feature Fields Pre-training
        sampled_gt_patch_fts = torch.cat(sampled_gt_patch_fts,dim=0).to(torch.float32)
        sampled_predicted_patch_fts = torch.cat(sampled_predicted_patch_fts,dim=0).to(torch.float32)

        subspace_sampled_gt_patch_fts = sampled_gt_patch_fts - sampled_gt_patch_fts.mean(1, keepdim=True)
        subspace_sampled_predicted_patch_fts = sampled_predicted_patch_fts - sampled_predicted_patch_fts.mean(1, keepdim=True)

        subspace_sampled_predicted_patch_fts = subspace_sampled_predicted_patch_fts / (torch.linalg.norm(subspace_sampled_predicted_patch_fts, dim=-1, keepdim=True) + 1e-5)
        subspace_sampled_gt_patch_fts = subspace_sampled_gt_patch_fts / (torch.linalg.norm(subspace_sampled_gt_patch_fts, dim=-1, keepdim=True) + 1e-5)
        loss += (1. - (subspace_sampled_predicted_patch_fts * subspace_sampled_gt_patch_fts).sum(-1)).mean() * 2.

        sampled_predicted_patch_fts = sampled_predicted_patch_fts / (torch.linalg.norm(sampled_predicted_patch_fts, dim=-1, keepdim=True) + 1e-5)
        sampled_gt_patch_fts = sampled_gt_patch_fts / (torch.linalg.norm(sampled_gt_patch_fts, dim=-1, keepdim=True) + 1e-5)

        sampled_predicted_patch_fts = sampled_predicted_patch_fts.view(-1,sampled_predicted_patch_fts.shape[-1])
        sampled_gt_patch_fts = sampled_gt_patch_fts.view(-1,sampled_gt_patch_fts.shape[-1])

        loss += (1. - (sampled_predicted_patch_fts * sampled_gt_patch_fts).sum(-1)).mean() * 5.
        loss += self.contrastive_loss(sampled_predicted_patch_fts, sampled_gt_patch_fts, logit_scale=10.) / 5.
        
        if len(gt_instance_text_fts) != 0 and torch.cat(gt_instance_text_fts,dim=0).sum().cpu().numpy().item() != 0:
            gt_instance_text_fts = torch.cat(gt_instance_text_fts,dim=0).to(torch.float32)
            gt_zone_text_fts = torch.cat(gt_zone_text_fts,dim=0).to(torch.float32)
            predicted_instance_fts = torch.cat(predicted_instance_fts,dim=0).to(torch.float32)
            predicted_zone_fts = torch.cat(predicted_zone_fts,dim=0).to(torch.float32)

            predicted_instance_fts = predicted_instance_fts / (torch.linalg.norm(predicted_instance_fts, dim=-1, keepdim=True) + 1e-5)
            gt_instance_text_fts = gt_instance_text_fts / (torch.linalg.norm(gt_instance_text_fts, dim=-1, keepdim=True) + 1e-5)
            
            target = torch.tensor(gt_instance_category_id, device=self.device)
            logits = predicted_instance_fts[target!=-1] @ self.category_embeddings.T * 10.
            target = target[target!=-1]
            loss +=  F.cross_entropy(logits, target) / 10.


            predicted_instance_fts = predicted_instance_fts[gt_instance_text_fts.sum(-1)!=0]
            gt_instance_text_fts = gt_instance_text_fts[gt_instance_text_fts.sum(-1)!=0]
            loss += self.contrastive_loss(predicted_instance_fts, gt_instance_text_fts, logit_scale=10.) / 5.

            predicted_zone_fts = predicted_zone_fts[gt_zone_text_fts.sum(-1)!=0]
            gt_zone_text_fts = gt_zone_text_fts[gt_zone_text_fts.sum(-1)!=0]
            
            predicted_zone_fts = predicted_zone_fts / (torch.linalg.norm(predicted_zone_fts, dim=-1, keepdim=True) + 1e-5)
            gt_zone_text_fts = gt_zone_text_fts / (torch.linalg.norm(gt_zone_text_fts, dim=-1, keepdim=True) + 1e-5)
            loss += self.contrastive_loss(predicted_zone_fts, gt_zone_text_fts, logit_scale=10.) / 5.

            #predicted_fine_grained_zone_fts = pad_sequence(predicted_fine_grained_zone_fts).permute(1,0,2)
            #loss += self.fine_grained_contrastive_loss(predicted_fine_grained_zone_fts,gt_zone_fine_grained_text_fts) / 5.

        # Feature Fields Pre-training
        if len(sampled_gt_patch_language_label) > 0:
            sampled_predicted_patch_language_fts = torch.cat(sampled_predicted_patch_language_fts,dim=0).to(torch.float32)
            sampled_predicted_patch_language_fts = sampled_predicted_patch_language_fts / torch.linalg.norm(sampled_predicted_patch_language_fts, dim=-1, keepdim=True)
            logits = sampled_predicted_patch_language_fts @ self.category_embeddings.T * 10.
            target = torch.tensor(sampled_gt_patch_language_label, device=self.device)
            loss += self.focal_loss(logits,target) / 10.
                
        
        policy_net.feature_fields.delete_feature_fields()

        return loss
    


    def run_on_3rscan(self, mode):
        loss = 0.
        policy_net = self.policy.net
        if hasattr(self.policy.net, 'module'):
            policy_net = self.policy.net.module

        batch_size = self.batch_size # Larger batch size is better, so different from batch_size of hm3d is also ok

        num_of_sampled_images = 16 # Number of sampled images to construct the feature fields
            
        batch_camera_intrinsic = []
        batch_grid_ft = []
        batch_image_ft = []
        batch_depth = []
        batch_rot = []
        batch_trans = []
        batch_extrinsic = []

        # Get the instance label
        batch_pcd_xyz = []
        batch_pcd_label = []
        batch_instance_id_to_label = []
        batch_scene_id = []
        batch_image = []

        gt_instance_text_fts = []
        gt_instance_category_id = []
        gt_zone_text_fts = []
        gt_zone_fine_grained_text_fts = []
        
        predicted_instance_fts = []
        predicted_zone_fts = []
        predicted_fine_grained_zone_fts = []

        sampled_gt_patch_fts, sampled_predicted_patch_fts = [], []
        sampled_gt_patch_language_label, sampled_predicted_patch_language_fts = [], []

        for batch_id in range(batch_size):

            scene_id = random.choice(list(set(self.rscan_pcd_with_global_alignment.keys()) & set(self.rscan_language_annotations.keys())))
            #file_path = self.rscan_scenes[scene_id]
            batch_scene_id.append(scene_id)

            # Get the instance label
            pcd_xyz = []
            pcd_label = []
            if scene_id in self.rscan_pcd_with_global_alignment:
                    
                pcd_file_list = self.rscan_pcd_with_global_alignment[scene_id]
                for pcd_file in pcd_file_list:
                    pcd_file = torch.load(pcd_file)
                    #align_matrix = torch.tensor(np.linalg.inv(self.scannet_align_matrix[scene_id])).to(self.device).to(torch.float32) # Align the coordinate using align_matrix
                    #pts = np.ones((pcd_file[0].shape[0], 4), dtype=np.float32)
                    pts = pcd_file[0]
                    #aligned_xyz = (torch.tensor(pts).to(self.device) @ align_matrix.T)[:, :3].cpu()
                    pcd_xyz.append(torch.tensor(pts))
                    pcd_label.append(torch.tensor(pcd_file[-1])) # 3 not 2, different from hm3d and structured

                batch_pcd_xyz.append(torch.cat(pcd_xyz,0))
                batch_pcd_label.append(torch.cat(pcd_label,0))

                instance_id_to_label_list = self.rscan_instance_id_to_label[scene_id]
                label_dict = {}
                for label_file in instance_id_to_label_list:
                    label_file = torch.load(label_file)
                    label_dict.update(label_file)
                batch_instance_id_to_label.append(label_dict)
            else:
                batch_pcd_xyz.append(None)
                batch_pcd_label.append(None)
                batch_instance_id_to_label.append(None)


            image_list = os.listdir('data/3RScan/scenes/'+scene_id+'/sequence/')
            image_list.remove('_info.txt')
            for image_id in range(len(image_list)):  
                image_list[image_id] = 'data/3RScan/scenes/'+scene_id+'/sequence/'+image_list[image_id].split(".")[0]

            image_list = list(set(image_list))
            try:
                image_list.remove('_info')
            except:
                pass

            random.shuffle(image_list)
            image_list = image_list[:num_of_sampled_images]

            while len(image_list) < num_of_sampled_images:
                image_list += image_list[:num_of_sampled_images-len(image_list)] # Full number for num_of_sampled_images

            intrinsic_list = []
            extrinsic_list = []
            R_list = []
            T_list = []
            rgb_list = []
            depth_list = []
            for image_id in image_list:
                intrinsic = np.eye(4)
                with open('data/3RScan/scenes/'+scene_id+'/sequence/_info.txt', 'r') as file:  
                    intrinsic_raw = [line.strip() for line in file]
                intrinsic_raw = intrinsic_raw[9].split(" ")[2:]

                for i in range(4):  
                    for j in range(4): 
                        intrinsic[i][j] = float(intrinsic_raw[i*4+j])

                intrinsic_list.append(intrinsic)
                extrinsic = np.eye(4)
                with open(image_id+'.pose.txt', 'r') as file:  
                    extrinsic_raw = [line.strip() for line in file]
                for i in range(4):  
                    for j in range(4): 
                        extrinsic[i][j] = float(extrinsic_raw[i].split()[j])

                R = extrinsic[:3,:3]
                T = extrinsic[:3,3:4]

                rgb_image = np.asarray(Image.open(image_id + ".color.jpg"))
                depth_image = np.asarray(Image.open(image_id + ".depth.pgm"),dtype=np.float32)

                R_list.append(R)
                T_list.append(T)
                extrinsic_list.append(np.linalg.inv(extrinsic))
                rgb_list.append(torch.tensor(rgb_image).unsqueeze(0))
                depth_list.append(torch.tensor(depth_image).unsqueeze(0))

            rgb_list = torch.cat(rgb_list,dim=0)
            depth_list = torch.cat(depth_list,dim=0)
            with torch.no_grad():
                img_fts, grid_fts = policy_net.rgb_encoder({'rgb':rgb_list})
                patch_fts = grid_fts.view(grid_fts.shape[0], 24,24,-1).permute(0,3,1,2)
                avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)
                patch_fts = avg_pool(patch_fts).permute(0,2,3,1)
                sampled_gt_patch_fts.append(patch_fts.view(num_of_sampled_images,-1,patch_fts.shape[-1]))


            batch_camera_intrinsic.append(np.stack(intrinsic_list,axis=0))
            batch_rot.append(np.stack(R_list,axis=0))
            batch_trans.append(np.stack(T_list,axis=0))
            batch_extrinsic.append(np.stack(extrinsic_list,axis=0))
            batch_grid_ft.append(grid_fts.cpu().numpy())
            batch_image.append(rgb_list)
            batch_image_ft.append(img_fts)
            batch_depth.append(depth_list)

        batch_camera_intrinsic = np.stack(batch_camera_intrinsic,axis=0)
        batch_rot = np.stack(batch_rot,axis=0)
        batch_trans = np.stack(batch_trans,axis=0)
        batch_extrinsic = np.stack(batch_extrinsic,axis=0)
        batch_image = torch.stack(batch_image,0).numpy()


        policy_net.feature_fields.reset(batch_size=batch_size, mode="scannet", batch_gt_pcd_xyz=batch_pcd_xyz, batch_gt_pcd_label=batch_pcd_label)  # Reset the settings of 3D feature fields
        # Do not change the order of the following two lines !!!!!
        policy_net.feature_fields.delete_old_features_from_camera_frustum(batch_depth=batch_depth, batch_camera_intrinsic=batch_camera_intrinsic, batch_extrinsic=batch_extrinsic)
        sim_loss, segm_loss, batch_gt_3d_instance_id,batch_predicted_3d_instancs_fts,batch_gt_3d_instance_ids_in_zone,batch_predicted_3d_zone_fts = policy_net.feature_fields.update_feature_fields(batch_depth, batch_grid_ft, batch_image, batch_image_ft=batch_image_ft, batch_camera_intrinsic=batch_camera_intrinsic, batch_rot=batch_rot, batch_trans=batch_trans, depth_scale=1000., depth_trunc=1000.)
        loss += sim_loss + segm_loss

        # Predict the novel view features and the instance id within this view
        for image_idx in range(num_of_sampled_images):
            batch_rendered_patch_fts, batch_rendered_patch_positions, gt_label = policy_net.feature_fields.render_view_3d_patch(batch_camera_intrinsic=batch_camera_intrinsic[:,image_idx], batch_rot=batch_rot[:,image_idx], batch_trans=batch_trans[:,image_idx], visualization=False)
            sampled_predicted_patch_fts.append(batch_rendered_patch_fts.view(batch_size,1, -1,batch_rendered_patch_fts.shape[-1]))

            # Get language supervision of patch features
            for batch_id in range(batch_size):
                if gt_label[batch_id] != None and batch_instance_id_to_label[batch_id] != None:
                    instance_id_to_label = batch_instance_id_to_label[batch_id]
                    selected_mask = (gt_label[batch_id]>0)#.view(-1)
                    selected_gt_label = gt_label[batch_id][selected_mask]
                    if selected_gt_label.shape[0] == 0:
                        continue

                    selected_patch_fts = batch_rendered_patch_fts[batch_id][selected_mask]

                    for ray_id in range(selected_gt_label.shape[0]):
                        instance_id = int(selected_gt_label[ray_id].item())
                        category_id = instance_id_to_label[instance_id]
                        category_id = category_id.replace("_"," ").replace("|"," ").replace("/"," ")
                        if category_id in self.category_dict:
                            category_embedding_index = self.category_dict[category_id]
                            sampled_gt_patch_language_label.append(category_embedding_index)
                            sampled_predicted_patch_language_fts.append(selected_patch_fts[ray_id:ray_id+1])

        sampled_predicted_patch_fts = [torch.cat(sampled_predicted_patch_fts,dim=1).view(-1,sampled_predicted_patch_fts[0].shape[-2],sampled_predicted_patch_fts[0].shape[-1])] # Keep the correct order of patch features

        # Calculate the text alignment loss
        for b in range(batch_size):
            scene_id = batch_scene_id[b]
            instance_ids = batch_gt_3d_instance_id[b].cpu().numpy().tolist()
            zones = batch_gt_3d_instance_ids_in_zone[b]

            # Store the predicted features
            predicted_instance_fts.append(batch_predicted_3d_instancs_fts[b])

            predicted_zone_fts.extend([tmp_fts[0:1] for tmp_fts in batch_predicted_3d_zone_fts[b]])
            predicted_fine_grained_zone_fts.extend([tmp_fts[1:] for tmp_fts in batch_predicted_3d_zone_fts[b]])

            if scene_id in self.rscan_language_annotations:
                # Instance alignment
                for instance_id in instance_ids:
                    if str(instance_id) not in self.rscan_language_annotations[scene_id]:
                        if instance_id in batch_instance_id_to_label[b]:
                            category_id = batch_instance_id_to_label[b][instance_id].replace("_"," ").replace("|"," ").replace("/"," ")
                            category_embedding_index = self.category_dict[category_id]
                            gt_instance_text_fts.append(self.category_embeddings[category_embedding_index:category_embedding_index+1])
                            gt_instance_category_id.append(category_embedding_index)
                        else:
                            gt_instance_text_fts.append(torch.zeros((1,768),device=self.device))
                            gt_instance_category_id.append(-1)
                    else:
                        object_category, object_text = random.choice(self.rscan_language_annotations[scene_id][str(instance_id)])
                        with torch.no_grad():
                            text_ids = policy_net.rgb_encoder.tokenize(object_text).to(policy_net.rgb_encoder.device)
                            text_fts = policy_net.rgb_encoder.model.encode_text(text_ids)
                            gt_instance_text_fts.append(text_fts)

                            category_id = object_category.replace("_"," ").replace("|"," ").replace("/"," ")
                            if category_id in self.category_dict:
                                category_embedding_index = self.category_dict[category_id]
                                gt_instance_category_id.append(category_embedding_index)
                            else:
                                gt_instance_category_id.append(-1)

                # Zone alignment
                for instance_ids_in_zone in zones:
                    if len(instance_ids_in_zone) > 0:
                        instance_ids_in_zone = list(set(instance_ids_in_zone.cpu().numpy().tolist()))
                        random.shuffle(instance_ids_in_zone)
                        for index, instance_id_in_zone in enumerate(instance_ids_in_zone):
                            if str(instance_id_in_zone) in self.rscan_language_annotations[scene_id]:
                                object_category, object_text = random.choice(self.rscan_language_annotations[scene_id][str(instance_id_in_zone)])
                                with torch.no_grad():
                                    text_ids = policy_net.rgb_encoder.tokenize(object_text).to(policy_net.rgb_encoder.device)
                                    text_fts, sep_fts = policy_net.rgb_encoder.model.encode_all_text(text_ids)
                                    gt_zone_text_fts.append(sep_fts)
                                    gt_zone_fine_grained_text_fts.append(text_fts)
                                break
                            
                            if index == len(instance_ids_in_zone) - 1:
                                if instance_id_in_zone in batch_instance_id_to_label[b]:
                                    category_id = batch_instance_id_to_label[b][instance_id_in_zone].replace("_"," ").replace("|"," ").replace("/"," ")
                                    category_embedding_index = self.category_dict[category_id]
                                    gt_zone_text_fts.append(self.category_embeddings[category_embedding_index:category_embedding_index+1])
                                    gt_zone_fine_grained_text_fts.append(self.category_embeddings[category_embedding_index:category_embedding_index+1])
                                else:
                                    gt_zone_text_fts.append(torch.zeros((1,768),device=self.device))
                                    gt_zone_fine_grained_text_fts.append(torch.zeros((1,768),device=self.device))
                    else:
                        gt_zone_text_fts.append(torch.zeros((1,768),device=self.device))
                        gt_zone_fine_grained_text_fts.append(torch.zeros((1,768),device=self.device))


        # Feature Fields Pre-training
        sampled_gt_patch_fts = torch.cat(sampled_gt_patch_fts,dim=0).to(torch.float32)
        sampled_predicted_patch_fts = torch.cat(sampled_predicted_patch_fts,dim=0).to(torch.float32)

        subspace_sampled_gt_patch_fts = sampled_gt_patch_fts - sampled_gt_patch_fts.mean(1, keepdim=True)
        subspace_sampled_predicted_patch_fts = sampled_predicted_patch_fts - sampled_predicted_patch_fts.mean(1, keepdim=True)

        subspace_sampled_predicted_patch_fts = subspace_sampled_predicted_patch_fts / (torch.linalg.norm(subspace_sampled_predicted_patch_fts, dim=-1, keepdim=True) + 1e-5)
        subspace_sampled_gt_patch_fts = subspace_sampled_gt_patch_fts / (torch.linalg.norm(subspace_sampled_gt_patch_fts, dim=-1, keepdim=True) + 1e-5)
        loss += (1. - (subspace_sampled_predicted_patch_fts * subspace_sampled_gt_patch_fts).sum(-1)).mean() * 2.

        sampled_predicted_patch_fts = sampled_predicted_patch_fts / (torch.linalg.norm(sampled_predicted_patch_fts, dim=-1, keepdim=True) + 1e-5)
        sampled_gt_patch_fts = sampled_gt_patch_fts / (torch.linalg.norm(sampled_gt_patch_fts, dim=-1, keepdim=True) + 1e-5)

        sampled_predicted_patch_fts = sampled_predicted_patch_fts.view(-1,sampled_predicted_patch_fts.shape[-1])
        sampled_gt_patch_fts = sampled_gt_patch_fts.view(-1,sampled_gt_patch_fts.shape[-1])


        loss += (1. - (sampled_predicted_patch_fts * sampled_gt_patch_fts).sum(-1)).mean() * 5.
        loss += self.contrastive_loss(sampled_predicted_patch_fts, sampled_gt_patch_fts, logit_scale=10.) / 5.

        
        if len(gt_instance_text_fts) != 0 and torch.cat(gt_instance_text_fts,dim=0).sum().cpu().numpy().item() != 0:
            gt_instance_text_fts = torch.cat(gt_instance_text_fts,dim=0).to(torch.float32)
            gt_zone_text_fts = torch.cat(gt_zone_text_fts,dim=0).to(torch.float32)
            predicted_instance_fts = torch.cat(predicted_instance_fts,dim=0).to(torch.float32)
            predicted_zone_fts = torch.cat(predicted_zone_fts,dim=0).to(torch.float32)

            predicted_instance_fts = predicted_instance_fts / (torch.linalg.norm(predicted_instance_fts, dim=-1, keepdim=True) + 1e-5)
            gt_instance_text_fts = gt_instance_text_fts / (torch.linalg.norm(gt_instance_text_fts, dim=-1, keepdim=True) + 1e-5)
            
            target = torch.tensor(gt_instance_category_id, device=self.device)
            logits = predicted_instance_fts[target!=-1] @ self.category_embeddings.T * 10.
            target = target[target!=-1]
            loss +=  F.cross_entropy(logits, target) / 10.

            predicted_instance_fts = predicted_instance_fts[gt_instance_text_fts.sum(-1)!=0]
            gt_instance_text_fts = gt_instance_text_fts[gt_instance_text_fts.sum(-1)!=0]
            loss += self.contrastive_loss(predicted_instance_fts, gt_instance_text_fts, logit_scale=10.) / 5.

            predicted_zone_fts = predicted_zone_fts[gt_zone_text_fts.sum(-1)!=0]
            gt_zone_text_fts = gt_zone_text_fts[gt_zone_text_fts.sum(-1)!=0]
            
            predicted_zone_fts = predicted_zone_fts / (torch.linalg.norm(predicted_zone_fts, dim=-1, keepdim=True) + 1e-5)
            gt_zone_text_fts = gt_zone_text_fts / (torch.linalg.norm(gt_zone_text_fts, dim=-1, keepdim=True) + 1e-5)
            loss += self.contrastive_loss(predicted_zone_fts, gt_zone_text_fts, logit_scale=10.) / 5.

            #predicted_fine_grained_zone_fts = pad_sequence(predicted_fine_grained_zone_fts).permute(1,0,2)
            #loss += self.fine_grained_contrastive_loss(predicted_fine_grained_zone_fts,gt_zone_fine_grained_text_fts) / 5.

        # Feature Fields Pre-training
        if len(sampled_gt_patch_language_label) > 0:
            sampled_predicted_patch_language_fts = torch.cat(sampled_predicted_patch_language_fts,dim=0).to(torch.float32)
            sampled_predicted_patch_language_fts = sampled_predicted_patch_language_fts / torch.linalg.norm(sampled_predicted_patch_language_fts, dim=-1, keepdim=True)
            logits = sampled_predicted_patch_language_fts @ self.category_embeddings.T * 10.
            target = torch.tensor(sampled_gt_patch_language_label, device=self.device)
            loss += self.focal_loss(logits,target) / 10.
        
        policy_net.feature_fields.delete_feature_fields()

        return loss
    


    def run_on_arkit(self, mode):
        loss = 0.
        policy_net = self.policy.net
        if hasattr(self.policy.net, 'module'):
            policy_net = self.policy.net.module

        batch_size = self.batch_size # Larger batch size is better, so different from batch_size of hm3d is also ok

        num_of_sampled_images = 16 # Number of sampled images to construct the feature fields
            
        batch_camera_intrinsic = []
        batch_grid_ft = []
        batch_image_ft = []
        batch_depth = []
        batch_rot = []
        batch_trans = []
        batch_extrinsic = []

        # Get the instance label
        batch_pcd_xyz = []
        batch_pcd_label = []
        batch_instance_id_to_label = []
        batch_scene_id = []
        batch_image = []

        gt_instance_text_fts = []
        gt_instance_category_id = []
        gt_zone_text_fts = []
        gt_zone_fine_grained_text_fts = []
        
        predicted_instance_fts = []
        predicted_zone_fts = []
        predicted_fine_grained_zone_fts = []

        sampled_gt_patch_fts, sampled_predicted_patch_fts = [], []
        sampled_gt_patch_language_label, sampled_predicted_patch_language_fts = [], []

        for batch_id in range(batch_size):

            scene_id = random.choice(list(set(self.arkit_pcd_with_global_alignment.keys()) & set(self.arkit_language_annotations.keys())))
            #file_path = self.arkit_scenes[scene_id]
            batch_scene_id.append(scene_id)

            # Get the instance label
            pcd_xyz = []
            pcd_label = []
            if scene_id in self.arkit_pcd_with_global_alignment:
                    
                pcd_file_list = self.arkit_pcd_with_global_alignment[scene_id]
                for pcd_file in pcd_file_list:
                    pcd_file = torch.load(pcd_file)
                    #align_matrix = torch.tensor(np.linalg.inv(self.scannet_align_matrix[scene_id])).to(self.device).to(torch.float32) # Align the coordinate using align_matrix
                    #pts = np.ones((pcd_file[0].shape[0], 4), dtype=np.float32)
                    pts = pcd_file[0]
                    #aligned_xyz = (torch.tensor(pts).to(self.device) @ align_matrix.T)[:, :3].cpu()
                    pcd_xyz.append(torch.tensor(pts))
                    pcd_label.append(torch.tensor(pcd_file[-1])) # 3 not 2, different from hm3d and structured

                batch_pcd_xyz.append(torch.cat(pcd_xyz,0))
                batch_pcd_label.append(torch.cat(pcd_label,0))

                instance_id_to_label_list = self.arkit_instance_id_to_label[scene_id]
                label_dict = {}
                for label_file in instance_id_to_label_list:
                    label_file = torch.load(label_file)
                    label_dict.update(label_file)
                batch_instance_id_to_label.append(label_dict)
            else:
                batch_pcd_xyz.append(None)
                batch_pcd_label.append(None)
                batch_instance_id_to_label.append(None)

            image_path = 'data/ARKitScenes/3dod/Training/'+scene_id+'/'+scene_id+'_frames/lowres_wide'
            image_list = os.listdir(image_path)
            image_list.sort()
            extrinsic_file = 'data/ARKitScenes/3dod/Training/'+scene_id+'/'+scene_id+'_frames/lowres_wide.traj'
            with open(extrinsic_file, 'r') as file:  
                extrinsic_list = [line.strip() for line in file]

            image_ids = [i for i in range(len(image_list))]
            random.shuffle(image_ids)

            image_ids = image_ids[:num_of_sampled_images]
            while len(image_ids) < num_of_sampled_images:
                image_ids += image_ids[:num_of_sampled_images-len(image_ids)] # Full number for num_of_sampled_images

            image_list = [image_path+'/'+image_list[i] for i in image_ids]
            extrinsic_list = [extrinsic_list[i] for i in image_ids]

            intrinsic_list = []
            R_list = []
            T_list = []
            rgb_list = []
            depth_list = []
            extrinsic_id = 0
            for image_path in image_list:
                intrinsic_file = 'data/ARKitScenes/3dod/Training/'+scene_id+'/'+scene_id+'_frames'+'/lowres_wide_intrinsics/' + image_path.split('/')[-1][:-4]+'.pincam'
                with open(intrinsic_file, 'r') as file:  
                    intrinsic_raw = [line.split() for line in file]
                intrinsic = self.st2_camera_intrinsics(intrinsic_raw[0])

                extrinsic = self.TrajStringToMatrix(extrinsic_list[extrinsic_id])
                R = extrinsic[:3,:3]
                T = extrinsic[:3,3:4]

                rgb_image = np.asarray(Image.open(image_path))
                depth_image = np.asarray(Image.open(image_path.replace("lowres_wide","lowres_depth")),dtype=np.float32)

                R_list.append(R)
                T_list.append(T)
                intrinsic_list.append(intrinsic)
                extrinsic_list[extrinsic_id] = np.linalg.inv(extrinsic)
                extrinsic_id += 1
                rgb_list.append(torch.tensor(rgb_image).unsqueeze(0))
                depth_list.append(torch.tensor(depth_image).unsqueeze(0))

            rgb_list = torch.cat(rgb_list,dim=0)
            depth_list = torch.cat(depth_list,dim=0)
            with torch.no_grad():
                img_fts, grid_fts = policy_net.rgb_encoder({'rgb':rgb_list})
                patch_fts = grid_fts.view(grid_fts.shape[0], 24,24,-1).permute(0,3,1,2)
                avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)
                patch_fts = avg_pool(patch_fts).permute(0,2,3,1)
                sampled_gt_patch_fts.append(patch_fts.view(num_of_sampled_images,-1,patch_fts.shape[-1]))


            batch_camera_intrinsic.append(np.stack(intrinsic_list,axis=0))
            batch_rot.append(np.stack(R_list,axis=0))
            batch_trans.append(np.stack(T_list,axis=0))
            batch_extrinsic.append(np.stack(extrinsic_list,axis=0))
            batch_grid_ft.append(grid_fts.cpu().numpy())
            batch_image.append(rgb_list)
            batch_image_ft.append(img_fts)
            batch_depth.append(depth_list)

        batch_camera_intrinsic = np.stack(batch_camera_intrinsic,axis=0)
        batch_rot = np.stack(batch_rot,axis=0)
        batch_trans = np.stack(batch_trans,axis=0)
        batch_extrinsic = np.stack(batch_extrinsic,axis=0)
        batch_image = torch.stack(batch_image,0).numpy()

        policy_net.feature_fields.reset(batch_size=batch_size, mode="scannet", batch_gt_pcd_xyz=batch_pcd_xyz, batch_gt_pcd_label=batch_pcd_label)  # Reset the settings of 3D feature fields
        # Do not change the order of the following two lines !!!!!
        policy_net.feature_fields.delete_old_features_from_camera_frustum(batch_depth=batch_depth, batch_camera_intrinsic=batch_camera_intrinsic, batch_extrinsic=batch_extrinsic)
        sim_loss, segm_loss, batch_gt_3d_instance_id,batch_predicted_3d_instancs_fts,batch_gt_3d_instance_ids_in_zone,batch_predicted_3d_zone_fts = policy_net.feature_fields.update_feature_fields(batch_depth, batch_grid_ft, batch_image, batch_image_ft=batch_image_ft, batch_camera_intrinsic=batch_camera_intrinsic, batch_rot=batch_rot, batch_trans=batch_trans, depth_scale=1000., depth_trunc=1000.)
        loss += sim_loss  # The quality of segmentation annotations is low, skip the segm loss !!!!!!!!!!!!

        # Predict the novel view features and the instance id within this view
        for image_idx in range(num_of_sampled_images):
            batch_rendered_patch_fts, batch_rendered_patch_positions, gt_label = policy_net.feature_fields.render_view_3d_patch(batch_camera_intrinsic=batch_camera_intrinsic[:,image_idx], batch_rot=batch_rot[:,image_idx], batch_trans=batch_trans[:,image_idx], visualization=False)
            sampled_predicted_patch_fts.append(batch_rendered_patch_fts.view(batch_size,1, -1,batch_rendered_patch_fts.shape[-1]))

            # Get language supervision of patch features
            for batch_id in range(batch_size):
                if gt_label[batch_id] != None and batch_instance_id_to_label[batch_id] != None:
                    instance_id_to_label = batch_instance_id_to_label[batch_id]
                    selected_mask = (gt_label[batch_id]>0)#.view(-1)
                    selected_gt_label = gt_label[batch_id][selected_mask]
                    if selected_gt_label.shape[0] == 0:
                        continue

                    selected_patch_fts = batch_rendered_patch_fts[batch_id][selected_mask]

                    for ray_id in range(selected_gt_label.shape[0]):
                        instance_id = int(selected_gt_label[ray_id].item())
                        category_id = instance_id_to_label[instance_id]
                        category_id = category_id.replace("_"," ").replace("|"," ").replace("/"," ")
                        if category_id in self.category_dict:
                            category_embedding_index = self.category_dict[category_id]
                            sampled_gt_patch_language_label.append(category_embedding_index)
                            sampled_predicted_patch_language_fts.append(selected_patch_fts[ray_id:ray_id+1])

        sampled_predicted_patch_fts = [torch.cat(sampled_predicted_patch_fts,dim=1).view(-1,sampled_predicted_patch_fts[0].shape[-2],sampled_predicted_patch_fts[0].shape[-1])] # Keep the correct order of patch features

        # Calculate the text alignment loss
        for b in range(batch_size):
            scene_id = batch_scene_id[b]
            instance_ids = batch_gt_3d_instance_id[b].cpu().numpy().tolist()
            zones = batch_gt_3d_instance_ids_in_zone[b]

            # Store the predicted features
            predicted_instance_fts.append(batch_predicted_3d_instancs_fts[b])

            predicted_zone_fts.extend([tmp_fts[0:1] for tmp_fts in batch_predicted_3d_zone_fts[b]])
            predicted_fine_grained_zone_fts.extend([tmp_fts[1:] for tmp_fts in batch_predicted_3d_zone_fts[b]])

            if scene_id in self.arkit_language_annotations:
                # Instance alignment
                for instance_id in instance_ids:
                    if str(instance_id) not in self.arkit_language_annotations[scene_id]:
                        if instance_id in batch_instance_id_to_label[b]:
                            category_id = batch_instance_id_to_label[b][instance_id].replace("_"," ").replace("|"," ").replace("/"," ")
                            category_embedding_index = self.category_dict[category_id]
                            gt_instance_text_fts.append(self.category_embeddings[category_embedding_index:category_embedding_index+1])
                            gt_instance_category_id.append(category_embedding_index)
                        else:
                            gt_instance_text_fts.append(torch.zeros((1,768),device=self.device))
                            gt_instance_category_id.append(-1)
                    else:
                        object_category, object_text = random.choice(self.arkit_language_annotations[scene_id][str(instance_id)])
                        with torch.no_grad():
                            text_ids = policy_net.rgb_encoder.tokenize(object_text).to(policy_net.rgb_encoder.device)
                            text_fts = policy_net.rgb_encoder.model.encode_text(text_ids)
                            gt_instance_text_fts.append(text_fts)

                            category_id = object_category.replace("_"," ").replace("|"," ").replace("/"," ")
                            if category_id in self.category_dict:
                                category_embedding_index = self.category_dict[category_id]
                                gt_instance_category_id.append(category_embedding_index)
                            else:
                                gt_instance_category_id.append(-1)

                # Zone alignment
                for instance_ids_in_zone in zones:
                    if len(instance_ids_in_zone) > 0:
                        instance_ids_in_zone = list(set(instance_ids_in_zone.cpu().numpy().tolist()))
                        random.shuffle(instance_ids_in_zone)
                        for index, instance_id_in_zone in enumerate(instance_ids_in_zone):
                            if str(instance_id_in_zone) in self.arkit_language_annotations[scene_id]:
                                object_category, object_text = random.choice(self.arkit_language_annotations[scene_id][str(instance_id_in_zone)])
                                with torch.no_grad():
                                    text_ids = policy_net.rgb_encoder.tokenize(object_text).to(policy_net.rgb_encoder.device)
                                    text_fts, sep_fts = policy_net.rgb_encoder.model.encode_all_text(text_ids)
                                    gt_zone_text_fts.append(sep_fts)
                                    gt_zone_fine_grained_text_fts.append(text_fts)
                                break
                            
                            if index == len(instance_ids_in_zone) - 1:
                                if instance_id_in_zone in batch_instance_id_to_label[b]:
                                    category_id = batch_instance_id_to_label[b][instance_id_in_zone]
                                    category_embedding_index = self.category_dict[category_id]
                                    gt_zone_text_fts.append(self.category_embeddings[category_embedding_index:category_embedding_index+1])
                                    gt_zone_fine_grained_text_fts.append(self.category_embeddings[category_embedding_index:category_embedding_index+1])
                                else:
                                    gt_zone_text_fts.append(torch.zeros((1,768),device=self.device))
                                    gt_zone_fine_grained_text_fts.append(torch.zeros((1,768),device=self.device))
                    else:
                        gt_zone_text_fts.append(torch.zeros((1,768),device=self.device))
                        gt_zone_fine_grained_text_fts.append(torch.zeros((1,768),device=self.device))


        # Feature Fields Pre-training
        sampled_gt_patch_fts = torch.cat(sampled_gt_patch_fts,dim=0).to(torch.float32)
        sampled_predicted_patch_fts = torch.cat(sampled_predicted_patch_fts,dim=0).to(torch.float32)

        subspace_sampled_gt_patch_fts = sampled_gt_patch_fts - sampled_gt_patch_fts.mean(1, keepdim=True)
        subspace_sampled_predicted_patch_fts = sampled_predicted_patch_fts - sampled_predicted_patch_fts.mean(1, keepdim=True)

        subspace_sampled_predicted_patch_fts = subspace_sampled_predicted_patch_fts / (torch.linalg.norm(subspace_sampled_predicted_patch_fts, dim=-1, keepdim=True) + 1e-5)
        subspace_sampled_gt_patch_fts = subspace_sampled_gt_patch_fts / (torch.linalg.norm(subspace_sampled_gt_patch_fts, dim=-1, keepdim=True) + 1e-5)
        loss += (1. - (subspace_sampled_predicted_patch_fts * subspace_sampled_gt_patch_fts).sum(-1)).mean() * 2.

        sampled_predicted_patch_fts = sampled_predicted_patch_fts / (torch.linalg.norm(sampled_predicted_patch_fts, dim=-1, keepdim=True) + 1e-5)
        sampled_gt_patch_fts = sampled_gt_patch_fts / (torch.linalg.norm(sampled_gt_patch_fts, dim=-1, keepdim=True) + 1e-5)

        sampled_predicted_patch_fts = sampled_predicted_patch_fts.view(-1,sampled_predicted_patch_fts.shape[-1])
        sampled_gt_patch_fts = sampled_gt_patch_fts.view(-1,sampled_gt_patch_fts.shape[-1])


        loss += (1. - (sampled_predicted_patch_fts * sampled_gt_patch_fts).sum(-1)).mean() * 5.
        loss += self.contrastive_loss(sampled_predicted_patch_fts, sampled_gt_patch_fts, logit_scale=10.) / 5.

        
        if len(gt_instance_text_fts) != 0 and torch.cat(gt_instance_text_fts,dim=0).sum().cpu().numpy().item() != 0:
            gt_instance_text_fts = torch.cat(gt_instance_text_fts,dim=0).to(torch.float32)
            gt_zone_text_fts = torch.cat(gt_zone_text_fts,dim=0).to(torch.float32)
            predicted_instance_fts = torch.cat(predicted_instance_fts,dim=0).to(torch.float32)
            predicted_zone_fts = torch.cat(predicted_zone_fts,dim=0).to(torch.float32)

            predicted_instance_fts = predicted_instance_fts / (torch.linalg.norm(predicted_instance_fts, dim=-1, keepdim=True) + 1e-5)
            gt_instance_text_fts = gt_instance_text_fts / (torch.linalg.norm(gt_instance_text_fts, dim=-1, keepdim=True) + 1e-5)
            
            target = torch.tensor(gt_instance_category_id, device=self.device)
            logits = predicted_instance_fts[target!=-1] @ self.category_embeddings.T * 10.
            target = target[target!=-1]
            loss +=  F.cross_entropy(logits, target) / 10.

            predicted_instance_fts = predicted_instance_fts[gt_instance_text_fts.sum(-1)!=0]
            gt_instance_text_fts = gt_instance_text_fts[gt_instance_text_fts.sum(-1)!=0]
            loss += self.contrastive_loss(predicted_instance_fts, gt_instance_text_fts, logit_scale=10.) / 5.

            predicted_zone_fts = predicted_zone_fts[gt_zone_text_fts.sum(-1)!=0]
            gt_zone_text_fts = gt_zone_text_fts[gt_zone_text_fts.sum(-1)!=0]
            
            predicted_zone_fts = predicted_zone_fts / (torch.linalg.norm(predicted_zone_fts, dim=-1, keepdim=True) + 1e-5)
            gt_zone_text_fts = gt_zone_text_fts / (torch.linalg.norm(gt_zone_text_fts, dim=-1, keepdim=True) + 1e-5)
            loss += self.contrastive_loss(predicted_zone_fts, gt_zone_text_fts, logit_scale=10.) / 5.

            #predicted_fine_grained_zone_fts = pad_sequence(predicted_fine_grained_zone_fts).permute(1,0,2)
            #loss += self.fine_grained_contrastive_loss(predicted_fine_grained_zone_fts,gt_zone_fine_grained_text_fts) / 5.

        # Feature Fields Pre-training
        if len(sampled_gt_patch_language_label) > 0:
            sampled_predicted_patch_language_fts = torch.cat(sampled_predicted_patch_language_fts,dim=0).to(torch.float32)
            sampled_predicted_patch_language_fts = sampled_predicted_patch_language_fts / torch.linalg.norm(sampled_predicted_patch_language_fts, dim=-1, keepdim=True)
            logits = sampled_predicted_patch_language_fts @ self.category_embeddings.T * 10.
            target = torch.tensor(sampled_gt_patch_language_label, device=self.device)
            loss += self.focal_loss(logits,target) / 10.
        
        policy_net.feature_fields.delete_feature_fields()

        return loss




    def run_on_structured3d(self, mode):
        loss = 0.
        policy_net = self.policy.net
        if hasattr(self.policy.net, 'module'):
            policy_net = self.policy.net.module

        batch_size = self.batch_size # Larger batch size is better, so different from batch_size of hm3d is also ok
        num_of_sampled_images = 16

        batch_camera_intrinsic = []
        batch_grid_ft = []
        batch_image_ft = []
        batch_depth = []
        batch_rot = []
        batch_trans = []
        batch_extrinsic = []


        # Get the instance label
        batch_pcd_xyz = []
        batch_pcd_label = []
        batch_instance_id_to_label = []
        batch_scene_id = []
        batch_image = []

        sampled_gt_patch_fts, sampled_predicted_patch_fts = [], []
        sampled_gt_patch_language_label, sampled_predicted_patch_language_fts = [], []

        for batch_id in range(batch_size):
            scene_id = random.choice(list(self.structure_3d_scenes.keys()))
            file_path = self.structure_3d_scenes[scene_id]
            '''
            if task_id == 0:
                scene_id = random.choice(list(self.structure_3d_scenes.keys()))
                file_path = self.structure_3d_scenes[scene_id]
            else:
                scene_id = random.choice(list(set(self.Structured3D_pcd_with_global_alignment.keys()) & set(self.Structured3D_language_annotations.keys())))
                file_path = self.structure_3d_scenes[scene_id]
            '''
            batch_scene_id.append(scene_id)

            '''
            # Get the instance label
            pcd_xyz = []
            pcd_label = []
            try:
                if scene_id in self.Structured3D_pcd_with_global_alignment:
                    
                    pcd_file_list = self.Structured3D_pcd_with_global_alignment[scene_id]
                    for pcd_file in pcd_file_list:
                        pcd_file = torch.load(pcd_file)
                        pcd_xyz.append(torch.tensor(pcd_file[0]))
                        pcd_label.append(torch.tensor(pcd_file[2]))
                    batch_pcd_xyz.append(torch.cat(pcd_xyz,0))
                    batch_pcd_label.append(torch.cat(pcd_label,0))

                    instance_id_to_label_list = self.Structured3D_instance_id_to_label[scene_id]
                    label_dict = {}
                    for label_file in instance_id_to_label_list:
                        label_file = torch.load(label_file)
                        label_dict.update(label_file)
                    batch_instance_id_to_label.append(label_dict)
                else:
                    batch_pcd_xyz.append(None)
                    batch_pcd_label.append(None)
                    batch_instance_id_to_label.append(None)
            except:
                batch_pcd_xyz.append(None)
                batch_pcd_label.append(None)
                batch_instance_id_to_label.append(None)
                print("File of Structured3D_pcd_with_global_alignment or Structured3D_instance_id_to_label has error, skip...")
            '''

            room_list = [file_path+'/'+item+'/perspective/full' for item in os.listdir(file_path)]
            image_list = []
            for i in room_list:
                if os.path.exists(i):
                    for j in os.listdir(i):
                        image_list.append(i+'/'+j)
            if len(image_list) == 0:
                print("Miss",file_path,"!!!!!!!!!!!!")
            
            random.shuffle(image_list)
            image_list = image_list[:num_of_sampled_images]

            while len(image_list) < num_of_sampled_images:
                image_list += image_list[:num_of_sampled_images-len(image_list)] # Full number for num_of_sampled_images

            intrinsic_list = []
            R_list = []
            T_list = []
            extrinsic_list = []
            rgb_list = []
            depth_list = []
            
            for image_id in image_list:
                camera_info = np.loadtxt(os.path.join(image_id, 'camera_pose.txt'))
                rot, trans, intrinsic =self.parse_camera_info(camera_info,720, 1280)
                intrinsic_list.append(intrinsic)
                extrinsic = np.eye(4)
                extrinsic[:3,:3] = rot
                extrinsic = np.linalg.inv(extrinsic)
                R = extrinsic[:3,:3]
                T = trans.reshape(3,1)
                rgb_image = Image.open(image_id + '/rgb_rawlight.png').convert('RGB')
                depth_image = Image.open(image_id + '/depth.png')

                R_list.append(R)
                T_list.append(T)
                extrinsic_list.append(extrinsic)

                rgb_list.append(torch.tensor(np.asarray(rgb_image)).unsqueeze(0))
                depth_list.append(torch.tensor(np.asarray(depth_image)).unsqueeze(0))

            if len(rgb_list)==0:
                print("Miss", image_list,"!!!!!!!!!!!!")

            rgb_list = torch.cat(rgb_list,dim=0)
            depth_list = torch.cat(depth_list,dim=0)
            with torch.no_grad():
                img_fts, grid_fts = policy_net.rgb_encoder({'rgb':rgb_list})
                patch_fts = grid_fts.view(grid_fts.shape[0], 24,24,-1).permute(0,3,1,2)
                avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)
                patch_fts = avg_pool(patch_fts).permute(0,2,3,1)
                sampled_gt_patch_fts.append(patch_fts.view(num_of_sampled_images,-1,patch_fts.shape[-1]))


            batch_camera_intrinsic.append(np.stack(intrinsic_list,axis=0))
            batch_rot.append(np.stack(R_list,axis=0))
            batch_trans.append(np.stack(T_list,axis=0))
            batch_extrinsic.append(np.stack(extrinsic_list,axis=0))
            batch_grid_ft.append(grid_fts.cpu().numpy())
            batch_image.append(rgb_list)
            batch_image_ft.append(img_fts)
            batch_depth.append(depth_list)

        batch_camera_intrinsic = np.stack(batch_camera_intrinsic,axis=0)
        batch_rot = np.stack(batch_rot,axis=0)
        batch_trans = np.stack(batch_trans,axis=0)
        batch_extrinsic = np.stack(batch_extrinsic,axis=0)
        batch_image = torch.stack(batch_image,0).numpy()

        
        policy_net.feature_fields.reset(batch_size=batch_size, mode="scannet")  # Reset the settings of 3D feature fields
        # Do not change the order of the following two lines !!!!!
        policy_net.feature_fields.delete_old_features_from_camera_frustum(batch_depth=batch_depth, batch_camera_intrinsic=batch_camera_intrinsic, batch_extrinsic=batch_extrinsic)
        sim_loss, segm_loss, batch_gt_3d_instance_id,batch_predicted_3d_instancs_fts,batch_gt_3d_instance_ids_in_zone,batch_predicted_3d_zone_fts = policy_net.feature_fields.update_feature_fields(batch_depth, batch_grid_ft, batch_image, batch_image_ft=batch_image_ft, batch_camera_intrinsic=batch_camera_intrinsic, batch_rot=batch_rot, batch_trans=batch_trans, depth_scale=1000., depth_trunc=1000.)
        loss += sim_loss  # The quality of segmentation annotations is low, skip the segm loss !!!!!!!!!!!!


        # Predict the novel view features and the instance id within this view
        for image_idx in range(num_of_sampled_images):
            batch_rendered_patch_fts, batch_rendered_patch_positions, gt_label = policy_net.feature_fields.render_view_3d_patch(batch_camera_intrinsic=batch_camera_intrinsic[:,image_idx], batch_rot=batch_rot[:,image_idx], batch_trans=batch_trans[:,image_idx], visualization=False)
            sampled_predicted_patch_fts.append(batch_rendered_patch_fts.view(batch_size,1, -1,batch_rendered_patch_fts.shape[-1]))

        sampled_predicted_patch_fts = [torch.cat(sampled_predicted_patch_fts,dim=1).view(-1,sampled_predicted_patch_fts[0].shape[-2],sampled_predicted_patch_fts[0].shape[-1])] # Keep the correct order of patch features


        # Feature Fields Pre-training
        sampled_gt_patch_fts = torch.cat(sampled_gt_patch_fts,dim=0).to(torch.float32)
        sampled_predicted_patch_fts = torch.cat(sampled_predicted_patch_fts,dim=0).to(torch.float32)

        subspace_sampled_gt_patch_fts = sampled_gt_patch_fts - sampled_gt_patch_fts.mean(1, keepdim=True)
        subspace_sampled_predicted_patch_fts = sampled_predicted_patch_fts - sampled_predicted_patch_fts.mean(1, keepdim=True)

        subspace_sampled_predicted_patch_fts = subspace_sampled_predicted_patch_fts / (torch.linalg.norm(subspace_sampled_predicted_patch_fts, dim=-1, keepdim=True) + 1e-5)
        subspace_sampled_gt_patch_fts = subspace_sampled_gt_patch_fts / (torch.linalg.norm(subspace_sampled_gt_patch_fts, dim=-1, keepdim=True) + 1e-5)
        loss += (1. - (subspace_sampled_predicted_patch_fts * subspace_sampled_gt_patch_fts).sum(-1)).mean() * 2.

        sampled_predicted_patch_fts = sampled_predicted_patch_fts / (torch.linalg.norm(sampled_predicted_patch_fts, dim=-1, keepdim=True) + 1e-5)
        sampled_gt_patch_fts = sampled_gt_patch_fts / (torch.linalg.norm(sampled_gt_patch_fts, dim=-1, keepdim=True) + 1e-5)

        sampled_predicted_patch_fts = sampled_predicted_patch_fts.view(-1,sampled_predicted_patch_fts.shape[-1])
        sampled_gt_patch_fts = sampled_gt_patch_fts.view(-1,sampled_gt_patch_fts.shape[-1])


        loss += (1. - (sampled_predicted_patch_fts * sampled_gt_patch_fts).sum(-1)).mean() * 5.
        loss += self.contrastive_loss(sampled_predicted_patch_fts, sampled_gt_patch_fts, logit_scale=10.) / 5.


        policy_net.feature_fields.delete_feature_fields()

        return loss
    


    def rollout(self, mode, ml_weight=None):
        loss = 0.     
        dataset_id = random.randint(0,4)
        dataset_id = torch.tensor(dataset_id,device=self.device)
        if self.world_size > 1: # sync the dataset_id for all gpus
            distr.broadcast(dataset_id, src=0)

        dataset_id = dataset_id.cpu().numpy().item()
        if dataset_id == 0: # Run on HM3D, MP3D
            loss = self.run_on_hm3d(mode)

        elif dataset_id == 1: # Run on ScanNet
            loss = self.run_on_scannet(mode)

        elif dataset_id == 2: # Run on 3RScan
            loss = self.run_on_3rscan(mode)

        elif dataset_id == 3: # Run on ARKit, but the segmention annotation is low quality, please remove the segmention training
            loss = self.run_on_arkit(mode)

        elif dataset_id == 4: # Run on Structured3D, but the segmention annotation is low quality, please remove the segmention training
            loss = self.run_on_structured3d(mode)

        print("Task id:",dataset_id,loss)
        if mode == 'train':
            loss = ml_weight * loss
            self.loss += loss

            if self.loss !=0 and torch.any(torch.isnan(self.loss)):
                print("loss is NaN, skip this step...")
                return 0.
            try:
                self.logs['IL_loss'].append(loss.item())
            except:
                self.logs['IL_loss'].append(loss)
