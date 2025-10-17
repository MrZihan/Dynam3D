import torch
#torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from torch_kdtree import build_kd_tree
import open3d as o3d
from joblib import Parallel
from einops import einsum
try:
    from apex.normalization.fused_layer_norm import FusedLayerNorm as BertLayerNorm
except (ImportError, AttributeError) as e:
    # logger.info("Better speed can be achieved with apex installed from https://www.github.com/nvidia/apex .")
    BertLayerNorm = torch.nn.LayerNorm

from vlnce_baselines.models.fastsam import FastSAM, FastSAMPrompt
import os


# Model Settings
def config_parser():

    import configargparse
    parser = configargparse.ArgumentParser()

    # Settings for the habitat simulator
    parser.add_argument("--input_hfov", type=float, default=90., 
                        help='hfov angle of input view')
    parser.add_argument("--input_vfov", type=float, default=90., 
                    help='vfov angle of input view')
    parser.add_argument("--input_height", type=int, default=24, 
                        help='height of the input view')
    parser.add_argument("--input_width", type=int, default=24, 
                        help='width of the input view')
    parser.add_argument("--fts_dim", type=float, default=768)
    
    parser.add_argument("--zone_x_length", type=float, default=2.)
    parser.add_argument("--zone_y_length", type=float, default=2.)
    parser.add_argument("--zone_z_length", type=float, default=2.)

    parser.add_argument("--deleted_frustum_distance", type=float, default=3.)

    parser.add_argument("--num_proposal_instances", type=int, default=2)

    return parser



def project_depth_to_3d(depth, intrinsic, depth_scale, depth_trunc, input_height, input_width): # Don't define in the Feature_Fields Class, to avoid the memory copy of entire Feature_Fields Class at each thread when parallel computing
    depth[depth==0] = 1 # filter out the noise
    o3d_depth = o3d.geometry.Image(depth.numpy().astype(np.uint16))
    pcd = o3d.geometry.PointCloud.create_from_depth_image(o3d_depth, o3d.camera.PinholeCameraIntrinsic(depth.shape[1],depth.shape[0],intrinsic[0][0],intrinsic[1][1],intrinsic[0][2],intrinsic[1][2]), depth_scale=depth_scale, depth_trunc=depth_trunc)
    points = np.asarray(pcd.points)
    points = torch.tensor(points).view(depth.shape[0],depth.shape[1],3).permute(2,0,1).unsqueeze(0) # num_of_samples x channels x height x width
    points = F.interpolate(points, size=(input_height,input_width), scale_factor=None, mode='nearest').squeeze(0).permute(1,2,0).view(-1,3).numpy()

    points_mask = points[:,2] > 0.002 # filter out the noise

    return (points, points_mask)



@torch.no_grad()
def get_frustum_mask(points, H, W, intrinsics, view_matrices, near = 0., far = 2.):

    intrinsics = intrinsics[:3,:3]
    ones = torch.ones_like(points[:, 0]).unsqueeze(-1).to(points.device)
    homo_points = torch.cat([points, ones], dim=-1)

    view_points = einsum(view_matrices, homo_points, "b c, N c -> N b")
    view_points = view_points[:, :3]

    uv_points = einsum(intrinsics, view_points, "b c, N c -> N b")

    z = uv_points[:, -1:]
    uv_points = uv_points[:, :2] / z
    u, v = uv_points[:, 0].to(torch.int64), uv_points[:, 1].to(torch.int64) # !!!!!!!!!!!
    depth = view_points[:, -1]
 
    cull_near_fars = (depth >= near) & (depth <= far)

    mask = cull_near_fars & (u >= 0) & (u <= W-1) & (v >= 0) & (v <= H-1)
    return mask, depth, u, v



@torch.no_grad()
def get_frustum_mask_habitat(points, H, W, vfov, hfov, camera_position, heading_angle, near = 0., far = 2.):
    intrinsics = torch.tensor(np.array([
            [W / np.tan(np.deg2rad(hfov) / 2.) / 2., 0., W/2.],
            [0., H / np.tan(np.deg2rad(vfov) / 2.) / 2., H/2.],
            [0., 0.,  1]]),dtype=torch.float32).to(points.device)

    heading_angle = - heading_angle
    points_x, points_y, points_z = points[:,0:1] - camera_position[0], points[:,1:2] - camera_position[1], points[:,2:3] - camera_position[2]

    rel_x = points_x * math.cos(heading_angle) - points_y * math.sin(heading_angle)
    rel_y = points_x * math.sin(heading_angle) + points_y * math.cos(heading_angle)
    rel_z = points_z
    
    rel_x, rel_y, rel_z = rel_x, - rel_z, rel_y

    view_points = torch.cat([rel_x,rel_y,rel_z],dim=-1)

    uv_points = einsum(intrinsics, view_points, "b c, N c -> N b")

    z = uv_points[:, -1:]
    uv_points = uv_points[:, :2] / z
    u, v = uv_points[:, 0].to(torch.int64), uv_points[:, 1].to(torch.int64) # !!!!!!!!!!!
    depth = view_points[:, -1]

    cull_near_fars = (depth >= near) & (depth <= far)
    mask = cull_near_fars & (u >= 0) & (u <= W-1) & (v >= 0) & (v <= H-1)
    return mask, depth, u, v



class Feature_Fields(nn.Module):
    def __init__(self, batch_size=1, device='cuda'):
        super(Feature_Fields, self).__init__()
        """
        Instantiate Feature Fields model.
        """
        self.device = device
        parser = config_parser()
        args, unknown = parser.parse_known_args()
        self.args = args

        self.thread_pool = Parallel(n_jobs=8,backend='threading') # Parallel computing with multiple CPUs, default is 8
        
        width = self.args.fts_dim
        scale = width ** -0.5
        enc_layer = nn.TransformerEncoderLayer(
            d_model=width, nhead=width//64, dim_feedforward=4*width, dropout=0.1,
                 activation="gelu", batch_first=True
        )

        self.patch_to_instance_position_embedding = nn.Sequential(
            nn.Linear(7, width),
            nn.LayerNorm(width),
            nn.GELU(),
            nn.Linear(width, width))
        
        self.aggregate_patch_to_instance_embedding = nn.Parameter(scale * torch.randn(1,width))
        self.aggregate_patch_to_instance_encoder = nn.TransformerEncoder(enc_layer, num_layers=2, norm=BertLayerNorm(width, eps=1e-12))

        self.instance_to_zone_position_embedding = nn.Sequential(
            nn.Linear(4, width),
            nn.LayerNorm(width),
            nn.GELU(),
            nn.Linear(width, width))
        
        self.aggregate_instance_to_zone_embedding = nn.Parameter(scale * torch.randn(1,width))
        self.aggregate_instance_to_zone_encoder = nn.TransformerEncoder(enc_layer, num_layers=2, norm=BertLayerNorm(width, eps=1e-12))
     
        self.instance_merge_discriminator = nn.Sequential(
            nn.Linear(2*width+3, 4*width),
            nn.LayerNorm(4*width),
            nn.GELU(),
            nn.Linear(4*width, 2))

        self.batch_size = batch_size
        self.global_patch_fts = [[] for i in range(self.batch_size)]
        self.global_patch_position = [[] for i in range(self.batch_size)]
        self.global_patch_scales = [[] for i in range(self.batch_size)]
        self.global_patch_directions = [[] for i in range(self.batch_size)]
        self.global_patch_to_instance_dict = [{} for i in range(self.batch_size)]

        self.global_instance_fts = [[] for i in range(self.batch_size)]
        self.global_instance_position = [[] for i in range(self.batch_size)]
        self.global_instance_to_patch_dict = [{} for i in range(self.batch_size)]

        self.global_zone_fts = [[] for i in range(self.batch_size)]
        self.global_zone_position = [[] for i in range(self.batch_size)]
        self.global_zone_key_to_id = [{} for i in range(self.batch_size)]
        self.global_zone_to_instance_dict = [{} for i in range(self.batch_size)]

        self.instance_tree = [[] for i in range(self.batch_size)]
        
        self.FastSAM = FastSAM('FastSAM.pt')
        self.keep_target_waypoint = [None for b in range(self.batch_size)]
        self.history_actions = [["none\n"]*4] * self.batch_size


    def reset(self, batch_size=1):
        self.batch_size = batch_size
        self.global_patch_fts = [[] for i in range(self.batch_size)]
        self.global_patch_position = [[] for i in range(self.batch_size)]
        self.global_patch_scales = [[] for i in range(self.batch_size)]
        self.global_patch_directions = [[] for i in range(self.batch_size)]
        self.global_patch_to_instance_dict = [{} for i in range(self.batch_size)]

        self.global_instance_fts = [[] for i in range(self.batch_size)]
        self.global_instance_position = [[] for i in range(self.batch_size)]
        self.global_instance_to_patch_dict = [{} for i in range(self.batch_size)]

        self.global_zone_fts = [[] for i in range(self.batch_size)]
        self.global_zone_position = [[] for i in range(self.batch_size)]
        self.global_zone_key_to_id = [{} for i in range(self.batch_size)]
        self.global_zone_to_instance_dict = [{} for i in range(self.batch_size)]

        self.instance_tree = [[] for i in range(self.batch_size)]

        self.keep_target_waypoint = [None for b in range(self.batch_size)]
        self.history_actions = [["none\n"]*4] * self.batch_size

        

    def pop(self, index):
        self.batch_size -= 1
        self.global_patch_fts.pop(index)
        self.global_patch_position.pop(index)
        self.global_patch_scales.pop(index)
        self.global_patch_directions.pop(index)
        self.global_patch_to_instance_dict.pop(index)

        self.global_instance_fts.pop(index)
        self.global_instance_position.pop(index)
        self.global_instance_to_patch_dict.pop(index)

        self.global_zone_fts.pop(index)
        self.global_zone_position.pop(index)
        self.global_zone_key_to_id.pop(index)
        self.global_zone_to_instance_dict.pop(index)

        self.instance_tree.pop(index)
        self.keep_target_waypoint.pop(index)
        self.history_actions.pop(index)

    def initialize_camera_setting(self, hfov, vfov):
        self.args.input_hfov = hfov
        self.args.input_vfov = vfov


    def delete_feature_fields(self): # Free the memory
        del self.global_patch_fts, self.global_patch_position, self.global_patch_scales, self.global_patch_directions, \
            self.global_patch_to_instance_dict, self.global_instance_fts, self.global_instance_position, self.global_instance_to_patch_dict, \
            self.global_zone_fts, self.global_zone_position, self.global_zone_key_to_id, self.global_zone_to_instance_dict, \
            self.instance_tree, self.keep_target_waypoint, self.history_actions


    def get_instance_tree(self, batch_id):
        if len(self.global_instance_position[batch_id]) == 0:
            return []
        instance_tree = build_kd_tree(self.global_instance_position[batch_id])
        return instance_tree


    def get_heading_angle(self, position):      
        dx = position[:,0]
        dy = position[:,1]
        dz = position[:,2]
        xy_dist = np.sqrt(np.square(dx) + np.square(dy))
        xy_dist[xy_dist < 1e-4] = 1e-4
        # the simulator's api is weired (x-y axis is transposed)
        heading_angle = - np.arcsin(dx/xy_dist) # [-pi/2, pi/2]
        heading_angle[dy < 0] =  heading_angle[dy < 0] - np.pi
        return heading_angle


    def get_rays(self, camera_intrinsic): # Originally for rendering of feature fields, here for calculating the patch information
        sampled_points = o3d.geometry.PointCloud()
        N_distance = self.args.deleted_frustum_distance
        N_depth = np.full((self.args.input_height,self.args.input_width),N_distance,dtype=np.float32)
        N_depth = o3d.geometry.Image(N_depth)
        N_points = o3d.geometry.PointCloud.create_from_depth_image(N_depth, o3d.camera.PinholeCameraIntrinsic(self.args.input_height,self.args.input_width, camera_intrinsic[0][0],camera_intrinsic[1][1],self.args.input_width/2,self.args.input_height/2), depth_scale=1., depth_trunc=1.)
        sampled_points += N_points
        sampled_points = np.asarray(sampled_points.points)
        rel_position = sampled_points.reshape((1, self.args.input_height*self.args.input_width,3)).transpose((1,0,2))
        rel_direction = - np.arctan(rel_position[...,-1:,0]/rel_position[...,-1:,2])
        rel_dist = rel_position[...,2]
        return (rel_position, rel_direction, rel_dist)
    

    def project_depth_to_3d_habitat(self, depth_map, heading_angle):
        W = self.args.input_width
        H = self.args.input_height
        half_W = W//2
        half_H = H//2
        depth_y = depth_map.astype(np.float32) # / 4000.

        tan_xy = np.array(([i/half_W+1/W for i in range(-half_W,half_W)])*H,np.float32) * math.tan(math.pi * self.args.input_hfov/360.)
        direction = - np.arctan(tan_xy)
        depth_x = depth_y * tan_xy
        depth_z = depth_y * (np.array([[i/half_H-1/H for i in range(half_H,-half_H,-1)]]*W,np.float32).T.reshape((-1,)) * math.tan(math.pi * self.args.input_vfov/360.))
        scale = depth_y * math.tan(math.pi * self.args.input_hfov/360.) * 2. / W

        direction = (direction+heading_angle) % (2*math.pi)
        rel_x = depth_x * math.cos(heading_angle) - depth_y * math.sin(heading_angle)
        rel_y = depth_x * math.sin(heading_angle) + depth_y * math.cos(heading_angle)
        rel_z = depth_z
        return rel_x, rel_y, rel_z, direction.reshape(-1), scale.reshape(-1)


    def get_patch_3d_info(self, batch_depth_map):
        batch_size = len(batch_depth_map)
        batch_rel_x, batch_rel_y, batch_rel_z, batch_direction, batch_scale = [],[],[],[],[]
        for b in range(batch_size):
            depth_map = batch_depth_map[b]
            W = self.args.input_width
            H = self.args.input_height
            half_W = W//2
            half_H = H//2
            depth_y = depth_map.astype(np.float32) # / 4000.

            tan_xy = np.array(([i/half_W+1/W for i in range(-half_W,half_W)])*H,np.float32) * math.tan(math.pi * self.args.input_hfov/360.)
            direction = - np.arctan(tan_xy)
            depth_x = depth_y * tan_xy
            depth_z = depth_y * (np.array([[i/half_H-1/H for i in range(half_H,-half_H,-1)]]*W,np.float32).T.reshape((-1,)) * math.tan(math.pi * self.args.input_vfov/360.))
            scale = depth_y * math.tan(math.pi * self.args.input_hfov/360.) * 2. / W

            direction = direction % (2*math.pi)

            batch_rel_x.append(np.expand_dims(depth_x,axis=0))
            batch_rel_y.append(np.expand_dims(depth_y,axis=0))
            batch_rel_z.append(np.expand_dims(depth_z,axis=0))
            batch_direction.append(np.expand_dims(direction.reshape(-1),axis=0))
            batch_scale.append(np.expand_dims(scale.reshape(-1),axis=0))

        batch_rel_x = torch.tensor(np.concatenate(batch_rel_x,axis=0),device=self.device).unsqueeze(-1)
        batch_rel_y = torch.tensor(np.concatenate(batch_rel_y,axis=0),device=self.device).unsqueeze(-1)
        batch_rel_z = torch.tensor(np.concatenate(batch_rel_z,axis=0),device=self.device).unsqueeze(-1)
        batch_direction = torch.tensor(np.concatenate(batch_direction,axis=0),device=self.device).unsqueeze(-1)
        batch_scale = torch.tensor(np.concatenate(batch_scale,axis=0),device=self.device).unsqueeze(-1)
        return batch_rel_x, batch_rel_y, batch_rel_z, batch_direction, batch_scale
            

    def delete_old_features_from_camera_frustum(self, batch_depth, batch_position=None, batch_heading=None, batch_camera_intrinsic=None, batch_extrinsic=None, num_of_views=1):
        zone_x_length, zone_y_length, zone_z_length = self.args.zone_x_length, self.args.zone_y_length, self.args.zone_z_length
        for b in range(self.batch_size): # Use prange for parallel
            if batch_extrinsic != None:
                num_of_views = batch_depth[b].shape[0]
            if batch_position != None:
                position = batch_position[b].copy()
                position[0], position[1], position[2] = batch_position[b][0], - batch_position[b][2], batch_position[b][1] # Note to swap y,z axis, - y
                heading_angle = batch_heading[b]

            for ix in range(num_of_views): # 1 or 12 views, rotation for panorama
                if len(self.global_patch_position[b])==0:
                    continue

                if batch_extrinsic != None:
                    frustum_mask, frustum_depth, u, v = get_frustum_mask(torch.tensor(self.global_patch_position[b],dtype=torch.float32,device=self.device), batch_depth[b][ix].shape[-2], batch_depth[b][ix].shape[-1], batch_camera_intrinsic[b][ix], batch_extrinsic[b][ix])

                if batch_position != None:
                    frustum_mask, frustum_depth, u, v = get_frustum_mask_habitat(torch.tensor(self.global_patch_position[b],dtype=torch.float32,device=self.device), batch_depth[b][ix].shape[-2],batch_depth[b][ix].shape[-1],self.args.input_vfov,self.args.input_hfov,position,heading_angle, far=self.args.deleted_frustum_distance)
                
                u = u % batch_depth[b][ix].shape[-1]
                v = v % batch_depth[b][ix].shape[-2]
                camera_depth = batch_depth[b][ix][v,u]

                frustum_mask = frustum_mask & (frustum_depth < camera_depth+0.1) # !!!!!!!!!!
                frustum_mask = frustum_mask.cpu().numpy()

                # Mark that all old information is removed
                self.global_patch_position[b][frustum_mask] = -10000. # inf
                self.global_patch_fts[b][frustum_mask] = 0.
                self.global_patch_directions[b][frustum_mask] = 0.
                self.global_patch_scales[b][frustum_mask] = 0.

                patch_id_list = np.arange(len(self.global_patch_fts[b]))[frustum_mask]
                
                for patch_id in patch_id_list.tolist():
                    if patch_id not in self.global_patch_to_instance_dict[b]:
                        continue
                    # Mark that the patch is removed
                    instance_id = self.global_patch_to_instance_dict[b].pop(patch_id)
                    filter_mask = self.global_instance_to_patch_dict[b][instance_id] != patch_id
                    self.global_instance_to_patch_dict[b][instance_id] = self.global_instance_to_patch_dict[b][instance_id][filter_mask]

                    if len(self.global_instance_to_patch_dict[b][instance_id]) == 0:
                        # Mark that the instance is removed
                        self.global_instance_to_patch_dict[b].pop(instance_id)

                        deleted_instance_position = self.global_instance_position[b][instance_id]
                        updated_zone_position = torch.cat([(deleted_instance_position[0:1]//zone_x_length)*zone_x_length+zone_x_length/2, (deleted_instance_position[1:2]//zone_y_length)*zone_y_length+zone_y_length/2, (deleted_instance_position[2:3]//zone_z_length)*zone_z_length+zone_z_length/2],dim=-1)
                        self.global_instance_position[b][instance_id] = -10000. # inf
                        self.global_instance_fts[b][instance_id] = 0.

                        zone_key = tuple(updated_zone_position.cpu().numpy().tolist())

                        if zone_key in self.global_zone_key_to_id[b]:
                            zone_id = self.global_zone_key_to_id[b][zone_key]
                            filter_mask = self.global_zone_to_instance_dict[b][zone_id] != instance_id
                            self.global_zone_to_instance_dict[b][zone_id] = self.global_zone_to_instance_dict[b][zone_id][filter_mask]

                            if len(self.global_zone_to_instance_dict[b][zone_id]) == 0:
                                # Mark that the zone is removed
                                self.global_zone_key_to_id[b].pop(zone_key)
                                self.global_zone_to_instance_dict[b].pop(zone_id)
                                self.global_zone_position[b][zone_id] = -10000. # inf
                                self.global_zone_fts[b][zone_id] = 0.
                        
            # Update the kd-tree
            self.instance_tree[b] = self.get_instance_tree(b)             



    @torch.no_grad()
    def get_patch_segm(self, batch_image, imgsz=(576,576), conf=0.4, iou=0.8):
        with torch.no_grad():
            batch_patch_segm = [b for b in range(len(batch_image))]
            for b in range(len(batch_image)): # use prange for parallel

                try:
                    segm_results = self.FastSAM(batch_image[b], device=self.device, retina_masks=True, imgsz=imgsz, conf=conf, iou=iou)
                    prompt_process = FastSAMPrompt(batch_image[b], segm_results, device=self.device)
                    # everything prompt
                    masks = prompt_process.everything_prompt()
                    patch_group = masks[0].clone()
                    for group_id in range(masks.shape[0]):
                        patch_group[masks[group_id]==1] = group_id
                    patch_group = torch.nn.functional.interpolate(patch_group.unsqueeze(0).unsqueeze(0),(self.args.input_height,self.args.input_width),mode='nearest').to(torch.int64).squeeze(0)
                    
                    patch_segm = patch_group.clone()
                    group_id = 0
                    for mask_id in torch.unique(patch_group).cpu().numpy().tolist():
                        patch_segm[patch_group==mask_id] = group_id
                        group_id += 1

                    batch_patch_segm[b] = patch_segm.unsqueeze(0).to(self.device)

                except:
                    print("FastSAM error, skip...")
                    batch_patch_segm[b] = torch.zeros((1, 1, self.args.input_height, self.args.input_width), dtype=torch.int64, device=self.device)

            batch_patch_segm = torch.cat(batch_patch_segm,0)
        
        return batch_patch_segm
    

    def assign_new_patch_ids(self, batch_id, num_patchs):
        new_patch_ids = []
        if len(self.global_patch_to_instance_dict[batch_id]) == 0:
            return np.array([i for i in range(num_patchs)])
        else:
            for i in range(len(self.global_patch_to_instance_dict[batch_id])+num_patchs):
                if len(new_patch_ids) == num_patchs:
                    break
                if i in self.global_patch_to_instance_dict[batch_id].keys():
                    continue
                else:
                    new_patch_ids.append(i)
        return np.array(new_patch_ids)
    

    def assign_new_instance_ids(self, batch_id, num_instances):
        new_instance_ids = []
        if len(self.global_instance_to_patch_dict[batch_id]) == 0:
            return np.array([i for i in range(num_instances)])
        else:
            for i in range(len(self.global_instance_to_patch_dict[batch_id])+num_instances):
                if len(new_instance_ids) == num_instances:
                    break
                if i in self.global_instance_to_patch_dict[batch_id].keys():
                    continue
                else:
                    new_instance_ids.append(i)
        return np.array(new_instance_ids)
    

    def assign_new_zone_ids(self, batch_id, num_zones):
        new_zone_ids = []
        if len(self.global_zone_to_instance_dict[batch_id]) == 0:
            return np.array([i for i in range(num_zones)])
        else:
            for i in range(len(self.global_zone_to_instance_dict[batch_id])+num_zones):
                if len(new_zone_ids) == num_zones:
                    break
                if i in self.global_zone_to_instance_dict[batch_id].keys():
                    continue
                else:
                    new_zone_ids.append(i)
        return np.array(new_zone_ids)


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
    

    def update_feature_fields(self, batch_depth, batch_grid_ft, batch_image, batch_position=None, batch_heading=None, batch_camera_intrinsic=None, batch_rot=None, batch_trans=None, depth_scale=1000.0, depth_trunc=1000.0, num_of_views=1):
        
        # Avoid the gradient loop and gpu out of memory
        if len(self.global_instance_fts[0]) > 0:
            self.global_instance_fts = [instance_fts.detach() for instance_fts in self.global_instance_fts]
            self.global_zone_fts = [zone_fts.detach() for zone_fts in self.global_zone_fts]

        batch_grid_ft = [grid_ft.astype(np.float16) for grid_ft in batch_grid_ft]
        if batch_camera_intrinsic != None: # Most 3D datasets
            self.sampled_rays = self.get_rays(batch_camera_intrinsic[0][0])
            batch_patch_segm = []
            for b in range(self.batch_size):
                batch_patch_segm.append(self.get_patch_segm(batch_image[b]))

        if batch_position != None: # Habitat simulator
            batch_image = batch_image.reshape((-1,batch_image.shape[-3],batch_image.shape[-2],batch_image.shape[-1]))
            batch_patch_segm = self.get_patch_segm(batch_image)
            batch_patch_segm = batch_patch_segm.view(self.batch_size, num_of_views, batch_patch_segm.shape[-2], batch_patch_segm.shape[-1])

        zone_x_length, zone_y_length, zone_z_length = self.args.zone_x_length, self.args.zone_y_length, self.args.zone_z_length


        for b in range(self.batch_size): # use prange for parallel

            if batch_camera_intrinsic != None: # Most 3D datasets
                thread_output = self.thread_pool([ [project_depth_to_3d, [ batch_depth[b][job_id], batch_camera_intrinsic[b][job_id],depth_scale,depth_trunc,self.args.input_height,self.args.input_width ], {} ] for job_id in range(len(batch_depth[b])) ]) # Parallel computing with multiple CPUs
                num_of_views = batch_depth[b].shape[0]

            if batch_position != None: # Habitat simulator
                position = batch_position[b].copy()
                position[0], position[1], position[2] = batch_position[b][0], - batch_position[b][2], batch_position[b][1] # Note to swap y,z axis, - y
                heading = batch_heading[b]
                depth = batch_depth[b]
                depth = depth.reshape((-1,self.args.input_height*self.args.input_width))
            

            for ix in range(num_of_views): # 12 views, rotation for panorama
                instance_fts = []
                instance_position = []
                proposal_num = min(len(self.global_instance_to_patch_dict[b]), self.args.num_proposal_instances)
                if batch_camera_intrinsic != None: # Most 3D datasets
                    # Get the patch information
                    points, points_mask = thread_output[ix]
                    points = points.astype(np.float32)
                    _, rel_direction, _ = self.sampled_rays
                    
                    patch_scale = points[:,-1] * math.fabs(math.tan(rel_direction[0][-1])) * 2. / self.args.input_width
                    R = batch_rot[b][ix]
                    T = batch_trans[b][ix]
                    points = (R @ points.T + T).T

                    patch_position = points.astype(np.float32)
                    patch_direction = self.get_heading_angle(points).astype(np.float32)
                    patch_scale = patch_scale.astype(np.float32)
                
                if batch_position != None: # Habitat simulator
                    # Get the patch information
                    rel_x, rel_y, rel_z, patch_direction, patch_scale = self.project_depth_to_3d_habitat(depth[ix:ix+1],ix*(-math.pi/6)+heading)  
                    patch_x = torch.tensor(rel_x + position[0],device=self.device).unsqueeze(-1)
                    patch_y = torch.tensor(rel_y + position[1],device=self.device).unsqueeze(-1)
                    patch_z = torch.tensor(rel_z + position[2],device=self.device).unsqueeze(-1)
                    patch_position = torch.cat([patch_x,patch_y,patch_z],dim=-1)[0].cpu().numpy()

                # Update the patch information
                if self.global_patch_position[b] == []:
                    self.global_patch_position[b] = patch_position
                    self.global_patch_scales[b] = patch_scale
                    self.global_patch_directions[b] = patch_direction
                else:
                    self.global_patch_position[b] = np.concatenate([self.global_patch_position[b], patch_position],0)
                    
                    self.global_patch_scales[b] = np.concatenate([self.global_patch_scales[b],patch_scale],0)
                    self.global_patch_directions[b] = np.concatenate([self.global_patch_directions[b],patch_direction],0)

                if self.global_patch_fts[b] == []:
                    self.global_patch_fts[b] = batch_grid_ft[b][ix]       
                else:
                    self.global_patch_fts[b] = np.concatenate((self.global_patch_fts[b],batch_grid_ft[b][ix]),axis=0)


                # Obtain the 2D instance information from the current observation
                patch_position = torch.tensor(patch_position,device=self.device, dtype=torch.float32)
                patch_direction = torch.tensor(patch_direction,device=self.device, dtype=torch.float32)
                patch_scale = torch.tensor(patch_scale,device=self.device, dtype=torch.float32)
                patch_segm = batch_patch_segm[b][ix].view(-1)
                patch_fts = torch.tensor(batch_grid_ft[b][ix],device=self.device, dtype=torch.float16)

                for segm_id in torch.unique(patch_segm).cpu().numpy().tolist():
                    
                    segm_patch_position = patch_position[patch_segm==segm_id]
                    cluster_center_position = segm_patch_position.mean(0)
                    patch_to_center_position = segm_patch_position - cluster_center_position
                    instance_position.append(cluster_center_position.unsqueeze(0))
                    
                    patch_to_center_distance = torch.sqrt(torch.square(segm_patch_position).sum(-1)).unsqueeze(-1)
                    patch_to_center_direction = patch_direction[patch_segm==segm_id].unsqueeze(-1)
                    patch_to_center_scale = patch_scale[patch_segm==segm_id].unsqueeze(-1)

                    patch_position_embedding = torch.cat([patch_to_center_position,patch_to_center_distance,torch.sin(patch_to_center_direction),torch.cos(patch_to_center_direction),patch_to_center_scale],dim=-1)
                    patch_embedding = patch_fts[patch_segm==segm_id] + self.patch_to_instance_position_embedding(patch_position_embedding)
                    patch_embedding = torch.cat([self.aggregate_patch_to_instance_embedding,patch_embedding],dim=0)
                    
                    instance_ft = self.aggregate_patch_to_instance_encoder(patch_embedding)[0:1]
                    
                    instance_fts.append(instance_ft)


                instance_fts = torch.cat(instance_fts,dim=0)
                instance_position = torch.cat(instance_position,dim=0)

                # Update the 3D instance information
                if self.instance_tree[b] != []:
                    instance_num = instance_fts.shape[0]
                    gt_dists, gt_inds = self.instance_tree[b].query(instance_position, nr_nns_searches=proposal_num)
                    if gt_dists.sum().cpu().numpy().item() > 1e6:
                        dist_mask = gt_dists.sum(0)
                        proposal_num = len(dist_mask[dist_mask<1e6])
                        gt_dists, gt_inds = self.instance_tree[b].query(instance_position,nr_nns_searches=proposal_num)

                    #gt_dists = torch.sqrt(gt_dists) #Note that the cupy_kdtree distances are squared
                    proposal_3d_instance_position = self.global_instance_position[b][gt_inds]
                    merged_instance_position = instance_position.unsqueeze(1).repeat(1,proposal_num,1) - proposal_3d_instance_position
                    proposal_3d_instance_fts = self.global_instance_fts[b][gt_inds]
                    merged_instance_fts = instance_fts.unsqueeze(1).repeat(1,proposal_num,1)
                    proposal_3d_instance_merge_input = torch.cat([proposal_3d_instance_fts, merged_instance_fts, merged_instance_position],dim=-1)
                    merge_logits = self.instance_merge_discriminator(proposal_3d_instance_merge_input)
                    merge_score = torch.softmax(merge_logits,dim=-1)

                    merge_target = torch.argmax(merge_score,dim=-1)

                    if torch.any(merge_target.sum(-1)==0): # Some new instances, no need to merge
                        new_instance_num = torch.zeros((instance_num,))
                        new_instance_num[merge_target.sum(-1)==0] = 1
                        new_instance_num = int(new_instance_num.sum().cpu().numpy().item())
                        assigned_instance_ids = self.assign_new_instance_ids(b, new_instance_num)

                    assigned_patch_ids = self.assign_new_patch_ids(b, patch_fts.shape[-2])

                    new_instance_idex = 0
                    for segm_id in range(instance_num):
                        if torch.all(merge_target[segm_id]==0): # No merged instance, get a new instance
                            patch_ids_belong_to_instance = assigned_patch_ids[patch_segm.cpu().numpy()==segm_id]

                            instance_id = assigned_instance_ids[new_instance_idex].item()
                            new_instance_idex += 1

                            self.global_instance_to_patch_dict[b][instance_id] = patch_ids_belong_to_instance
                            for patch_id in patch_ids_belong_to_instance.tolist():
                                self.global_patch_to_instance_dict[b][patch_id] = instance_id

                            if instance_id < len(self.global_instance_position[b]):
                                self.global_instance_position[b][instance_id] = instance_position[segm_id]
                                self.global_instance_fts[b][instance_id] = instance_fts[segm_id]
                            else:
                                self.global_instance_position[b] = torch.cat([self.global_instance_position[b],instance_position[segm_id:segm_id+1]],dim=0)
                                self.global_instance_fts[b] = torch.cat([self.global_instance_fts[b],instance_fts[segm_id:segm_id+1]],dim=0)


                        else: # Merge instance
                            for proposal_3d_instance_id in range(proposal_num):
                                if merge_target[segm_id, proposal_3d_instance_id].cpu().numpy().item() != 0: # Only merge into the nearest 3d instance
                                    instance_id = gt_inds[segm_id,proposal_3d_instance_id].cpu().numpy().item()

                                    patch_ids_belong_to_instance = assigned_patch_ids[patch_segm.cpu().numpy()==segm_id]
                                                                          
                                    self.global_instance_to_patch_dict[b][instance_id] = np.concatenate([self.global_instance_to_patch_dict[b][instance_id],patch_ids_belong_to_instance ],axis=0) # Merge all patchs into this instance, update the instance dict
                                    for patch_id in patch_ids_belong_to_instance.tolist():
                                        self.global_patch_to_instance_dict[b][patch_id] = instance_id
                                    
                                    patch_position_set = torch.tensor(self.global_patch_position[b][self.global_instance_to_patch_dict[b][instance_id]],device=self.device)
                                    self.global_instance_position[b][instance_id] = patch_position_set.mean(0) # Update merged instance position

                                    patch_fts_set = torch.tensor(self.global_patch_fts[b][self.global_instance_to_patch_dict[b][instance_id]],device=self.device)

                                    patch_to_center_position = patch_position_set - self.global_instance_position[b][instance_id]
                                    patch_to_center_distance = torch.sqrt(torch.square(patch_position_set).sum(-1)).unsqueeze(-1)

                                    patch_to_center_direction = torch.tensor( self.global_patch_directions[b][self.global_instance_to_patch_dict[b][instance_id]], device=self.device).unsqueeze(-1)
                                    patch_to_center_scale = torch.tensor( self.global_patch_scales[b][self.global_instance_to_patch_dict[b][instance_id]], device=self.device).unsqueeze(-1)

                                    patch_position_embedding = torch.cat([patch_to_center_position,patch_to_center_distance,torch.sin(patch_to_center_direction),torch.cos(patch_to_center_direction),patch_to_center_scale],dim=-1)
                                    patch_fts_set = patch_fts_set + self.patch_to_instance_position_embedding(patch_position_embedding)
                                    
                                    patch_fts_set = torch.cat([self.aggregate_patch_to_instance_embedding,patch_fts_set],dim=0)
                                    
                                    total_memory = torch.cuda.get_device_properties(self.device).total_memory / (1024 ** 3)
                                    used_memory = torch.cuda.memory_allocated(self.device) / (1024 ** 3)
                                    free_memory = total_memory - used_memory

                                    if free_memory > 10: # Check the GPU Memory, it's very important to reduce the GPU memory
                                        new_instance_fts = self.aggregate_patch_to_instance_encoder(patch_fts_set)[0:1]
                                    else:
                                        with torch.no_grad():
                                            new_instance_fts = self.aggregate_patch_to_instance_encoder(patch_fts_set)[0:1]

                                    self.global_instance_fts[b][instance_id] = new_instance_fts # Update merged instance feature


                                    break # Only merge into the nearest 3d instance

                    # Update the zone information
                    global_zone_position = torch.cat([(self.global_instance_position[b][:,0:1]//zone_x_length)*zone_x_length+zone_x_length/2, (self.global_instance_position[b][:,1:2]//zone_y_length)*zone_y_length+zone_y_length/2, (self.global_instance_position[b][:,2:3]//zone_z_length)*zone_z_length+zone_z_length/2],dim=-1)
                    updated_zone_position = torch.cat([(instance_position[:,0:1]//zone_x_length)*zone_x_length+zone_x_length/2, (instance_position[:,1:2]//zone_y_length)*zone_y_length+zone_y_length/2, (instance_position[:,2:3]//zone_z_length)*zone_z_length+zone_z_length/2],dim=-1)
                    assigned_zone_list, instance_counts = torch.unique(updated_zone_position,return_counts=True,dim=0)
                    assigned_zone_num = len(assigned_zone_list)
                    assigned_zone_ids = self.assign_new_zone_ids(b,assigned_zone_num)
                    updated_zone_position = updated_zone_position.cpu().numpy()


                    assigned_zone_index = 0
                    for i in range(assigned_zone_num):
                        assigned_zone_key = assigned_zone_list[i].cpu().numpy().tolist()
                        assigned_zone_key = tuple(assigned_zone_key)

                        if assigned_zone_key not in self.global_zone_key_to_id[b]: # Add a new zone
                            assigned_instance_masks = (global_zone_position[:,0]==assigned_zone_key[0]) & (global_zone_position[:,1]==assigned_zone_key[1]) & (global_zone_position[:,2]==assigned_zone_key[2])
                            zone_id = assigned_zone_ids[assigned_zone_index].item()
                            self.global_zone_key_to_id[b][assigned_zone_key] = zone_id
                            self.global_zone_to_instance_dict[b][zone_id] = np.arange(len(assigned_instance_masks))[assigned_instance_masks.cpu().numpy()]
                            assigned_zone_index += 1 # New zone

                            instance_position_set = self.global_instance_position[b][assigned_instance_masks]
                            self.global_zone_position[b] = torch.cat([self.global_zone_position[b],instance_position_set.mean(0,keepdim=True)],dim=0) # Add the zone position
                            
                            instance_fts_set = self.global_instance_fts[b][assigned_instance_masks]

                            instance_to_center_position = instance_position_set - instance_position_set.mean(0,keepdim=True)
                            
                            instance_to_center_distance = torch.sqrt(torch.square(instance_position_set).sum(-1)).unsqueeze(-1)

                            instance_position_embedding = torch.cat([instance_to_center_position,instance_to_center_distance],dim=-1)
                            
                            instance_fts_set = instance_fts_set + self.instance_to_zone_position_embedding(instance_position_embedding)
                            
                            instance_fts_set = torch.cat([self.aggregate_instance_to_zone_embedding,instance_fts_set],dim=0)
                            new_zone_fts = self.aggregate_instance_to_zone_encoder(instance_fts_set)

                            self.global_zone_fts[b] = torch.cat([self.global_zone_fts[b],new_zone_fts[0:1]],dim=0) # Add the zone feature
                            
                            

                        else: # Update the old zone
                            zone_id = self.global_zone_key_to_id[b][assigned_zone_key]
                            assigned_instance_masks = (global_zone_position[:,0]==assigned_zone_key[0]) & (global_zone_position[:,1]==assigned_zone_key[1]) & (global_zone_position[:,2]==assigned_zone_key[2])
                            self.global_zone_to_instance_dict[b][zone_id] = np.arange(len(assigned_instance_masks))[assigned_instance_masks.cpu().numpy()]

                            instance_position_set = global_zone_position[assigned_instance_masks]

                            self.global_zone_position[b][zone_id] = instance_position_set.mean(0) # Update the zone position
                            
                            instance_fts_set = self.global_instance_fts[b][assigned_instance_masks]

                            
                            instance_to_center_position = instance_position_set - self.global_zone_position[b][zone_id]
                            instance_to_center_distance = torch.sqrt(torch.square(instance_position_set).sum(-1)).unsqueeze(-1)

                            instance_position_embedding = torch.cat([instance_to_center_position,instance_to_center_distance],dim=-1)
                            
                            instance_fts_set = instance_fts_set + self.instance_to_zone_position_embedding(instance_position_embedding)
                            
                            instance_fts_set = torch.cat([self.aggregate_instance_to_zone_embedding,instance_fts_set],dim=0)
                            new_zone_fts = self.aggregate_instance_to_zone_encoder(instance_fts_set)

                            self.global_zone_fts[b][zone_id] = new_zone_fts[0:1] # Update the zone feature

                    
                else:
                    # Initialize the 3d instances information
                    self.global_instance_position[b].append(instance_position)
                    self.global_instance_position[b] = torch.cat(self.global_instance_position[b],dim=0)
                    self.global_instance_fts[b].append(instance_fts)
                    self.global_instance_fts[b] = torch.cat(self.global_instance_fts[b],dim=0)

                    assigned_instance_ids = self.assign_new_instance_ids(b, instance_fts.shape[0])
                    assigned_patch_ids = self.assign_new_patch_ids(b, patch_fts.shape[-2])
                    
                    for segm_id in torch.unique(patch_segm).cpu().numpy().tolist():
                        patch_ids_belong_to_instance = assigned_patch_ids[patch_segm.cpu().numpy()==segm_id]
                        instance_id = assigned_instance_ids[segm_id].item()
                        self.global_instance_to_patch_dict[b][instance_id] = patch_ids_belong_to_instance
                        for patch_id in patch_ids_belong_to_instance.tolist():
                            self.global_patch_to_instance_dict[b][patch_id] = instance_id


                    # Initialize the zone information
                    updated_zone_position = torch.cat([(instance_position[:,0:1]//zone_x_length)*zone_x_length+zone_x_length/2, (instance_position[:,1:2]//zone_y_length)*zone_y_length+zone_y_length/2, (instance_position[:,2:3]//zone_z_length)*zone_z_length+zone_z_length/2],dim=-1)
                    assigned_zone_list, instance_counts = torch.unique(updated_zone_position,return_counts=True,dim=0)
                    assigned_zone_num = len(assigned_zone_list)
                    assigned_zone_ids = self.assign_new_zone_ids(b,assigned_zone_num)
                    updated_zone_position = updated_zone_position.cpu().numpy()


                    for assigned_zone_index in range(assigned_zone_num):
                        assigned_zone_key = assigned_zone_list[assigned_zone_index].cpu().numpy().tolist()
                        assigned_zone_key = tuple(assigned_zone_key)
                        assigned_instance_masks = (updated_zone_position[:,0]==assigned_zone_key[0]) & (updated_zone_position[:,1]==assigned_zone_key[1]) & (updated_zone_position[:,2]==assigned_zone_key[2])
                        zone_id = assigned_zone_ids[assigned_zone_index].item()
                        self.global_zone_key_to_id[b][assigned_zone_key] = zone_id
                        self.global_zone_to_instance_dict[b][zone_id] = np.arange(len(assigned_instance_masks))[assigned_instance_masks]

                        instance_position_set = instance_position[assigned_instance_masks]
                        self.global_zone_position[b].append(instance_position_set.mean(0,keepdim=True))
                        
                        instance_fts_set = instance_fts[assigned_instance_masks]

                        instance_to_center_position = instance_position_set - self.global_zone_position[b][-1]
                        instance_to_center_distance = torch.sqrt(torch.square(instance_position_set).sum(-1)).unsqueeze(-1)

                        instance_position_embedding = torch.cat([instance_to_center_position,instance_to_center_distance],dim=-1)
                        
                        instance_fts_set = instance_fts_set + self.instance_to_zone_position_embedding(instance_position_embedding)
                        
                        instance_fts_set = torch.cat([self.aggregate_instance_to_zone_embedding,instance_fts_set],dim=0)
                        new_zone_fts = self.aggregate_instance_to_zone_encoder(instance_fts_set)
                        
                        self.global_zone_fts[b].append(new_zone_fts[0:1]) # Add the zone feature


                    self.global_zone_position[b] = torch.cat(self.global_zone_position[b],dim=0)
                    self.global_zone_fts[b] = torch.cat(self.global_zone_fts[b],dim=0)

                # Update the kd-tree
                self.instance_tree[b] = self.get_instance_tree(b)


    def get_environment_features(self, agent_position, agent_heading_angle, instance_distance=5., zone_distance=100.):
        batch_instance_fts = []
        batch_instance_relative_position = []
        batch_zone_fts = []
        batch_zone_relative_position = []
        for b in range(self.batch_size):
            # Get the instance features
            instance_ids = torch.tensor(list(self.global_instance_to_patch_dict[b].keys()),device=self.device)
            instance_fts = self.global_instance_fts[b][instance_ids]
            instance_position = self.global_instance_position[b][instance_ids]

            camera_position = agent_position[b].copy()
            camera_position[0], camera_position[1], camera_position[2] = agent_position[b][0], - agent_position[b][2], agent_position[b][1] # Note to swap y,z axis, - y
            heading_angle = - agent_heading_angle[b]

            points_x, points_y, points_z = instance_position[:,0:1] - camera_position[0], instance_position[:,1:2] - camera_position[1], instance_position[:,2:3] - camera_position[2]

            rel_x = points_x * math.cos(heading_angle) - points_y * math.sin(heading_angle)
            rel_y = points_x * math.sin(heading_angle) + points_y * math.cos(heading_angle)
            rel_z = points_z
            instance_relative_position = torch.cat([rel_x,rel_y,rel_z],dim=-1)
            filt_mask = torch.sqrt(torch.square(instance_relative_position).sum(-1)) <= instance_distance
            batch_instance_relative_position.append(instance_relative_position[filt_mask])
            batch_instance_fts.append(instance_fts[filt_mask])

            # Get the zone features
            zone_ids = torch.tensor(list(self.global_zone_to_instance_dict[b].keys()),device=self.device)
            zone_fts = self.global_zone_fts[b][zone_ids]
            zone_position = self.global_zone_position[b][zone_ids]
            
            points_x, points_y, points_z = zone_position[:,0:1] - camera_position[0], zone_position[:,1:2] - camera_position[1], zone_position[:,2:3] - camera_position[2]
            rel_x = points_x * math.cos(heading_angle) - points_y * math.sin(heading_angle)
            rel_y = points_x * math.sin(heading_angle) + points_y * math.cos(heading_angle)
            rel_z = points_z
            zone_relative_position = torch.cat([rel_x,rel_y,rel_z],dim=-1)
            filt_mask = torch.sqrt(torch.square(zone_relative_position).sum(-1)) <= zone_distance
            batch_zone_relative_position.append(zone_relative_position[filt_mask])
            batch_zone_fts.append(zone_fts[filt_mask])

        return {
                "batch_instance_fts":batch_instance_fts,
                "batch_instance_relative_position":batch_instance_relative_position,
                "batch_zone_fts":batch_zone_fts,
                "batch_zone_relative_position":batch_zone_relative_position
                }