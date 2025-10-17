import torch
#torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from torch_kdtree import build_kd_tree
import tinycudann as tcnn
import open3d as o3d
from joblib import Parallel
from einops import einsum
try:
    from apex.normalization.fused_layer_norm import FusedLayerNorm as BertLayerNorm
except (ImportError, AttributeError) as e:
    # logger.info("Better speed can be achieved with apex installed from https://www.github.com/nvidia/apex .")
    BertLayerNorm = torch.nn.LayerNorm

from src_3dff.models.fastsam import FastSAM, FastSAMPrompt
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

    parser.add_argument("--num_proposal_instances", type=int, default=4)


    # Feature Fields
    parser.add_argument("--near", type=float, default=0., 
                        help='near distance')
    parser.add_argument("--far", type=float, default=10., 
                        help='far distance')
    
    # Novel view settings
    parser.add_argument("--view_hfov", type=float, default=90., 
                        help='hfov angle of novel view')
    parser.add_argument("--view_vfov", type=float, default=90., 
                    help='vfov angle of novel view')
    parser.add_argument("--view_height", type=int, default=12, 
                        help='height of the novel view')
    parser.add_argument("--view_width", type=int, default=12, 
                        help='width of the novel view')

    parser.add_argument("--feature_fields_search_radius", type=float, default=1., 
                        help='search radius for near features')
    parser.add_argument("--feature_fields_search_num", type=int, default=4, 
                        help='The number of searched near features')
    parser.add_argument("--mlp_net_layers", type=int, default=4, 
                        help='layers in mlp network')
    parser.add_argument("--mlp_net_width", type=int, default=768, 
                        help='channels per layer in mlp network')
    parser.add_argument("--N_samples", type=int, default=501, 
                        help='number of coarse samples per ray')
    parser.add_argument("--N_importance", type=int, default=8,
                        help='number of fine samples per ray')

    return parser



def project_depth_to_3d(depth, intrinsic, depth_scale, depth_trunc, input_height, input_width): # Don't define in the Feature_Fields Class, to avoid the memory copy of entire Feature_Fields Class at each thread when parallel computing
    depth[depth==0] = depth.max() # filter out the noise
    o3d_depth = o3d.geometry.Image(depth.numpy().astype(np.uint16))
    pcd = o3d.geometry.PointCloud.create_from_depth_image(o3d_depth, o3d.camera.PinholeCameraIntrinsic(depth.shape[1],depth.shape[0],intrinsic[0][0],intrinsic[1][1],intrinsic[0][2],intrinsic[1][2]), depth_scale=depth_scale, depth_trunc=depth_trunc)
    points = np.asarray(pcd.points)
    try:
        points = torch.tensor(points).view(depth.shape[0],depth.shape[1],3).permute(2,0,1).unsqueeze(0) # num_of_samples x channels x height x width
        points = F.interpolate(points, size=(input_height,input_width), scale_factor=None, mode='nearest').squeeze(0).permute(1,2,0).view(-1,3).numpy()
    except:
        points = np.zeros((input_height*input_width,3))

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

        self.global_gt_instance_ids = [[] for i in range(self.batch_size)]

        self.instance_tree = [[] for i in range(self.batch_size)]
        self.patch_tree = [[] for i in range(self.batch_size)]
        

        # For 3d patch rendering via 3d feature fields

        self.nerf_encoder = tcnn.Network(
            n_input_dims=args.mlp_net_width,
            n_output_dims=args.mlp_net_width+1,
            network_config={
                "otype": "CutlassMLP",
                "activation": "LeakyReLU",
                "output_activation": "LeakyReLU",
                "n_neurons": args.mlp_net_width,
                "n_hidden_layers": args.mlp_net_layers//2,
            },
        )

        self.nerf_decoder = tcnn.Network(
            n_input_dims=args.mlp_net_width,
            n_output_dims=args.mlp_net_width,
            network_config={
                "otype": "CutlassMLP",
                "activation": "LeakyReLU",
                "output_activation": "None",
                "n_neurons": args.mlp_net_width,
                "n_hidden_layers": args.mlp_net_layers - args.mlp_net_layers//2,
            },
        )

        width = args.mlp_net_width

        self.patch_to_nerf_position_embedding = nn.Sequential(
            nn.Linear(6, width),
            BertLayerNorm(width, eps=1e-12)
        )
        self.aggregate_patch_to_nerf_encoder = nn.Sequential(
            nn.Linear(args.mlp_net_width*args.feature_fields_search_num, width),
            BertLayerNorm(width, eps=1e-12)
        )

        self.initialize_parameters()
        self.FastSAM = FastSAM('FastSAM.pt')

    def initialize_parameters(self):
        std = self.args.fts_dim ** -0.5
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.normal_(param, std=std)

    def reset(self, batch_size=1, mode='habitat', batch_gt_pcd_xyz=None, batch_gt_pcd_label=None):
        self.batch_size = batch_size
        self.mode = mode
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
        self.global_gt_instance_ids = [[] for i in range(self.batch_size)]

        self.patch_tree = [[] for i in range(self.batch_size)]
        self.instance_tree = [[] for i in range(self.batch_size)]
        

        if batch_gt_pcd_xyz is not None: # Load the point cloud and instance label of 3d scenes
            self.gt_pcd_xyz = []
            self.gt_pcd_tree = []
            self.gt_pcd_label = []
            for i in range(self.batch_size):
                if batch_gt_pcd_xyz[i] is not None:
                    gt_pcd_xyz = batch_gt_pcd_xyz[i].clone()

                    if mode == "habitat":
                        gt_pcd_xyz[:,0], gt_pcd_xyz[:,1], gt_pcd_xyz[:,2] = batch_gt_pcd_xyz[i][:,0], batch_gt_pcd_xyz[i][:,1], batch_gt_pcd_xyz[i][:,2]-1.25 # Agent's height is 1.25 meters
                    else:
                        gt_pcd_xyz[:,0], gt_pcd_xyz[:,1], gt_pcd_xyz[:,2] = batch_gt_pcd_xyz[i][:,0], batch_gt_pcd_xyz[i][:,1], batch_gt_pcd_xyz[i][:,2]


                    self.gt_pcd_xyz.append(gt_pcd_xyz) 
                    self.gt_pcd_tree.append(build_kd_tree(gt_pcd_xyz.to(torch.float32).to(self.device)))
                    self.gt_pcd_label.append(batch_gt_pcd_label[i].to(torch.int64).to(self.device)) 
                else:
                    self.gt_pcd_xyz.append(None)
                    self.gt_pcd_tree.append(None)
                    self.gt_pcd_label.append(None)

            del batch_gt_pcd_xyz, batch_gt_pcd_label

        else:
            self.gt_pcd_xyz = None
            self.gt_pcd_tree = None
            self.gt_pcd_label = None


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
        self.global_gt_instance_ids.pop(index)

        self.instance_tree.pop(index)
        self.patch_tree.pop(index)

        if self.gt_pcd_xyz is not None:
            self.gt_pcd_xyz.pop(index)
            self.gt_pcd_tree.pop(index)
            self.gt_pcd_label.pop(index)


    def initialize_camera_setting(self, hfov, vfov):
        self.args.input_hfov = hfov
        self.args.input_vfov = vfov


    def delete_feature_fields(self): # Free the memory
        del self.global_patch_fts, self.global_patch_position, self.global_patch_scales, self.global_patch_directions, \
            self.global_patch_to_instance_dict, self.global_instance_fts, self.global_instance_position, self.global_instance_to_patch_dict, \
            self.global_zone_fts, self.global_zone_position, self.global_zone_key_to_id, self.global_zone_to_instance_dict, self.global_gt_instance_ids, \
            self.instance_tree, self.patch_tree
        
        if self.gt_pcd_xyz is not None:
            del self.gt_pcd_xyz, self.gt_pcd_tree, self.gt_pcd_label


    def get_patch_tree(self, batch_id):
        if len(self.global_patch_position[batch_id]) == 0:
            return []
        patch_tree = build_kd_tree(self.global_patch_position[batch_id])
        return patch_tree


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


    def get_rays(self,camera_intrinsic):
        N_spacing = (self.args.far - self.args.near) / self.args.N_samples
        sampled_points = o3d.geometry.PointCloud()
        for N_index in range(self.args.N_samples):
            N_distance = self.args.near + N_spacing * (N_index+1)
            N_depth = np.full((self.args.view_height,self.args.view_width),N_distance,dtype=np.float32)
            N_depth = o3d.geometry.Image(N_depth)
            N_points = o3d.geometry.PointCloud.create_from_depth_image(N_depth, o3d.camera.PinholeCameraIntrinsic(self.args.view_width,self.args.view_height,camera_intrinsic[0][0],camera_intrinsic[1][1],self.args.view_width/2,self.args.view_height/2), depth_scale=1., depth_trunc=1000.)
            sampled_points += N_points
        sampled_points = np.asarray(sampled_points.points)

        rel_position = sampled_points.reshape((self.args.N_samples, self.args.view_height*self.args.view_width,3)).transpose((1,0,2))

        rel_direction = - np.arctan(rel_position[...,-1:,0]/rel_position[...,-1:,2])
        rel_dist = rel_position[...,2]
        return (rel_position, rel_direction, rel_dist)


    def get_rays_habitat(self):
        H = self.args.view_height
        W = self.args.view_width
        rel_y = np.expand_dims(np.linspace(self.args.near, self.args.far, self.args.N_samples),axis=0).repeat(H*W,axis=0)    
        hfov_angle = np.deg2rad(self.args.view_hfov)
        vfov_angle = np.deg2rad(self.args.view_vfov)
        half_H = H//2
        half_W = W//2
        tan_xy = np.array(([[i/half_W+1/W] for i in range(-half_W,half_W)])*H,np.float32) * math.tan(hfov_angle/2.)
        rel_direction = - np.arctan(tan_xy)
        rel_x = rel_y * tan_xy
        rel_z = rel_y * (np.array([[i/half_H-1/H for i in range(half_H,-half_H,-1)]]*W,np.float32).T.reshape((-1,1)) * math.tan(vfov_angle/2.))
        rel_position = (rel_x,rel_y,rel_z)
        rel_dist = rel_y
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



    def raw2feature(self, sample_feature, sample_density, rel_dist, topk_inds):
        """Transforms model's predictions to semantically meaningful values.
        Args:
            sample_feature: [num_rays, num_important_samples along ray, dimension of feature]. Prediction from model.
            sample_density: [num_rays, num_important_samples along ray]. Prediction from model.
            rel_dist: [num_rays, num_samples along ray]. Integration time.
            topk_inds: [num_rays, num_important_samples along ray]. Important sample_id along ray
        Returns:
            feature_map: [num_rays, 768]. Estimated semantic feature of a ray.
            depth_map: [num_rays]. Estimated distance to camera.
        """
        sample_density = F.softplus(sample_density) # Make sure sample_density > 0.

        raw2alpha = lambda raw, dists, act_fn=F.relu: 1.-torch.exp(-act_fn(raw)*dists)
        dists = torch.abs(rel_dist[...,1:] - rel_dist[...,:-1])
        dists = torch.cat([dists, torch.Tensor([1e10]).expand(dists[...,:1].shape).to(dists.device)], -1)  # [N_rays, N_samples]
        density = torch.zeros(rel_dist.shape,dtype=sample_density.dtype,device=sample_density.device)     
        density = torch.scatter(density,1,topk_inds,sample_density)

        alpha = raw2alpha(density, dists)  # [N_rays, N_samples]
        
        weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)).to(dists.device), 1.-alpha + 1e-10], -1), -1)[:, :-1]
        sample_weights = torch.gather(weights,1,topk_inds)
        feature_map = torch.sum(sample_weights[...,None] * sample_feature, -2)  # [N_rays, 768]
        feature_map = feature_map / torch.max(torch.linalg.norm(feature_map, dim=-1, keepdim=True),torch.tensor(1e-7,dtype=feature_map.dtype,device=feature_map.device))

        depth_map = torch.sum(weights * rel_dist, -1) / torch.max(torch.sum(weights, -1),torch.tensor(1e-7,dtype=weights.dtype,device=weights.device))

        return feature_map, depth_map


    def patch_to_nerf_encode(self, sample_ft_neighbor_embedding, sample_ft_neighbor_xyzds):

        sample_ft_neighbor_embedding = sample_ft_neighbor_embedding.view(-1,self.args.mlp_net_width*self.args.feature_fields_search_num).to(torch.float16)

        sample_ft_neighbor_xyzds = self.patch_to_nerf_position_embedding(sample_ft_neighbor_xyzds).view(-1,self.args.mlp_net_width*self.args.feature_fields_search_num).to(torch.float16)

        sample_input = self.aggregate_patch_to_nerf_encoder(sample_ft_neighbor_embedding+sample_ft_neighbor_xyzds)
        encoded_input = self.nerf_encoder(sample_input)
        encoded_input, density = encoded_input[::,:-1], encoded_input[::,-1]

        encoded_input = encoded_input + sample_input # Residual
        outputs = self.nerf_decoder(encoded_input).view(-1,self.args.N_importance, self.args.mlp_net_width)
        density = density.view(-1,self.args.N_importance)

        return outputs.to(torch.float16), density.to(torch.float16)


    def render_view_3d_patch(self, batch_position=None, batch_heading=None, batch_camera_intrinsic=None, batch_rot=None, batch_trans=None, visualization=False):

        batch_rendered_patch_fts = []
        batch_rendered_patch_positions = []
        batch_gt_label = []
        for batch_id in range(self.batch_size):
            if self.mode == 'habitat':
                position = batch_position[batch_id].copy()
                position[0], position[1], position[2] = batch_position[batch_id][0], - batch_position[batch_id][2], batch_position[batch_id][1] # Note to swap y,z axis, and -y
                camera_direction = batch_heading[batch_id]
            else:
                R = batch_rot[batch_id]
                T = batch_trans[batch_id]
                points = np.array([[0.,0.,0.]])
                points = (R @ points.T + T).T
                position = points[0] # Get position of camera

                points = np.array([[0.,0.,1.]])
                points = (R @ points.T + T).T
                
                camera_direction = torch.tensor(self.get_heading_angle(points)[0], dtype=torch.float32,device=self.device) # Get direction of camera

            camera_x, camera_y, camera_z = position[0], position[1], position[2]
            scene_fts, patch_directions, patch_scales, patch_positions, patch_tree = self.global_patch_fts[batch_id], self.global_patch_directions[batch_id], self.global_patch_scales[batch_id], self.global_patch_position[batch_id], self.patch_tree[batch_id]

            patch_directions = torch.tensor(patch_directions, dtype=torch.float32,device=self.device) - camera_direction

            patch_scales = torch.tensor(patch_scales, dtype=torch.float32, device=self.device).unsqueeze(-1)

            patch_positions = patch_positions.to(self.device)
            if self.mode == 'habitat':
                rel_position, rel_direction, rel_dist = self.sampled_rays
                rel_x, rel_y, rel_z = rel_position
                ray_x = rel_x * math.cos(camera_direction) - rel_y * math.sin(camera_direction) + camera_x
                ray_y = rel_x * math.sin(camera_direction) + rel_y * math.cos(camera_direction) + camera_y

                ray_z = rel_z + camera_z
                ray_xyz = torch.tensor(np.concatenate((np.expand_dims(ray_x,-1),np.expand_dims(ray_y,-1),np.expand_dims(ray_z,-1)),axis=-1),dtype=torch.float32, device=self.device)
            else:
                rel_position, rel_direction, rel_dist = self.sampled_rays
                ray_xyz = (R @ rel_position.reshape((-1,3)).T + T).T

                ray_xyz = torch.tensor(ray_xyz,dtype=torch.float32, device=self.device).view(self.args.view_height*self.args.view_width,self.args.N_samples,3)

            with torch.no_grad():
                shaped_ray_xyz = ray_xyz.view(-1,3)
                searched_ray_k_neighbor_dists, searched_ray_k_neighbor_inds = patch_tree.query(shaped_ray_xyz, nr_nns_searches=self.args.feature_fields_search_num)

            searched_ray_k_neighbor_dists = torch.sqrt(searched_ray_k_neighbor_dists) #Note that the cupy_kdtree distances are squared
            searched_ray_k_neighbor_inds[searched_ray_k_neighbor_dists >= self.args.feature_fields_search_radius] = -1
            searched_ray_k_neighbor_dists[searched_ray_k_neighbor_dists >= self.args.feature_fields_search_radius] = self.args.feature_fields_search_radius


            searched_ray_k_neighbor_inds = searched_ray_k_neighbor_inds.view(ray_xyz.shape[0],ray_xyz.shape[1],self.args.feature_fields_search_num)
            searched_ray_k_neighbor_dists = searched_ray_k_neighbor_dists.view(ray_xyz.shape[0],ray_xyz.shape[1],self.args.feature_fields_search_num)

            tmp_distance = searched_ray_k_neighbor_dists.sum(-1)
            tmp_density = (1/tmp_distance)                
            topk_inds = torch.topk(tmp_density, k=self.args.N_importance, dim=-1, largest=True)[1]
            # topk_inds = torch.sort(topk_inds, dim=-1)[0] # Search important sampled points

            sample_ray_xyz = torch.gather(ray_xyz,1,topk_inds.unsqueeze(-1).repeat(1,1,3))
            batch_rendered_patch_positions.append(sample_ray_xyz[:,0].unsqueeze(0)) # !!!!!!!!!!!!!!
            sample_ray_direction = rel_direction[::,-1]

            if self.gt_pcd_tree is not None and self.gt_pcd_tree[batch_id] is not None:
                gt_ray_xyz = torch.gather(ray_xyz,1,topk_inds.unsqueeze(-1).repeat(1,1,3))[...,0,:]
                with torch.no_grad():
                    gt_dists, gt_inds = self.gt_pcd_tree[batch_id].query(gt_ray_xyz,nr_nns_searches=1)
                gt_dists = torch.sqrt(gt_dists) # Note that the cupy_kdtree distances are squared
                gt_label = self.gt_pcd_label[batch_id][gt_inds]
                gt_label[gt_dists >= self.args.feature_fields_search_radius] = -100
                gt_label[searched_ray_k_neighbor_inds.sum(-1).sum(-1) == -self.args.N_samples*self.args.feature_fields_search_num] = -100
                batch_gt_label.append(gt_label.view(self.args.view_height, self.args.view_width))

            elif self.gt_pcd_tree is not None and self.gt_pcd_tree[batch_id] == None:
                batch_gt_label.append(None)

            if visualization: # Visualize the feature points of feature fields and sampled points of rays, set `True` for debug
                rays_pcd=o3d.geometry.PointCloud()
                rays_pcd.points = o3d.utility.Vector3dVector(ray_xyz.view(-1,3).cpu().numpy()) # ray_xyz, sample_ray_xyz
                feature_pcd=o3d.geometry.PointCloud()
                feature_pcd.points = o3d.utility.Vector3dVector(self.global_patch_position[batch_id][torch.abs(self.global_patch_position[batch_id]).sum(-1)<1000].cpu().numpy())
                #gt_pcd=o3d.geometry.PointCloud()
                #gt_pcd.points = o3d.utility.Vector3dVector(self.gt_pcd_xyz[batch_id])
                #feature_pcd += gt_pcd
                feature_pcd += rays_pcd
                o3d.visualization.draw_geometries([feature_pcd])

            with torch.no_grad():
                sample_feature_k_neighbor_dists, sample_feature_k_neighbor_inds = patch_tree.query(sample_ray_xyz.view(-1,3), nr_nns_searches=self.args.feature_fields_search_num)

            sample_feature_k_neighbor_dists = torch.sqrt(sample_feature_k_neighbor_dists) #Note that the cupy_kdtree distances are squared
            sample_feature_k_neighbor_inds[sample_feature_k_neighbor_dists >= self.args.feature_fields_search_radius] = -1
            sample_feature_k_neighbor_inds = sample_feature_k_neighbor_inds.view(sample_ray_xyz.shape[0],sample_ray_xyz.shape[1],self.args.feature_fields_search_num)


            sample_ft_neighbor_xyzds = torch.zeros((sample_ray_xyz.shape[0],sample_ray_xyz.shape[1],self.args.feature_fields_search_num,6),dtype=torch.float32, device=self.device)
            idx = sample_feature_k_neighbor_inds 
            sample_ft_neighbor_xyzds[...,:3] = patch_positions[idx] - sample_ray_xyz.unsqueeze(-2)
            # Get the relative positions of features to the sampled point

            sample_ft_neighbor_x = sample_ft_neighbor_xyzds[...,0]
            sample_ft_neighbor_y = sample_ft_neighbor_xyzds[...,1]
            sample_ft_neighbor_xyzds[...,0] = sample_ft_neighbor_x * math.cos(-camera_direction) - sample_ft_neighbor_y * math.sin(-camera_direction)
            sample_ft_neighbor_xyzds[...,1] = sample_ft_neighbor_x * math.sin(-camera_direction) + sample_ft_neighbor_y * math.cos(-camera_direction)


            sample_ft_neighbor_xyzds[...,:3][idx==-1] = self.args.far

            sample_ray_direction = torch.tensor(sample_ray_direction,device=sample_ft_neighbor_xyzds.device)
            sample_ray_direction =  patch_directions[idx] - sample_ray_direction.unsqueeze(-1).unsqueeze(-1)
            sample_ray_direction = torch.cat((torch.sin(sample_ray_direction).unsqueeze(-1), torch.cos(sample_ray_direction).unsqueeze(-1)), dim=-1)

            sample_ft_neighbor_xyzds[...,3:5] = sample_ray_direction
            sample_ft_neighbor_xyzds[...,3:5][idx==-1] = 0
            sample_ft_neighbor_xyzds[...,5:] = patch_scales[idx]
            sample_ft_neighbor_xyzds[...,5:][idx==-1] = 0

            sample_ft_neighbor_embedding = scene_fts[idx.cpu().numpy()]
            sample_ft_neighbor_embedding = torch.tensor(sample_ft_neighbor_embedding,dtype=torch.float16,device=self.device)
            sample_ft_neighbor_embedding[idx==-1] = 0

            sample_feature, sample_density = self.patch_to_nerf_encode(sample_ft_neighbor_embedding, sample_ft_neighbor_xyzds)
            rel_dist = torch.tensor(rel_dist,dtype=torch.float16,device=self.device)
            feature_map, depth_map = self.raw2feature(sample_feature, sample_density, rel_dist.view(-1,self.args.N_samples), topk_inds)
            batch_rendered_patch_fts.append(feature_map.unsqueeze(0))

        batch_rendered_patch_fts = torch.cat(batch_rendered_patch_fts,dim=0).view(self.batch_size, self.args.view_height, self.args.view_width,-1)
        batch_rendered_patch_positions = torch.cat(batch_rendered_patch_positions,dim=0).view(self.batch_size, self.args.view_height, self.args.view_width,-1)

        return batch_rendered_patch_fts, batch_rendered_patch_positions, batch_gt_label



    def render_panoramic_3d_patch(self, batch_position=None, batch_heading=None, batch_rot=None, batch_trans=None, visualization=False):
        view_num = 4
        batch_panorama_patch_fts = []
        batch_panorama_patch_positions = []
        for view_id in range(view_num):
            batch_view_heading_angle = []
            batch_view_position = []
            batch_view_rot = []
            batch_view_trans = []
            for batch_id in range(self.batch_size):
                if self.mode == 'habitat':
                    position = batch_position[batch_id]
                    camera_direction = batch_heading[batch_id]
                    batch_view_position.append(position)
                    
                    view_heading_angle = (camera_direction + view_id*(-math.pi/2) + math.pi * 3. / 4. ) % (2.*math.pi) # For 3d patchs of panorama, starting from the rays directly behind the agent and proceeding clockwise
                    batch_view_heading_angle.append(view_heading_angle)
                else:
                    R = batch_rot[batch_id]
                    T = batch_trans[batch_id]
                    batch_view_trans.append(T)
                    
                    pano_angle = view_id*(-math.pi/2) + math.pi * 3. / 4. # For 3d patchs of panorama, starting from the rays directly behind the agent and proceeding clockwise
                    pano_R = np.array([[math.cos(pano_angle),-math.sin(pano_angle),0.],
                                        [math.sin(pano_angle),math.cos(pano_angle),0.],
                                        [0.,0.,1.]]) # Rotation for panorama
                    R = pano_R @ R
                    batch_view_rot.append(R)
                

            with torch.no_grad(): # No grad for saving GPU memory
                if self.mode == 'habitat':
                    batch_rendered_patch_fts, batch_rendered_patch_positions, _ = self.render_view_3d_patch(batch_position=batch_view_position, batch_heading=batch_view_heading_angle, visualization=visualization)
                    batch_panorama_patch_fts.append(batch_rendered_patch_fts)
                    batch_panorama_patch_positions.append(batch_rendered_patch_positions)
                else:
                    batch_rendered_patch_fts, batch_rendered_patch_positions,  _ = self.render_view_3d_patch(batch_rot=batch_view_rot, batch_trans=batch_view_trans, visualization=visualization)
                    batch_panorama_patch_fts.append(batch_rendered_patch_fts)
                    batch_panorama_patch_positions.append(batch_rendered_patch_positions)

        batch_panorama_patch_fts = torch.cat(batch_panorama_patch_fts,dim=2)
        batch_panorama_patch_positions = torch.cat(batch_panorama_patch_positions,dim=2)
        return batch_panorama_patch_fts, batch_panorama_patch_positions


    def delete_old_features_from_camera_frustum(self, batch_depth, batch_position=None, batch_heading=None, batch_camera_intrinsic=None, batch_extrinsic=None, view_ids=None):
        zone_x_length, zone_y_length, zone_z_length = self.args.zone_x_length, self.args.zone_y_length, self.args.zone_z_length
        if view_ids == None:
            view_ids =  torch.tensor([0],device=self.device)

        for b in range(self.batch_size): # Use prange for parallel
            if batch_extrinsic is not None:
                view_ids = torch.tensor([i for i in range(batch_depth[b].shape[0])],device=self.device)
            if batch_position is not None:
                position = batch_position[b].copy()
                position[0], position[1], position[2] = batch_position[b][0], - batch_position[b][2], batch_position[b][1] # Note to swap y,z axis, - y
                heading_angle = batch_heading[b]

            for ix in range(len(view_ids)): # rotation for panorama
                if len(self.global_patch_position[b])==0:
                    continue

                if batch_extrinsic is not None:
                    frustum_mask, frustum_depth, u, v = get_frustum_mask(self.global_patch_position[b], batch_depth[b][ix].shape[-2], batch_depth[b][ix].shape[-1], batch_camera_intrinsic[b][ix], batch_extrinsic[b][ix], far=self.args.deleted_frustum_distance)

                elif batch_position is not None:
                    frustum_mask, frustum_depth, u, v = get_frustum_mask_habitat(self.global_patch_position[b], batch_depth[b][ix].shape[-2],batch_depth[b][ix].shape[-1],self.args.input_vfov,self.args.input_hfov,position,view_ids[ix].cpu().numpy()*(-math.pi/6)+heading_angle, far=self.args.deleted_frustum_distance)
                
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
                        self.global_gt_instance_ids[b][instance_id] = -10000 # inf

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
            self.patch_tree[b] = self.get_patch_tree(b)



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
    

    def update_feature_fields(self, batch_depth, batch_grid_ft, batch_image, batch_image_ft=None, batch_position=None, batch_heading=None, batch_camera_intrinsic=None, batch_rot=None, batch_trans=None, depth_scale=1000.0, depth_trunc=1000.0, view_ids=None, is_training=True):
        
        batch_grid_ft = [grid_ft.astype(np.float16) for grid_ft in batch_grid_ft]
        if view_ids == None:
            view_ids =  torch.tensor([0],device=self.device)

        if batch_camera_intrinsic is not None: # Most 3D datasets
            # Calculate the camera intrinsic for novel view rendering
            init_camera_intrinsic = batch_camera_intrinsic[0][0].copy()
            init_camera_intrinsic[0][0] *= (self.args.view_width / batch_depth[0].shape[-1])
            init_camera_intrinsic[1][1] *= (self.args.view_height / batch_depth[0].shape[-2])
            init_camera_intrinsic[0, 2] = self.args.view_width / 2
            init_camera_intrinsic[1, 2] = self.args.view_height / 2
            self.sampled_rays = self.get_rays(init_camera_intrinsic)
            batch_image = batch_image.reshape((-1,batch_image.shape[-3],batch_image.shape[-2],batch_image.shape[-1]))
            batch_patch_segm = self.get_patch_segm(batch_image)
            batch_patch_segm = batch_patch_segm.view(self.batch_size, -1, batch_patch_segm.shape[-2], batch_patch_segm.shape[-1])

        if batch_position is not None: # Habitat simulator
            self.sampled_rays = self.get_rays_habitat()
            batch_image = batch_image.reshape((-1,batch_image.shape[-3],batch_image.shape[-2],batch_image.shape[-1]))
            batch_patch_segm = self.get_patch_segm(batch_image)
            batch_patch_segm = batch_patch_segm.view(self.batch_size, len(view_ids), batch_patch_segm.shape[-2], batch_patch_segm.shape[-1])

        zone_x_length, zone_y_length, zone_z_length = self.args.zone_x_length, self.args.zone_y_length, self.args.zone_z_length
        segm_loss = 0.
        count_segm_loss = 0
        target_2d_instance_fts = []
        predicted_2d_instance_fts = []
        target_instance_subspace_fts = []
        predicted_instance_subspace_fts = []
        target_2d_zone_fts = []
        predicted_2d_zone_fts = []
        target_zone_subspace_fts = []
        predicted_zone_subspace_fts = []

        batch_gt_3d_instance_id = [[] for b in range(self.batch_size)]
        batch_predicted_3d_instancs_fts = [[] for b in range(self.batch_size)]
        batch_gt_3d_instance_ids_in_zone = [[] for b in range(self.batch_size)]
        batch_predicted_3d_zone_fts = [[] for b in range(self.batch_size)]

        for b in range(self.batch_size): # use prange for parallel

            if batch_camera_intrinsic is not None: # Most 3D datasets
                thread_output = self.thread_pool([ [project_depth_to_3d, [ batch_depth[b][job_id], batch_camera_intrinsic[b][job_id],depth_scale,depth_trunc,self.args.input_height,self.args.input_width ], {} ] for job_id in range(len(batch_depth[b])) ]) # Parallel computing with multiple CPUs
                view_ids = torch.tensor([i for i in range(batch_depth[b].shape[0])],device=self.device)

            if batch_position is not None: # Habitat simulator
                position = batch_position[b].copy()
                position[0], position[1], position[2] = batch_position[b][0], - batch_position[b][2], batch_position[b][1] # Note to swap y,z axis, - y
                heading = batch_heading[b]
                depth = batch_depth[b]
                depth = depth.reshape((-1,self.args.input_height*self.args.input_width))
            

            for ix in range(len(view_ids)): # rotation for panorama
                instance_fts = []
                gt_instance_ids = []
                instance_position = []
                proposal_num = min(len(self.global_instance_to_patch_dict[b]), self.args.num_proposal_instances)
                if batch_camera_intrinsic is not None: # Most 3D datasets
                    # Get the patch information
                    points, points_mask = thread_output[ix]
                    points = points.astype(np.float32)
                    _, rel_direction, _ = self.sampled_rays
                    
                    patch_scale = points[:,-1] * math.fabs(math.tan(rel_direction[0][-1])) * 2. / self.args.input_width
                    R = batch_rot[b][ix]
                    T = batch_trans[b][ix]
                    points = (R @ points.T + T).T

                    patch_position = torch.tensor(points.astype(np.float32),device=self.device)
                    patch_direction = self.get_heading_angle(points).astype(np.float32)
                    patch_scale = patch_scale.astype(np.float32)
                
                if batch_position is not None: # Habitat simulator
                    # Get the patch information
                    rel_x, rel_y, rel_z, patch_direction, patch_scale = self.project_depth_to_3d_habitat(depth[ix:ix+1],view_ids[ix].cpu().numpy()*(-math.pi/6)+heading)  
                    patch_x = torch.tensor(rel_x + position[0],device=self.device).unsqueeze(-1)
                    patch_y = torch.tensor(rel_y + position[1],device=self.device).unsqueeze(-1)
                    patch_z = torch.tensor(rel_z + position[2],device=self.device).unsqueeze(-1)
                    patch_position = torch.cat([patch_x,patch_y,patch_z],dim=-1)[0]

                # Update the patch information
                if self.global_patch_position[b] == []:
                    self.global_patch_position[b] = patch_position
                    self.global_patch_scales[b] = patch_scale
                    self.global_patch_directions[b] = patch_direction
                else:
                    self.global_patch_position[b] = torch.cat([self.global_patch_position[b], patch_position],0)
                    
                    self.global_patch_scales[b] = np.concatenate([self.global_patch_scales[b],patch_scale],0)
                    self.global_patch_directions[b] = np.concatenate([self.global_patch_directions[b],patch_direction],0)

                if self.global_patch_fts[b] == []:
                    self.global_patch_fts[b] = batch_grid_ft[b][ix]       
                else:
                    self.global_patch_fts[b] = np.concatenate((self.global_patch_fts[b],batch_grid_ft[b][ix]),axis=0)


                # Obtain the 2D instance information from the current observation
                # patch_position = torch.tensor(patch_position,device=self.device, dtype=torch.float32)
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

                    if is_training:
                        # Instance semantic alignment loss
                        target_2d_instance_fts.append(patch_fts[patch_segm==segm_id].mean(0, keepdim=True))
                        predicted_2d_instance_fts.append(instance_ft)
                        target_instance_subspace_fts.append(patch_fts[patch_segm==segm_id].mean(0, keepdim=True)-patch_fts.mean(0, keepdim=True))
                        predicted_instance_subspace_fts.append(instance_ft-patch_fts.mean(0, keepdim=True))

                    if self.gt_pcd_tree is not None and self.gt_pcd_tree[b] is not None:
                
                        gt_dists, gt_inds = self.gt_pcd_tree[b].query(segm_patch_position,nr_nns_searches=1)
                        #gt_dists = torch.sqrt(gt_dists) #Note that the cupy_kdtree distances are squared
                        gt_instance_id = self.gt_pcd_label[b][gt_inds]

                        unique_vals, counts = torch.unique(gt_instance_id, return_counts=True)
                        gt_instance_id = unique_vals[counts.argmax()]
                        gt_instance_ids.append(gt_instance_id.unsqueeze(0))

                if self.gt_pcd_tree is not None and self.gt_pcd_tree[b] is not None:
                    gt_instance_ids = torch.cat(gt_instance_ids,dim=0)
                    
                instance_fts = torch.cat(instance_fts,dim=0)
                instance_position = torch.cat(instance_position,dim=0)

                if is_training and batch_image_ft is not None:
                    instance_to_center_position = instance_position - instance_position.mean(0,keepdim=True)
                            
                    instance_to_center_distance = torch.sqrt(torch.square(instance_position).sum(-1)).unsqueeze(-1)

                    instance_position_embedding = torch.cat([instance_to_center_position,instance_to_center_distance],dim=-1)
                    
                    instance_fts_set = instance_fts + self.instance_to_zone_position_embedding(instance_position_embedding)
                    
                    instance_fts_set = torch.cat([self.aggregate_instance_to_zone_embedding,instance_fts_set],dim=0)
                    new_zone_fts = self.aggregate_instance_to_zone_encoder(instance_fts_set)

                    predicted_2d_zone_fts.append(new_zone_fts[0:1])
                    target_2d_zone_fts.append(batch_image_ft[b][ix:ix+1])

                    target_zone_subspace_fts.append(batch_image_ft[b][ix:ix+1]-batch_image_ft[b].mean(0,keepdim=True))
                    predicted_zone_subspace_fts.append(new_zone_fts[0:1]-batch_image_ft[b].mean(0,keepdim=True))
                            

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

                    if is_training:
                        if self.gt_pcd_tree is not None and self.gt_pcd_tree[b] is not None:

                            proposal_gt_3d_instance_ids = self.global_gt_instance_ids[b][gt_inds].unsqueeze(0)
                            merge_gt_instance_ids = gt_instance_ids.unsqueeze(-1).repeat(1,1,proposal_num)
                            merge_target = torch.zeros(merge_gt_instance_ids.shape,dtype=torch.int64,device=merge_score.device)
                            merge_target[proposal_gt_3d_instance_ids == merge_gt_instance_ids] = 1
                            
                            # Segmentation loss
                            true_count = len(merge_target[merge_target==1])
                            false_count = len(merge_target[merge_target==0])
                            if true_count != 0 and false_count != 0:
                                min_count = min(true_count,false_count)
                                merge_score = merge_score.view(-1,2)
                                merge_gt = merge_target.view(-1)
                                merge_score = torch.cat([merge_score[merge_gt==1][:min_count],merge_score[merge_gt==0][:min_count]],dim=0)
                                merge_gt = torch.cat([merge_gt[merge_gt==1][:min_count],merge_gt[merge_gt==0][:min_count]],dim=0)
                                segm_loss += torch.nn.functional.cross_entropy(merge_score, merge_gt)
                                count_segm_loss += 1
                            merge_target = merge_target.squeeze(0)
                            

                            # Subspace alignment
                            #subspace_center = proposal_3d_instance_fts.view(-1,self.args.fts_dim).mean(dim=0)
                            #merged_instance_fts = merged_instance_fts - subspace_center
                            #proposal_3d_instance_fts = proposal_3d_instance_fts - subspace_center
                            #merged_instance_fts = merged_instance_fts[merge_target==1]
                            #proposal_3d_instance_fts = proposal_3d_instance_fts[merge_target==1]
                            #merged_instance_fts = merged_instance_fts / (torch.linalg.norm(merged_instance_fts, dim=-1, keepdim=True) + 1e-7)
                            #proposal_3d_instance_fts = proposal_3d_instance_fts / (torch.linalg.norm(proposal_3d_instance_fts, dim=-1, keepdim=True) + 1e-7)
                            #tmp_loss = (1. - (merged_instance_fts * proposal_3d_instance_fts).sum(-1)).mean()
                            #if not torch.any(torch.isnan(tmp_loss)):
                            #    segm_loss += tmp_loss

                        else:
                            merge_target = torch.zeros(torch.argmax(merge_score,dim=-1).shape,dtype=torch.int64,device=merge_score.device)
                    else:
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
                                self.global_instance_fts[b][instance_id] = instance_fts[segm_id].detach()
                                if self.gt_pcd_tree is not None and self.gt_pcd_tree[b] is not None:
                                    self.global_gt_instance_ids[b][instance_id] = gt_instance_ids[segm_id]
                            else:
                                self.global_instance_position[b] = torch.cat([self.global_instance_position[b],instance_position[segm_id:segm_id+1]],dim=0)
                                self.global_instance_fts[b] = torch.cat([self.global_instance_fts[b],instance_fts[segm_id:segm_id+1]],dim=0).detach()
                                if self.gt_pcd_tree is not None and self.gt_pcd_tree[b] is not None:
                                    self.global_gt_instance_ids[b] = torch.cat([self.global_gt_instance_ids[b],gt_instance_ids[segm_id:segm_id+1]],dim=0)

                            batch_predicted_3d_instancs_fts[b].append(instance_fts[segm_id:segm_id+1]) # !!!!!!!!!!!
                            batch_gt_3d_instance_id[b].append(gt_instance_ids[segm_id:segm_id+1])

                        else: # Merge instance
                            for proposal_3d_instance_id in range(proposal_num):
                                if merge_target[segm_id, proposal_3d_instance_id].cpu().numpy().item() != 0: # Only merge into the nearest 3d instance
                                    instance_id = gt_inds[segm_id,proposal_3d_instance_id].cpu().numpy().item()

                                    patch_ids_belong_to_instance = assigned_patch_ids[patch_segm.cpu().numpy()==segm_id]
                                                                          
                                    self.global_instance_to_patch_dict[b][instance_id] = np.concatenate([self.global_instance_to_patch_dict[b][instance_id],patch_ids_belong_to_instance ],axis=0) # Merge all patchs into this instance, update the instance dict
                                    for patch_id in patch_ids_belong_to_instance.tolist():
                                        self.global_patch_to_instance_dict[b][patch_id] = instance_id
                                    
                                    patch_position_set = self.global_patch_position[b][self.global_instance_to_patch_dict[b][instance_id]]
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

                                    if free_memory > 14: # Check the GPU Memory, it's very important to reduce the GPU memory
                                        new_instance_fts = self.aggregate_patch_to_instance_encoder(patch_fts_set)[0:1]
                                    else:
                                        with torch.no_grad():
                                            new_instance_fts = self.aggregate_patch_to_instance_encoder(patch_fts_set.detach())[0:1]

                                    self.global_instance_fts[b][instance_id] = new_instance_fts.detach() # Update merged instance feature

                                    batch_predicted_3d_instancs_fts[b].append(new_instance_fts) # !!!!!!!!!!!
                                    batch_gt_3d_instance_id[b].append(self.global_gt_instance_ids[b][instance_id:instance_id+1])

                                    break # Only merge into the nearest 3d instance

                    # Update the zone information
                    global_zone_position = torch.cat([(self.global_instance_position[b][:,0:1]//zone_x_length)*zone_x_length+zone_x_length/2, (self.global_instance_position[b][:,1:2]//zone_y_length)*zone_y_length+zone_y_length/2, (self.global_instance_position[b][:,2:3]//zone_z_length)*zone_z_length+zone_z_length/2],dim=-1)
                    updated_zone_position = torch.cat([(instance_position[:,0:1]//zone_x_length)*zone_x_length+zone_x_length/2, (instance_position[:,1:2]//zone_y_length)*zone_y_length+zone_y_length/2, (instance_position[:,2:3]//zone_z_length)*zone_z_length+zone_z_length/2],dim=-1)
                    assigned_zone_list, instance_counts = torch.unique(updated_zone_position,return_counts=True,dim=0)
                    assigned_zone_num = len(assigned_zone_list)
                    assigned_zone_ids = self.assign_new_zone_ids(b,assigned_zone_num)
                    updated_zone_position = updated_zone_position.cpu().numpy()

                    # Get the main zone of current observation, for calculating the zone alignment loss
                    main_zone_key = tuple(assigned_zone_list[instance_counts.argmax()].cpu().numpy().tolist())

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

                            self.global_zone_fts[b] = torch.cat([self.global_zone_fts[b],new_zone_fts[0:1].detach()],dim=0) # Add the zone feature
                            
                            batch_predicted_3d_zone_fts[b].append(new_zone_fts) # !!!!!!!!!!!
                            batch_gt_3d_instance_ids_in_zone[b].append(torch.tensor(self.global_zone_to_instance_dict[b][zone_id], device=self.device))
                            

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

                            self.global_zone_fts[b][zone_id] = new_zone_fts[0:1].detach() # Update the zone feature

                            batch_predicted_3d_zone_fts[b].append(new_zone_fts) # !!!!!!!!!!!
                            batch_gt_3d_instance_ids_in_zone[b].append(torch.tensor(self.global_zone_to_instance_dict[b][zone_id], device=self.device))

                    
                else:
                    # Initialize the 3d instances information
                    self.global_instance_position[b].append(instance_position)
                    self.global_instance_position[b] = torch.cat(self.global_instance_position[b],dim=0)
                    self.global_instance_fts[b].append(instance_fts.detach())
                    self.global_instance_fts[b] = torch.cat(self.global_instance_fts[b],dim=0)

                    batch_predicted_3d_instancs_fts[b].append(instance_fts) # !!!!!!!!!!!
                    batch_gt_3d_instance_id[b].append(gt_instance_ids)

                    assigned_instance_ids = self.assign_new_instance_ids(b, instance_fts.shape[0])
                    assigned_patch_ids = self.assign_new_patch_ids(b, patch_fts.shape[-2])
                    
                    for segm_id in torch.unique(patch_segm).cpu().numpy().tolist():
                        patch_ids_belong_to_instance = assigned_patch_ids[patch_segm.cpu().numpy()==segm_id]
                        instance_id = assigned_instance_ids[segm_id].item()
                        self.global_instance_to_patch_dict[b][instance_id] = patch_ids_belong_to_instance
                        for patch_id in patch_ids_belong_to_instance.tolist():
                            self.global_patch_to_instance_dict[b][patch_id] = instance_id

                    if self.gt_pcd_tree is not None and self.gt_pcd_tree[b] is not None:
                        self.global_gt_instance_ids[b].append(gt_instance_ids)
                        self.global_gt_instance_ids[b] = torch.cat(self.global_gt_instance_ids[b],dim=0)  

                    # Initialize the zone information
                    updated_zone_position = torch.cat([(instance_position[:,0:1]//zone_x_length)*zone_x_length+zone_x_length/2, (instance_position[:,1:2]//zone_y_length)*zone_y_length+zone_y_length/2, (instance_position[:,2:3]//zone_z_length)*zone_z_length+zone_z_length/2],dim=-1)
                    assigned_zone_list, instance_counts = torch.unique(updated_zone_position,return_counts=True,dim=0)
                    assigned_zone_num = len(assigned_zone_list)
                    assigned_zone_ids = self.assign_new_zone_ids(b,assigned_zone_num)
                    updated_zone_position = updated_zone_position.cpu().numpy()

                    # Get the main zone of current observation, for calculating the zone alignment loss
                    main_zone_key = tuple(assigned_zone_list[instance_counts.argmax()].cpu().numpy().tolist())

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
                        
                        self.global_zone_fts[b].append(new_zone_fts[0:1].detach()) # Add the zone feature

                        batch_predicted_3d_zone_fts[b].append(new_zone_fts) # !!!!!!!!!!!
                        batch_gt_3d_instance_ids_in_zone[b].append(torch.tensor(self.global_zone_to_instance_dict[b][zone_id], device=self.device))


                    self.global_zone_position[b] = torch.cat(self.global_zone_position[b],dim=0)
                    self.global_zone_fts[b] = torch.cat(self.global_zone_fts[b],dim=0)

                # Update the kd-tree
                self.instance_tree[b] = self.get_instance_tree(b)
                self.patch_tree[b] = self.get_patch_tree(b)

            # Visualization
            '''
            feature_pcd=o3d.geometry.PointCloud()
            feature_pcd.points = o3d.utility.Vector3dVector(self.global_patch_position[b][torch.abs(self.global_patch_position[b]).sum(-1)<1000].cpu().numpy())
            if self.gt_pcd_xyz is not None:
                gt_pcd=o3d.geometry.PointCloud()
                gt_pcd.points = o3d.utility.Vector3dVector(self.gt_pcd_xyz[b])
                feature_pcd += gt_pcd
            o3d.visualization.draw_geometries([feature_pcd])
            
            #print(b,len(self.global_zone_to_instance_dict[b]),len(self.global_zone_key_to_id[b]),len(self.global_instance_to_patch_dict[b]),len(self.global_patch_to_instance_dict[b]))
            '''

        if is_training:
            sim_loss = 0.
            predicted_2d_instance_fts = torch.cat(predicted_2d_instance_fts)
            target_2d_instance_fts = torch.cat(target_2d_instance_fts)
            predicted_2d_instance_fts = predicted_2d_instance_fts / torch.linalg.norm(predicted_2d_instance_fts, dim=-1, keepdim=True)
            target_2d_instance_fts = target_2d_instance_fts / torch.linalg.norm(target_2d_instance_fts, dim=-1, keepdim=True)
            sim_loss += self.contrastive_loss(predicted_2d_instance_fts,target_2d_instance_fts) / 5.
            sim_loss += (1. - (predicted_2d_instance_fts * target_2d_instance_fts).sum(-1)).mean()

            # Subspace alignment
            target_instance_subspace_fts = torch.cat(target_instance_subspace_fts)
            predicted_instance_subspace_fts = torch.cat(predicted_instance_subspace_fts)
            target_instance_subspace_fts = target_instance_subspace_fts / (torch.linalg.norm(target_instance_subspace_fts, dim=-1, keepdim=True) + 1e-7)
            predicted_instance_subspace_fts = predicted_instance_subspace_fts / (torch.linalg.norm(predicted_instance_subspace_fts, dim=-1, keepdim=True) + 1e-7)
            sim_loss += (1. - (predicted_instance_subspace_fts * target_instance_subspace_fts).sum(-1)).mean()

            if batch_image_ft is not None:
                predicted_2d_zone_fts = torch.cat(predicted_2d_zone_fts)
                target_2d_zone_fts = torch.cat(target_2d_zone_fts)
                predicted_2d_zone_fts = predicted_2d_zone_fts / torch.linalg.norm(predicted_2d_zone_fts, dim=-1, keepdim=True)
                target_2d_zone_fts = target_2d_zone_fts / torch.linalg.norm(target_2d_zone_fts, dim=-1, keepdim=True)
                sim_loss += self.contrastive_loss(predicted_2d_zone_fts,target_2d_zone_fts) / 5.
                sim_loss += (1. - (predicted_2d_zone_fts * target_2d_zone_fts).sum(-1)).mean()

                predicted_zone_subspace_fts = torch.cat(predicted_zone_subspace_fts)
                target_zone_subspace_fts = torch.cat(target_zone_subspace_fts)
                if target_zone_subspace_fts.detach().sum().cpu().numpy().item() != 0:
                    predicted_zone_subspace_fts = predicted_zone_subspace_fts / torch.linalg.norm(predicted_zone_subspace_fts, dim=-1, keepdim=True)
                    target_zone_subspace_fts = target_zone_subspace_fts / torch.linalg.norm(target_zone_subspace_fts, dim=-1, keepdim=True)
                    sim_loss += (1. - (predicted_zone_subspace_fts * target_zone_subspace_fts).sum(-1)).mean()

            if self.gt_pcd_tree is not None and self.gt_pcd_tree[b] is not None:
                for b in range(len(batch_gt_3d_instance_id)):
                    batch_gt_3d_instance_id[b] = torch.cat(batch_gt_3d_instance_id[b],dim=0)
                    batch_predicted_3d_instancs_fts[b] = torch.cat(batch_predicted_3d_instancs_fts[b],dim=0)
                    # batch_gt_3d_instance_ids_in_zone[b] = torch.cat(batch_gt_3d_instance_ids_in_zone[b],dim=0)
                    # batch_predicted_3d_zone_fts[b] = torch.cat(batch_predicted_3d_zone_fts[b],dim=0)

            sim_loss += predicted_2d_instance_fts.sum() * 0. + batch_predicted_3d_instancs_fts[0][0].sum() * 0. + self.instance_merge_discriminator(torch.zeros(1,(2*self.args.fts_dim+3),device=self.device)).sum() * 0. # Avoid DDP bug
            if count_segm_loss != 0:
                return sim_loss, segm_loss/count_segm_loss, batch_gt_3d_instance_id,batch_predicted_3d_instancs_fts,batch_gt_3d_instance_ids_in_zone, batch_predicted_3d_zone_fts
            else:
                #print("loss:", sim_loss)
                return sim_loss, 0., batch_gt_3d_instance_id,batch_predicted_3d_instancs_fts,batch_gt_3d_instance_ids_in_zone, batch_predicted_3d_zone_fts
