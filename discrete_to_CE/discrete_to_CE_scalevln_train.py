import json
import math
import MatterSim
import habitat
from habitat import get_config
from habitat.sims import make_sim
from habitat.utils.visualizations import maps
import numpy as np
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
from utils.habitat_utils import HabitatUtils
import matplotlib.pyplot as plt
#import torch.multiprocessing as mp
from multiprocessing import Process, Queue
import time
import os

def build_habitat_sim(scan):
    sim = HabitatUtils(f'data/scene_datasets/hm3d/train/{scan}/'+scan.split('-')[-1]+'.basis.glb', int(0), int(math.degrees(HFOV)), HEIGHT, WIDTH)
    return sim

class HabitatUtils:
    def __init__(self, scene, level, hfov, h, w, housetype='hm3d'):
        # -- scene = data/hm3d/house/house.glb
        self.scene = scene
        self.level = level  # -- int
        self.house = scene.split('/')[-2]
        self.housetype = housetype

        #-- setup config
        self.config = get_config()
        self.config.defrost()
        self.config.SIMULATOR.SCENE = scene
        self.config.SIMULATOR.AGENT_0.SENSORS = ["RGB_SENSOR",
                                                 "DEPTH_SENSOR",
                                                 "SEMANTIC_SENSOR"]
        self.config.SIMULATOR.RGB_SENSOR.HFOV = hfov
        self.config.SIMULATOR.RGB_SENSOR.HEIGHT = h
        self.config.SIMULATOR.RGB_SENSOR.WIDTH = w
        self.config.SIMULATOR.DEPTH_SENSOR.HFOV = hfov
        self.config.SIMULATOR.DEPTH_SENSOR.HEIGHT = h
        self.config.SIMULATOR.DEPTH_SENSOR.WIDTH = w
        self.config.SIMULATOR.SEMANTIC_SENSOR.HFOV = hfov
        self.config.SIMULATOR.SEMANTIC_SENSOR.HEIGHT = h
        self.config.SIMULATOR.SEMANTIC_SENSOR.WIDTH = w
        # self.config.SIMULATOR.AGENT_0.HEIGHT = 0

        # -- Original resolution
        self.config.SIMULATOR.FORWARD_STEP_SIZE = 0.1
        self.config.SIMULATOR.TURN_ANGLE = 9

        # -- fine resolution setps
        #self.config.SIMULATOR.FORWARD_STEP_SIZE = 0.05
        #self.config.SIMULATOR.TURN_ANGLE = 3

        # -- render High Rez images
        #self.config.SIMULATOR.RGB_SENSOR.HEIGHT = 720
        #self.config.SIMULATOR.RGB_SENSOR.WIDTH = 1280

        # -- LOOK DOWN
        #theta = 30 * np.pi / 180
        #self.config.SIMULATOR.RGB_SENSOR.ORIENTATION = [-theta, 0.0, 0.0]
        #self.config.SIMULATOR.DEPTH_SENSOR.ORIENTATION = [-theta, 0.0, 0.0]
        #self.config.SIMULATOR.SEMANTIC_SENSOR.ORIENTATION = [-theta, 0.0, 0.0]

        # -- OUTDATED (might be able to re-instantiate those in future commits)
        #self.config.TASK.POSSIBLE_ACTIONS = ["STOP", "MOVE_FORWARD",
        #                                     "TURN_LEFT", "TURN_RIGHT",
        #                                     "LOOK_UP", "LOOK_DOWN"]

        # -- ObjNav settings
        #self.config.SIMULATOR.FORWARD_STEP_SIZE = 0.25
        #self.config.SIMULATOR.TURN_ANGLE = 30


        self.config.freeze()

        self.agent_height = self.config.SIMULATOR.AGENT_0.HEIGHT

        self.sim = make_sim(id_sim=self.config.SIMULATOR.TYPE, config=self.config.SIMULATOR)

        self.semantic_annotations = self.sim.semantic_annotations()

        self.sim.reset()

        agent_state = self.get_agent_state()
        self.position = agent_state.position
        self.rotation = agent_state.rotation

        # -- get level dimensions
        # -- read it directly from the saved data from the .house files
        # Tries to set the agent on the given floor. It's actually quite hard..
        # if housetype == 'hm3d':
        #     env = '_'.join([self.house, str(self.level)])
        #     houses_dim = json.load(open('data/houses_dim.json', 'r'))
        #     self.center = np.array(houses_dim[env]['center'])
        #     self.sizes = np.array(houses_dim[env]['sizes'])
        #     self.start_height = self.center[1] - self.sizes[1]/2

        #     self.set_agent_on_level()
        # else:
        #     pass

        self.all_objects = self.get_objects_in_house()

    @property
    def position(self):
        return self._position


    @position.setter
    def position(self, p):
        self._position = p


    @property
    def rotation(self):
        return self._rotation


    @rotation.setter
    def rotation(self, r):
        self._rotation = r


    def set_agent_state(self):
        self.sim.set_agent_state(self._position,
                                 self._rotation)

    def get_agent_state(self):
        return self.sim.get_agent_state()


    def get_sensor_pos(self):
        ags = self.sim.get_agent_state()
        return ags.sensor_states['rgb'].position

    def get_sensor_ori(self):
        ags = self.sim.get_agent_state()
        return ags.sensor_states['rgb'].rotation



    def reset(self):
        self.sim.reset()
        agent_state = self.get_agent_state()
        self.position = agent_state.position
        self.rotation = agent_state.rotation

    def set_agent_on_level(self):
        """
        It is very hard to know exactly the z value of a level as levels can
        have stairs and difference in elevation etc..
        We use the level.aabb to get an idea of the z-value of the level but
        that isn't very robust (eg r1Q1Z4BcV1o_0 -> start_height of floor 0:
            -1.3 but all sampled point will have a z-value around 0.07, when
            manually naivagting in the env we can see a pth going downstairs..)
        """
        point = self.sample_navigable_point()
        self.position = point
        self.set_agent_state()

    def step(self, action):
        self.sim.step(action)


    def sample_navigable_point(self):
        """
        If house has only one level we sample directly a nav point
        Else we iter until we get a point on the right floor..
        """
        if len(self.semantic_annotations.levels) == 1:
            return self.sim.sample_navigable_point()
        else:
            for _ in range(1000):
                point = self.sim.sample_navigable_point()
                #return point
                if np.abs(self.start_height - point[1]) <= 1.5:
                #if np.all(((self.center-self.sizes/2)<=point) &
                #          ((self.center+self.sizes/2)>=point)):
                    return point
            print('No navigable point on this floor')
            return None


    def sample_rotation(self):
        theta = np.random.uniform(high=np.pi)
        quat = np.array([0, np.cos(theta/2), 0, np.sin(theta/2)])
        return quat



    def get_house_dimensions(self):
        return self.semantic_annotations.aabb



    def get_objects_in_scene(self):
        """

            returns dict with {int obj_id: #pixels in frame}

        """
        buf = self.sim.render(mode="semantic")
        unique, counts = np.unique(buf, return_counts=True)
        objects = {int(u): c for u, c in zip(unique, counts)}
        return objects


    def render(self, mode='rgb'):
        return self.sim.render(mode=mode)



    def render_semantic_mpcat40(self):
        buf = self.sim.render(mode="semantic")
        out = np.zeros(buf.shape, dtype=np.uint8) # class 0 -> void
        object_ids = np.unique(buf)
        for oid in object_ids:
            object = self.all_objects[oid]
            # -- mpcat40_name = object.category.name(mapping='mpcat40')
            mpcat40_index = object.category.index(mapping='mpcat40')
            # remap everything void/unlabeled/misc/etc .. to misc class 40
            # (void:0,  unlabeled: 41, misc=40)
            if mpcat40_index <= 0 or mpcat40_index > 40: mpcat40_index = 40 # remap -1 to misc
            out[buf==oid] = mpcat40_index
        return out



    def get_objects_in_level(self):
        # /!\ /!\ level IDs are noisy in HM3D
        # /!\ /!\

        if self.housetype == 'hm3d':

            assert self.level == int(self.semantic_annotations.levels[self.level].id)

            objects = {}
            for region in self.semantic_annotations.levels[self.level].regions:
                for object in region.objects:
                    objects[int(object.id.split('_')[-1])] = object
        else:
            objects = self.all_objects

        return objects


    def get_objects_in_house(self):
        objects = {int(o.id.split('_')[-1]): o for o in self.semantic_annotations.objects if o is not None}
        return objects


    def __del__(self):
        self.sim.close()


WIDTH = 336
HEIGHT = 336
VFOV = 90
HFOV = 90

def build_simulator(connectivity_dir):
    sim = MatterSim.Simulator()
    sim.setNavGraphPath(connectivity_dir)
    sim.setRenderingEnabled(False)
    sim.setCameraResolution(WIDTH, HEIGHT)
    sim.setCameraVFOV(math.radians(VFOV))
    sim.setDiscretizedViewingAngles(True)
    sim.setDepthEnabled(False)
    sim.setPreloadingEnabled(False)
    sim.setBatchSize(1)
    sim.initialize()
    return sim

def build_habitat_sim(scan):
    sim = HabitatUtils(f'data/scene_datasets/hm3d/train/{scan}/'+scan.split('-')[-1]+'.basis.glb', int(0), int(math.degrees(HFOV)), HEIGHT, WIDTH)
    return sim


def process_features(item_id, episode_id, discrete_data, out_queue):
    sim = build_simulator("connectivity_hm3d")
    habitat_sim = build_habitat_sim(discrete_data[0]['scan'])

    for item in tqdm(discrete_data[item_id:]):
        scan_id = item['scan']
        path = item['path']
        
        reference_path = []
        for viewpoint_id in path:
            sim.newEpisode([scan_id], [viewpoint_id], [0], [0])
            state = sim.getState()[0]
            # set habitat to the same position & rotation
            x, y, z, h, e = state.location.x, state.location.y, state.location.z, item['heading'], state.elevation #state.heading, state.elevation
            habitat_position = [x, z, -y]
            reference_path.append(habitat_position)

            if len(reference_path) == 1:
                hm3d_h = np.array([0, 2*math.pi-h, 0]) # counter-clock heading
                hm3d_e = np.array([e, 0, 0])
                rotvec_h = R.from_rotvec(hm3d_h)
                rotvec_e = R.from_rotvec(hm3d_e)
                habitat_rotation = (rotvec_h * rotvec_e).as_quat().tolist()

                #habitat_sim.sim.set_agent_state(habitat_position, habitat_rotation)
                #image = np.array(habitat_sim.render('rgb'), copy=True)  # in RGB channel
                #plt.imshow(image)
                #plt.show()

        geodesic_distance = habitat_sim.sim.geodesic_distance(reference_path[0], reference_path[-1])

        if geodesic_distance < 100.:
            pass
        else:
            continue

        episode_id += 1
        item_id += 1
        CE_item = {
            "episode_id":episode_id,
            "trajectory_id":episode_id,
            "scene_id":"hm3d/train/"+item['scan']+"/"+item['scan'][6:]+".basis.glb",
            "start_position":reference_path[0],
            "start_rotation":habitat_rotation,
            "info": {
                "geodesic_distance": geodesic_distance
            },
            "goals": [{"position": reference_path[-1], "radius": 3.0}],
            "instruction": {
                "instruction_text": item['instructions'][0], 
                "instruction_tokens": item['instr_encodings'][0]
            },
            "reference_path": reference_path
        }
        CE_gt_item = {
            "locations":reference_path,
            "forward_steps": len(reference_path)*6,
            "actions": [1] * (len(reference_path)*6)
        }
        out_queue.put((item_id, episode_id, CE_item, CE_gt_item))

    habitat_sim.sim.close()
    exit()


discrete_data = json.load(open("R2R_scalevln_ft_aug_enc.json","r"))
discrete_data_new = {}
for item in discrete_data:
    scan_id = item['scan']
    if scan_id in discrete_data_new:
        discrete_data_new[scan_id].append(item)
    else:
        discrete_data_new[scan_id] = [item]

discrete_data = discrete_data_new
CE_data = {"episodes":[]}
CE_data_gt = {}


episode_id = 100000
key_id = 0
key_id_json = [0]
if os.path.exists("tmp_CE_data.json") and os.path.exists("tmp_CE_data_gt.json") and os.path.exists("key_id.json"):
    CE_data = json.load(open("tmp_CE_data.json","r"))
    CE_data_gt = json.load(open("tmp_CE_data_gt.json","r"))
    key_id_json = json.load(open("key_id.json","r"))
    episode_id += len(CE_data['episodes']) + 1

for key in tqdm(discrete_data):
    key_id += 1
    if key_id <= key_id_json[0]:
        continue

    item_id = 0
    try_again = True
    while True:
        out_queue = Queue()
        process = Process(
            target=process_features,
            args=(item_id, episode_id, discrete_data[key], out_queue)
        )
        process.start()

        if try_again:
            if len(discrete_data[key]) < 2000:
                time.sleep(5)
            elif len(discrete_data[key]) < 10000:
                time.sleep(10)
            elif len(discrete_data[key]) < 20000:
                time.sleep(15)
            elif len(discrete_data[key]) < 40000:
                time.sleep(40)
            elif len(discrete_data[key]) < 100000:
                time.sleep(80)
            else:
                time.sleep(200)
        else:
            if len(discrete_data[key]) < 2000:
                process.join(timeout=2)
            elif len(discrete_data[key]) < 10000:
                process.join(timeout=4)
            elif len(discrete_data[key]) < 20000:
                process.join(timeout=8)
            elif len(discrete_data[key]) < 40000:
                process.join(timeout=16)
            elif len(discrete_data[key]) < 100000:
                process.join(timeout=32)
            else:
                process.join(timeout=64)

        while True:
            print(episode_id)
            if out_queue.empty():
                break
            else:
                res = out_queue.get()
                item_id, episode_id, CE_item, CE_gt_item = res
                CE_data['episodes'].append(CE_item)
                CE_data_gt[str(episode_id)] = CE_gt_item
        process.terminate()

        if item_id > len(discrete_data[key]) - 10:
            break
        if try_again:
            item_id += 1
            try_again = False
        else:
            break

    key_id_json = [key_id]
    if key_id % 25 == 0:
        json.dump(CE_data, open("tmp_CE_data.json","w"))
        json.dump(CE_data_gt, open("tmp_CE_data_gt.json","w"))
        json.dump(key_id_json, open("key_id.json","w"))
        exit()

episode_id = 0
# For R2R-CE Data
r2r_ce_data = json.load(open("train_bertidx.json","r"))['episodes']
r2r_ce_data_gt = json.load(open("train_gt_all.json","r"))

for i in tqdm(range(len(r2r_ce_data))):
    episode_id += 1
    r2r_ce_data[i]['episode_id'] = episode_id

CE_data['episodes'].extend(r2r_ce_data)
CE_data_gt.update(r2r_ce_data_gt)

scene_dict = {}
for item in CE_data['episodes']:
    if item['scene_id'] in scene_dict:
        scene_dict[item['scene_id']].append(item)
    else:
        scene_dict[item['scene_id']] = [item]

for scene_id in scene_dict:
    json.dump(scene_dict[scene_id], open("r2r_scalevln_training_data/"+scene_id.split("/")[-1]+".json","w"))

json.dump(CE_data_gt, open("r2r_scalevln_train_gt.json","w"))