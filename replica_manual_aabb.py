# Adapted from https://github.com/shamitlal/HabitatScripts/blob/master/replica_manual_aabb.py

import habitat_sim
from habitat_sim.utils.common import quat_to_coeffs
import cv2

import random
#%matplotlib inline
import matplotlib.pyplot as plt
import time
import numpy as np
import ipdb
st = ipdb.set_trace
import os 
import sys
import pickle
import json
import skimage.io
import argparse
import progressbar
import pickle as pkl
from scipy.spatial.transform import Rotation
from utils import quat_from_coeffs, quat_to_coeffs
from datetime import datetime

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--mapname", default="frl_apartment_0", help="Name of the map in Replica dataset")
# parser.add_argument("-p", "--datapath", default="/media/rpl/Data/mscvLongterm/habitat/Replica-Dataset/out", help="path to the replica dataset")
# parser.add_argument("-o", "--outpath", default="/media/rpl/Data/mscvLongterm/habitat/generated", help="path to the replica dataset")
parser.add_argument("-p", "--datapath", default="/media/prakhar/BIG_BAG/Replica-Dataset", help="path to the replica dataset")
parser.add_argument("-o", "--outpath", default="/media/prakhar/BIG_BAG/Capstone/habitat", help="path to the replica dataset")
args = parser.parse_args()

mapname = args.mapname
test_scene = "{}/{}/habitat/mesh_semantic.ply".format(args.datapath, mapname)
object_json = "{}/{}/habitat/info_semantic.json".format(args.datapath, mapname)

# Read world transform np
# TODO
with open(args.mapname[:-2]+"_transforms.pkl","rb") as fp:
    dict_transforms = pkl.load(fp)
Twl = dict_transforms[args.mapname]
# Twl = np.linalg.inv(Twl)
Tlw = np.linalg.inv(Twl)

def World2Local(p_w, r_w_quat):
    #r_w: [x,y,z,w]
    #np.quat form: [w,x,y,z]
    # st()
    r_w = quat_to_coeffs(r_w_quat)
    P = p_w.reshape((3,1))
    P = np.concatenate((P, np.ones((1,1))), axis=0)
    Pl = Tlw@P
    p_l = Pl[:3].flatten()

    # TODO
    Rw = Rotation.from_quat(r_w)
    Rwmat = Rw.as_dcm()
    # st()
    # print(Rwmat)
    Rlmat = Tlw[:3,:3]@Rwmat
    Rl = Rotation.from_dcm(Rlmat)
    r_l = Rl.as_quat()
    quat_r_l = quat_from_coeffs(r_l)
    print(p_l, quat_r_l)
    return p_l, quat_r_l

def Local2World(p_l, r_l_quat):
    # TODO
    r_l = quat_to_coeffs(r_l_quat)
    Pl = p_l.reshape((3,1))
    Pl = np.concatenate((Pl, np.ones((1,1))), axis=0)
    Pw = Twl@Pl
    p_w = Pw[:3].flatten()

    # TODO
    Rl = Rotation.from_quat(r_l)
    Rlmat = Rl.as_dcm()
    Rwmat = Twl[:3,:3]@Rlmat
    Rw = Rotation.from_dcm(Rwmat)
    r_w = Rw.as_quat()
    print(p_w, r_w)
    quat_r_w = quat_from_coeffs(r_w)
    return p_w, quat_r_w


# object_id_to_obj_map = get_obj_id_to_obj_info_map(object_json)
ignore_classes = ['base-cabinet','beam','blanket','blinds','cloth','clothing','coaster','comforter','curtain','ceiling','countertop','floor','handrail','mat','paper-towel','picture','pillar','pipe','scarf','shower-stall','switch','tissue-paper','towel','vent','wall','wall-plug','window','rug','logo','set-of-clothing']
sim_settings = {
    "width": 480,  # Spatial resolution of the observations
    "height": 480,
    "scene": test_scene,  # Scene path
    "default_agent": 0,
    "sensor_height": 1.5,  # Height of sensors in meters
    "color_sensor": True,  # RGB sensor
    "semantic_sensor": True,  # Semantic sensor
    "depth_sensor": True,  # Depth sensor
    "seed": 1,
}


FORWARD_KEY="w"
LEFT_KEY="a"
RIGHT_KEY="d"
BACK_KEY="s"
QUIT="q"
# Always save
# SAVE = "o"

# Replaced with up and down arrow
# UP = "u" 
# DOWN = "l"
QUIT = "q"

def make_cfg(settings):
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.gpu_device_id = 0
    sim_cfg.scene.id = settings["scene"]

    # Note: all sensors must have the same resolution
    sensors = {
        "color_sensor": {
            "sensor_type": habitat_sim.SensorType.COLOR,
            "resolution": [settings["height"], settings["width"]],
            "position": [0.0, settings["sensor_height"], 0.0],
        },
        "depth_sensor": {
            "sensor_type": habitat_sim.SensorType.DEPTH,
            "resolution": [settings["height"], settings["width"]],
            "position": [0.0, settings["sensor_height"], 0.0],
        },
        "semantic_sensor": {
            "sensor_type": habitat_sim.SensorType.SEMANTIC,
            "resolution": [settings["height"], settings["width"]],
            "position": [0.0, settings["sensor_height"], 0.0],
        },
    }

    sensor_specs = []
    for sensor_uuid, sensor_params in sensors.items():
        if settings[sensor_uuid]:
            sensor_spec = habitat_sim.SensorSpec()
            sensor_spec.uuid = sensor_uuid
            sensor_spec.sensor_type = sensor_params["sensor_type"]
            sensor_spec.resolution = sensor_params["resolution"]
            sensor_spec.position = sensor_params["position"]

            sensor_specs.append(sensor_spec)

    # Here you can specify the amount of displacement in a forward action and the turn angle
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = sensor_specs
    agent_cfg.action_space = {
        "move_forward": habitat_sim.agent.ActionSpec(
            "move_forward", habitat_sim.agent.ActuationSpec(amount=0.05)
        ),
        "move_back": habitat_sim.agent.ActionSpec(
            "move_forward", habitat_sim.agent.ActuationSpec(amount=-0.5)
        ),
        "turn_left": habitat_sim.agent.ActionSpec(
            "turn_left", habitat_sim.agent.ActuationSpec(amount=2.0)
        ),
        "turn_right": habitat_sim.agent.ActionSpec(
            "turn_right", habitat_sim.agent.ActuationSpec(amount=2.0)
        ),
        "look_up":habitat_sim.ActionSpec(
            "look_up", habitat_sim.ActuationSpec(amount=2.0)
        ),
        "look_down":habitat_sim.ActionSpec(
            "look_down", habitat_sim.ActuationSpec(amount=2.0)
        )
    }

    return habitat_sim.Configuration(sim_cfg, [agent_cfg])

cfg = make_cfg(sim_settings)
sim = habitat_sim.Simulator(cfg)

random.seed(sim_settings["seed"])
sim.seed(sim_settings["seed"])

# Set agent state
agent = sim.initialize_agent(sim_settings["default_agent"])
agent_state = habitat_sim.AgentState()

# st()

initial_position_world = np.array([ 1.5177672, -1.4410627,  4.8951297])
initial_quat_world = quat_from_coeffs(np.array([0, -0.469471454620361, 0, -0.882947683334351]))

initial_position, initial_quat = World2Local(initial_position_world, initial_quat_world)

agent_state.position = initial_position
agent_state.rotation = initial_quat
agent.set_state(agent_state)

# Get agent state
agent_state = agent.get_state()
print("agent_state: position", agent_state.position, "rotation", agent_state.rotation)

from PIL import Image
from habitat_sim.utils.common import d3_40_colors_rgb

def display_sample(rgb_obs, semantic_obs, depth_obs, visualize=False):
    rgb_img = Image.fromarray(rgb_obs, mode="RGBA")

    semantic_img = Image.new("P", (semantic_obs.shape[1], semantic_obs.shape[0]))
    semantic_img.putpalette(d3_40_colors_rgb.flatten())
    semantic_img.putdata((semantic_obs.flatten() % 40).astype(np.uint8))
    semantic_img = semantic_img.convert("RGBA")
    # st()
    
    
    depth_img = Image.fromarray((depth_obs / 10 * 255).astype(np.uint8), mode="L")

    display_img = cv2.cvtColor(np.asarray(rgb_img), cv2.COLOR_RGB2BGR)

    #display_img = cv2.
    cv2.imshow('img',display_img)
    if visualize:
        arr = [rgb_img, semantic_img, depth_img]
        titles = ['rgb', 'semantic', 'depth']
        plt.figure(figsize=(12 ,8))
        for i, data in enumerate(arr):
            ax = plt.subplot(1, 3, i+1)
            ax.axis('off')
            ax.set_title(titles[i])
            plt.imshow(data)
            # plt.pause()
        plt.show()
        plt.pause(0.5)
        # cv2.imshow()
        plt.close()


# object_id_to_obj_map = {int(obj.id.split("_")[-1]): obj for obj in sim.semantic_scene.objects}

def save_datapoint(agent, observations, data_path, timestamp:str, assoc_file, gt_file):
    
    print("Print Sensor States.",agent.state.sensor_states)
    rgb = observations["color_sensor"]
    semantic = observations["semantic_sensor"]
    
    # Extract objects from instance segmentation
    object_list = []
    obj_ids = np.unique(semantic[50:200, 50:200])
    print("Unique semantic ids: ", obj_ids)

    # st()
    for obj_id in obj_ids:
        if obj_id < 1 or obj_id > len(sim.semantic_scene.objects):
            continue
        if sim.semantic_scene.objects[obj_id] == None:
            continue
        if sim.semantic_scene.objects[obj_id].category == None:
            continue
        try:
            class_name = sim.semantic_scene.objects[obj_id].category.name()
            # print("Class name is : ", class_name)
        except Exception as e:
            print(e)
            st()
            print("done")
        if class_name not in ignore_classes:
            obj_instance = sim.semantic_scene.objects[obj_id]
            # print("Object name {}, Object category id {}, Object instance id {}".format(class_name, obj_instance['id'], obj_instance['class_id']))

            obj_data = {'instance_id': obj_id, 'category_id': obj_instance.category.index(), 'category_name': obj_instance.category.name(), 'bbox_center': obj_instance.obb.to_aabb().center, 'bbox_size': obj_instance.obb.to_aabb().sizes}
            # object_list.append(obj_instance)
            object_list.append(obj_data)

    # st()
    depth = observations["depth_sensor"]
    # display_sample(rgb, semantic, depth, visualize=True)
    agent_pos = agent.state.position
    agent_rot = agent.state.rotation
    # Assuming all sensors have same extrinsics
    color_sensor_pos = agent.state.sensor_states['color_sensor'].position
    color_sensor_rot = agent.state.sensor_states['color_sensor'].rotation
    
    # Using pickle since faster
    # save rgb
    color_path = os.path.join('color', timestamp+".png")
    # skimage.io.imsave(color_path, save_data['rgb_camX'])
    # save depth
    depth_path = os.path.join('depth', timestamp+".png")
    # skimage.io.imsave(depth_path, save_data['depth_camX'])

    # save_data = {'objects_info': object_list,'rgb_camX':rgb, 'depth_camX': depth, 'semantic_camX': semantic, 'agent_pos':agent_pos, 'agent_rot': agent_rot, 'sensor_pos': color_sensor_pos, 'sensor_rot': color_sensor_rot}
    save_data = {'timestamp':timestamp, 'rgb_camX':rgb, 'depth_camX': depth, 'sensor_pos': color_sensor_pos, 'sensor_rot': color_sensor_rot}

    assoc_file.write(f"{timestamp} {color_path} {timestamp} {depth_path}\n")
    color_sensor_pos_global, color_sensor_rot_global = Local2World(color_sensor_pos, color_sensor_rot)
    px, py, pz = color_sensor_pos_global
    qx, qy, qz, qw = quat_to_coeffs(color_sensor_rot_global)

    gt_file.write(f"{timestamp} {px} {py} {pz} {qx} {qy} {qz} {qw}\n")

    pickle_filename = os.path.join(data_path, timestamp + ".p")
    with open(pickle_filename, 'wb') as f:
        pickle.dump(save_data, f)
    return pickle_filename

total_frames = 0
action_names = list(
    cfg.agents[
        sim_settings["default_agent"]
    ].action_space.keys()
)

max_frames = 1000000
plt.figure(figsize=(12 ,8))
start_flag = 0
num_saved = 0
frame_rate = 10
time_increment = 1/frame_rate
# total_views_per_scene = 6 # Keep first 3 to be camR (0 elevation)

data_folder = None
data_path = None
basepath = args.outpath

data_folder = datetime.now().strftime("%Y%m%d_h%H_m%M_s%S_") + mapname

timestamp = float(time.time())
data_path = os.path.join(basepath, data_folder)
os.mkdir(data_path)
color_path = os.path.join(data_path, "color")
os.mkdir(color_path)
depth_path = os.path.join(data_path, "depth")
os.mkdir(depth_path)

assoc_filename = "assoc.txt"
gt_filename = "gt.txt"
assoc_file = open(os.path.join(data_path, assoc_filename), 'w')
gt_file = open(os.path.join(data_path, gt_filename), 'w')

pickle_filenames = []


# cat = [obj.category for obj in sim.semantic_scene.objects if obj!=None]
# idx = [c.index() for c in cat if c!=None]
# # st()
# for obj in sim.semantic_scene.objects:
#     if obj != None:
#         if np.sum(obj.aabb.center) != 0:
#             st()
#             print("success")
lskeys = ["w","d"]*12 
lskeys += ["q"]
lskeys = [ord(ch) for ch in lskeys]
curr_ind = 0

while total_frames < max_frames:

    action = "move_forward"
    # st()

    if(start_flag == 0):
        start_flag = 1
        observations = sim.step(action)
        rgb = observations["color_sensor"]
        semantic = observations["semantic_sensor"]
        depth = observations["depth_sensor"]
        display_sample(rgb, semantic, depth)

    
    keystroke = cv2.waitKey(0)
    # keystroke = lskeys[curr_ind]
    curr_ind+=1

    print("keystroke: ", keystroke)

    if( 255!=keystroke and keystroke!=(-1) ):  


        if keystroke == ord(FORWARD_KEY):
            action = "move_forward"
            print("action: FORWARD")
        elif keystroke == ord(BACK_KEY):
            action = "move_back"
            print("action: BACK")
        elif keystroke == ord(LEFT_KEY):
            action = "turn_left"
            print("action: LEFT")
        elif keystroke == ord(RIGHT_KEY):
            action = "turn_right"
            print("action: RIGHT")
        elif keystroke == ord(QUIT):
            action = "turn_right"
            print("action: QUIT")
            break
        # elif keystroke == ord(SAVE):
        #     action = "save_data"
        #     print("action: SAVE")
        elif keystroke == 82:
            action = "look_up"
            print("action: look up")
        elif keystroke == 84:
            action = "look_down"
            print("action: look down")
        else:
            print(keystroke)
            print("INVALID KEY")
            continue
        


        
        print("action", action)
        if action != "save_data":
            print("Performing action")
            observations = sim.step(action)
            print("agent_state: position", agent.state.position, "rotation", agent.state.rotation)
            rgb = observations["color_sensor"]
            semantic = observations["semantic_sensor"]
            depth = observations["depth_sensor"]
            display_sample(rgb, semantic, depth)
        
        # if num_saved % total_views_per_scene == 0:
        #     # Create new directory
        #     data_folder = str(int(time.time()))
        #     data_path = os.path.join(basepath, data_folder)
        #     os.mkdir(data_path)
        # st()
        timestamp_str = "{:.6f}".format(timestamp)
        pickle_filename = save_datapoint(agent, observations, data_path, timestamp_str, assoc_file, gt_file)
        pickle_filenames.append(pickle_filename)
        timestamp += time_increment
        num_saved += 1

        
        total_frames += 1

bar = progressbar.ProgressBar(max_value=len(pickle_filenames)-1)
for i, pickle_filename in enumerate(pickle_filenames):
    with open(pickle_filename, 'rb') as f:
        save_data = pickle.load(f)
    os.remove(pickle_filename)
    timestamp = save_data['timestamp']
    # save rgb
    color_path = os.path.join(data_path, 'color', timestamp+".png")
    im_rgb = save_data['rgb_camX'].astype(np.uint8)
    skimage.io.imsave(color_path, im_rgb)
    # save depth
    depth_path = os.path.join(data_path, 'depth', timestamp+".png")
    im_d = save_data['depth_camX']
    im_d = np.uint16(im_d*1000)
    skimage.io.imsave(depth_path, im_d)
    bar.update(i)
    

assoc_file.close()
gt_file.close()