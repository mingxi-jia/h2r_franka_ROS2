import numpy as np
import torch
import argparse

def strToBool(value):
    if value.lower() in {'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
        return True
    # raise ValueError(f'{value} is not a valid boolean value')

parser = argparse.ArgumentParser()
env_group = parser.add_argument_group('environment')
env_group.add_argument('--env', type=str, default='block_stacking', help='block_picking, block_stacking, brick_stacking, '
                                                                         'brick_inserting, block_cylinder_stacking')
env_group.add_argument('--reward_type', type=str, default='sparse')
env_group.add_argument('--simulator', type=str, default='pybullet')
env_group.add_argument('--robot', type=str, default='ur5')
env_group.add_argument('--num_objects', type=int, default=3)
env_group.add_argument('--max_episode_steps', type=int, default=10)
env_group.add_argument('--fast_mode', type=strToBool, default=True)
env_group.add_argument('--simulate_grasp', type=strToBool, default=True)
env_group.add_argument('--action_sequence', type=str, default='xyrp')
env_group.add_argument('--random_orientation', type=strToBool, default=True)
env_group.add_argument('--place_rot', action='store_true')
env_group.add_argument('--num_processes', type=int, default=5)
env_group.add_argument('--render', type=strToBool, default=False)
# env_group.add_argument('--heightmap_size', type=int, default=100)
env_group.add_argument('--perfect_grasp', action='store_true')
env_group.add_argument('--perfect_place', action='store_true')
env_group.add_argument('--tiny', action='store_true')
env_group.add_argument('--num_random_objects', type=int, default=0)
env_group.add_argument('--in_hand_mode', type=str, default='raw', choices=['raw', 'proj'])

training_group = parser.add_argument_group('training')
training_group.add_argument('--alg', default='dqn')
training_group.add_argument('--model', type=str, default='df')
training_group.add_argument('--num_rotations', type=int, default=8)
training_group.add_argument('--half_rotation', type=strToBool, default=True)
training_group.add_argument('--lr', type=float, default=5e-5)
training_group.add_argument('--gamma', type=float, default=0.5)
training_group.add_argument('--explore', type=int, default=10000)
training_group.add_argument('--fixed_eps', action='store_true')
training_group.add_argument('--init_eps', type=float, default=1.0)
training_group.add_argument('--final_eps', type=float, default=0.)
training_group.add_argument('--init_coef', type=float, default=0.1)
training_group.add_argument('--final_coef', type=float, default=0.01)
training_group.add_argument('--training_iters', type=int, default=1)
training_group.add_argument('--training_offset', type=int, default=1000)
training_group.add_argument('--max_episode', type=int, default=50000)
training_group.add_argument('--device_name', type=str, default='cuda')
training_group.add_argument('--target_update_freq', type=int, default=100)
training_group.add_argument('--iter_update', action='store_true')
training_group.add_argument('--save_freq', type=int, default=500)
training_group.add_argument('--action_selection', type=str, default='egreedy')
training_group.add_argument('--load_pre', type=str, default=None)
training_group.add_argument('--sl', action='store_true')
training_group.add_argument('--expert_sl', action='store_true')
training_group.add_argument('--sl_anneal_episode', type=int, default=20000)
training_group.add_argument('--planner_episode', type=int, default=0)
training_group.add_argument('--note', type=str, default=None)
training_group.add_argument('--seed', type=int, default=None)
training_group.add_argument('--divide_factor', type=int, default=1)
training_group.add_argument('--perlin', type=float, default=0.0)
training_group.add_argument('--fill_buffer', action='store_true')
training_group.add_argument('--fill_buffer_check_sl_valid', action='store_true')
training_group.add_argument('--fill_buffer_deconstruct', action='store_true')
training_group.add_argument('--load_buffer', type=str, default=None)
training_group.add_argument('--load_buffer_pick_z_offset', type=int, default=1)
training_group.add_argument('--load_buffer_place_z_offset', type=int, default=1)
training_group.add_argument('--pre_train_step', type=int, default=0)
training_group.add_argument('--num_zs', type=int, default=16)
training_group.add_argument('--min_z', type=float, default=0.02)
training_group.add_argument('--max_z', type=float, default=0.17)

planner_group = parser.add_argument_group('planner')
planner_group.add_argument('--planner_pos_noise', type=float, default=0)
planner_group.add_argument('--planner_rot_noise', type=float, default=0)

margin_group = parser.add_argument_group('margin')
margin_group.add_argument('--margin', default='ce', choices=['ce', 'bce', 'bcel', 'l', 'oril'])
margin_group.add_argument('--margin_l', type=float, default=0.1)
margin_group.add_argument('--margin_weight', type=float, default=0.1)
margin_group.add_argument('--margin_beta', type=float, default=10)

buffer_group = parser.add_argument_group('buffer')
buffer_group.add_argument('--buffer', default='normal', choices=['normal', 'per', 'expert', 'per_expert'])
buffer_group.add_argument('--per_eps', type=float, default=1e-6, help='Epsilon parameter for PER')
buffer_group.add_argument('--per_alpha', type=float, default=0.6, help='Alpha parameter for PER')
buffer_group.add_argument('--per_beta', type=float, default=0.4, help='Initial beta parameter for PER')
buffer_group.add_argument('--per_expert_eps', type=float, default=0.1)
buffer_group.add_argument('--batch_size', type=int, default=32)
buffer_group.add_argument('--buffer_size', type=int, default=100000)

logging_group = parser.add_argument_group('logging')
logging_group.add_argument('--log_pre', type=str, default='/tmp')
logging_group.add_argument('--log_sub', type=str, default=None)
logging_group.add_argument('--no_bar', action='store_true')
logging_group.add_argument('--time_limit', type=float, default=10000)
logging_group.add_argument('--load_sub', type=str, default=None)

test_group = parser.add_argument_group('test')
test_group.add_argument('--test', action='store_true')

args = parser.parse_args()
# env
random_orientation = args.random_orientation
reward_type = args.reward_type
env = args.env
simulator = args.simulator
num_objects = args.num_objects
max_episode_steps = args.max_episode_steps
fast_mode = args.fast_mode
simulate_grasp = args.simulate_grasp
action_sequence = args.action_sequence
place_rot = args.place_rot
pick_rot = not place_rot
num_processes = args.num_processes
render = args.render
perfect_grasp = args.perfect_grasp
perfect_place = args.perfect_place
tiny = args.tiny
scale = 1.
robot = args.robot

if simulator == 'numpy':
    # workspace = np.asarray([[0., 250.],
    #                         [0., 250.],
    #                         [0., 500.]])
    # workspace_size = 250.
    # heightmap_size = 250
    # height_range = (0., 25.)
    if tiny:
        workspace = np.asarray([[0., 60.],
                                [0., 60.],
                                [0., 500.]])
        workspace_size = 60.
        heightmap_size = 60
        height_range = (0., 16.)
        scale = 1.7
    else:
        workspace = np.asarray([[0., 90.],
                                [0., 90.],
                                [0., 500.]])
        workspace_size = 90.
        heightmap_size = 90
        height_range = (0., 16.)
        scale = 1.

else:
    if tiny:
        workspace = np.asarray([[0.4, 0.6],
                                [-0.1, 0.1],
                                [0, 0.50]])
        workspace_size = workspace[0][1] - workspace[0][0]
        heightmap_size = 60
        height_range = (0.025, 0.10)
    else:
        workspace = np.asarray([[0.35, 0.65],
                                [-0.15, 0.15],
                                [0, 0.50]])
        workspace_size = workspace[0][1] - workspace[0][0]
        heightmap_size = 90
        height_range = (0.025, 0.10)

if env in ['block_picking', 'random_picking', 'random_float_picking', 'cube_float_picking']:
    num_primitives = 1
else:
    num_primitives = 2

heightmap_resolution = workspace_size/heightmap_size
action_space = [0, heightmap_size]

num_rotations = args.num_rotations
half_rotation = args.half_rotation
if half_rotation:
    rotations = [np.pi / num_rotations * i for i in range(num_rotations)]
else:
    rotations = [(2 * np.pi) / num_rotations * i for i in range(num_rotations)]
num_random_objects = args.num_random_objects
in_hand_mode = args.in_hand_mode

######################################################################################
# training
alg = args.alg
if alg == 'dqn_sl_anneal':
    args.sl = True
model = args.model
lr = args.lr
gamma = args.gamma
explore = args.explore
fixed_eps = args.fixed_eps
init_eps = args.init_eps
final_eps = args.final_eps
init_coef = args.init_coef
final_coef = args.final_coef
training_iters = args.training_iters
training_offset = args.training_offset
max_episode = args.max_episode
device = torch.device(args.device_name)
target_update_freq = args.target_update_freq
iter_update = args.iter_update
small_net = heightmap_size < 200
if heightmap_size > 200:
    patch_size = 64
else:
    patch_size = 24
save_freq = args.save_freq
action_selection = args.action_selection
sl = args.sl
expert_sl = args.expert_sl
sl_anneal_episode = args.sl_anneal_episode
planner_episode = args.planner_episode

load_pre = args.load_pre
is_test = args.test
note = args.note
seed = args.seed
divide_factor = args.divide_factor
perlin = args.perlin

# pre train
fill_buffer = args.fill_buffer
fill_buffer_check_sl_valid = args.fill_buffer_check_sl_valid
fill_buffer_deconstruct = args.fill_buffer_deconstruct
load_buffer = args.load_buffer
load_buffer_pick_z_offset = args.load_buffer_pick_z_offset
load_buffer_place_z_offset = args.load_buffer_place_z_offset
pre_train_step = args.pre_train_step

# planner
planner_pos_noise = args.planner_pos_noise
planner_rot_noise = args.planner_rot_noise

# buffer
buffer_type = args.buffer
per_eps = args.per_eps
per_alpha = args.per_alpha
per_beta = args.per_beta
per_expert_eps = args.per_expert_eps
batch_size = args.batch_size
buffer_size = args.buffer_size

# margin
margin = args.margin
margin_l = args.margin_l
margin_weight = args.margin_weight
margin_beta = args.margin_beta

# logging
log_pre = args.log_pre
log_sub = args.log_sub
no_bar = args.no_bar
time_limit = args.time_limit
load_sub = args.load_sub
if load_sub == 'None':
    load_sub = None

# z
num_zs = args.num_zs
min_z = args.min_z
max_z = args.max_z

######################################################################################
env_config = {'workspace': workspace, 'max_steps': max_episode_steps, 'obs_size': heightmap_size,
              'fast_mode': fast_mode, 'action_sequence': action_sequence, 'render': render,
              'num_objects': num_objects, 'random_orientation':random_orientation, 'reward_type': reward_type,
              'simulate_grasp': simulate_grasp, 'pick_rot': pick_rot, 'place_rot': place_rot,
              'perfect_grasp': perfect_grasp, 'perfect_place': perfect_place, 'scale': scale, 'robot': robot,
              'workspace_check': 'point', 'in_hand_mode': in_hand_mode, 'num_random_objects': num_random_objects}
planner_config = {'pos_noise': planner_pos_noise, 'rot_noise': planner_rot_noise}
if seed is not None:
    env_config['seed'] = seed
######################################################################################
hyper_parameters = {}
for key in sorted(vars(args)):
    hyper_parameters[key] = vars(args)[key]

for key in hyper_parameters:
    print('{}: {}'.format(key, hyper_parameters[key]))