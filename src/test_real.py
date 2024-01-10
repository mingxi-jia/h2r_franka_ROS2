import os
import time
from termcolor import colored
import numpy as np
from networks.pick_agent import Actor
import argparse
import torch
import rospy
import matplotlib.pyplot as plt
from instructions import generate_pick_instruction_shape_part

import pickle

from src.envs.env import Env
import src.simulator.utils as utils
import utils.demo_util as demo_util
# import src.utils.demo_util

# import time
# from termcolor import colored
# from networks.pick_agent_lan_img import Actor as ActorLO
# import clip

parser = argparse.ArgumentParser(description='lan_pick')
parser.add_argument('--save_dir', type=str, default='.')
parser.add_argument('--data_dir', type=str, default='./demos')
parser.add_argument('--assets_root', type=str, default='./simulator/data_robot')
# parser.add_argument('--task', type=str, default='pick-ell')
parser.add_argument('--task', type=str, default='pick-parts-real')
parser.add_argument('--n_demos', type=int, default=6)
parser.add_argument('--n_steps', type=int, default=40024)  # when to stop
parser.add_argument('--n_runs', type=int, default=1)
parser.add_argument('--device', type=int, default=1)
# change to equi_obs_net to str to handle different types of image encoder
parser.add_argument('--equi_obs_net', action='store_true', default=False)
parser.add_argument('--disp', action='store_true', default=False)
# parser.add_argument('--lan_emd_key', type=str, default='lan_emd_clip_vit32')
parser.add_argument('--lan_ins_key', type=str, default='lan')
parser.add_argument('--no_save', action='store_true', default=False)
parser.add_argument('--mul', action='store_true', default=False)
parser.add_argument('--lan_kernel', action='store_true', default=False)
parser.add_argument('--actor', type=str, choices=['unetl_real'], default='unetl_real')
args = parser.parse_args()


if __name__ == "__main__":
    rospy.init_node('image_proxy')

    global model, alg, action_sequence, in_hand_mode, workspace, max_z, min_z
    ws_center = [-0.5539, 0.0298, -0.1625]
    workspace = np.asarray([[ws_center[0] - 0.15, ws_center[0] + 0.15],
                            [ws_center[1] - 0.15, ws_center[1] + 0.15],
                            [ws_center[2], 0.50]])
    max_z = 0.12
    min_z = 0.02

    # Initial Agent
    name = f'{args.task}-{args.n_demos}-0-{args.actor}'

    model_task = args.task
    model_name = f'{args.task}-{args.n_demos}-0-{args.actor}'
    print(f"N steps is {args.n_steps}")
    if args.actor == 'unetl_real':
        print('test with phi==unet')
        agent = Actor(device=args.device, name=model_name, task=model_task, save_dir=args.save_dir,
                      equi_obs_net=args.equi_obs_net, init=False, model_name='unetl_real',
                      in_shape=(208, 288, 3))
    else:
        NotImplementedError

    action_sequence = 'pxyzr'
    obs_source = 'raw'
    env = Env(ws_center=ws_center, action_sequence=action_sequence, obs_source=obs_source)
    rospy.sleep(2)
    env.ur5.moveToHome()
    agent.load(args.n_steps)
    while True:
        input("Press Enter to take photo...")
        rgbd = env.getObs(None)
        plt.imshow(rgbd[:, :, :3].numpy().astype(int))
        plt.show(block=False)
        plt.pause(1)

        obj  =  input('type the target object: ')
        part = input('type the part: ')

        info = {'shape':obj,'part':part}
        lan = generate_pick_instruction_shape_part(info,method='random')[0]
        print(colored(lan, 'green'))
        x, y, z, rz = agent.act_real(rgbd, lan=lan)

        command = input("Press Enter to execute the action..or q to go home")
        # rz = rz + np.pi/2
        # adjust the z value to make the grasp safely
        if command=='q':
            env.ur5.moveToHome()
        else:
            z_command = input('y to use default and N to use the lowerest')
            if z_command == 'y':
                z = -0.1551397
                if obj=='cup' or obj =='mug':
                    z = -0.151 + 0.055
                elif obj == 'bowl':
                    z = -0.151 + 0.025
                elif part =='heel':
                    z = -0.151 + 0.023
            else:
                z = -0.1551397

            env.ur5.pick_and_throw(x, y, z, [0,0,rz])
            env.ur5.moveToHome()
