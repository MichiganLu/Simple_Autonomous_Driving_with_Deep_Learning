import numpy as np
import torch
import gym
import argparse
import os
import time
import utils
import TD3

parser = argparse.ArgumentParser(description='RL SAC LongiControl')

parser.add_argument("--policy", default="TD3")                  # Policy name (TD3, DDPG or OurDDPG)
parser.add_argument("--env", default="DeterministicTrack-v0", type=str)          # OpenAI gym environment name
parser.add_argument("--seed", default=2, type=int)              # Sets Gym, PyTorch and Numpy seeds
parser.add_argument("--start_timesteps", default=25e3, type=int)# Time steps initial random policy is used
parser.add_argument("--eval_freq", default=1e3, type=int)       # How often (time steps) we evaluate
parser.add_argument("--max_timesteps", default=3e5, type=int)   # Max time steps to run environment
parser.add_argument("--expl_noise", default=0.1)                # Std of Gaussian exploration noise
parser.add_argument("--batch_size", default=256, type=int)      # Batch size for both actor and critic
parser.add_argument("--discount", default=0.99)                 # Discount factor
parser.add_argument("--tau", default=0.005)                     # Target network update rate
parser.add_argument("--policy_noise", default=0.2)              # Noise added to target policy during critic update
parser.add_argument("--noise_clip", default=0.5)                # Range to clip target policy noise
parser.add_argument("--policy_freq", default=2, type=int)       # Frequency of delayed policy updates
parser.add_argument("--save_model", action="store_true")        # Save model and optimizer parameters
parser.add_argument("--load_model", default="")                 # Model load file name, "" doesn't load, "default" uses file_name
parser.add_argument('--visualize','-vis', action='store_true')
parser.add_argument('--record','-rec',action='store_true')  
parser.add_argument('--save_id',metavar='',type=int,default=0)
parser.add_argument('--load_id',metavar='',type=int,default=None)
parser.add_argument('--car_id',metavar='',type=str,default='BMW_electric_i3_2014')
parser.add_argument('--env_id',metavar='',type=str,default='DeterministicTrack-v0')
parser.add_argument('--reward_weights','-rw',metavar='',nargs='+',type=float,default=[1.0, 0.5, 1.0, 1.0])
parser.add_argument('--energy_factor',metavar='',type=float,default=1.0)
parser.add_argument('--speed_limit_positions',metavar='',nargs='+',type=float,default=[0.0, 0.25, 0.5, 0.75])
parser.add_argument('--speed_limits',metavar='',nargs='+',type=int,default=[50, 80, 40, 50])
args = parser.parse_args()
'''
def torch_to_numpy(torch_input):
    if isinstance(torch_input, tuple):
        return tuple(torch_to_numpy(e) for e in torch_input)
    return torch_input.data.numpy()
def numpy_to_torch(numpy_input):
    if isinstance(numpy_input, tuple):
        return tuple(numpy_to_torch(e) for e in numpy_input)
    return torch.from_numpy(numpy_input).float()
'''  
env = gym.make('gym_longicontrol:' + args.env_id,
               car_id=args.car_id,
               speed_limit_positions=args.speed_limit_positions,
               speed_limits=args.speed_limits,
               reward_weights=args.reward_weights,
               energy_factor=args.energy_factor)
    
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0] 
max_action = float(env.action_space.high[0])
kwargs = {
"state_dim": state_dim,
"action_dim": action_dim,
"max_action": max_action,
"discount": args.discount,
"tau": args.tau,
}

policy = TD3.TD3(**kwargs)
file_name = f"{args.policy}_{args.env}_{args.seed}"
policy.load(filename=f"./models/{file_name}")

env.seed(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)



from gym import wrappers
from time import time
save_dname = os.path.join(os.path.dirname(__file__),
                          f'out/{args.env_id}/SAC_id{args.save_id}')
save_dname = os.path.join(save_dname,
                          'videos/' + str(time()) + '/')
env = wrappers.Monitor(env, save_dname)
env.seed(2)


state, done = env.reset(), False
episode_return = 0
while True:
    action = policy.select_action(np.array(state))
    #action = torch_to_numpy(action).reshape(env.action_space.shape)
    state, reward, done, info = env.step(action)
    episode_return += reward

    #state = numpy_to_torch(next_state)
    env.render()

    if done:
        #state = numpy_to_torch(env.reset())
        state = env.reset()
        print(episode_return)
        break
env.close()


