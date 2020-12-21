# Simple-Autonomous-Driving-with-TD3-and-DDPG
## Foreword
Before getting further, it is more than necessary to mention that this wonderful illustrative environment is provided by dynamik1703, the original environment could be found at https://github.com/dynamik1703/gym_longicontrol. 

The TD3 and DDPG algorithms implemented in this repository was cited from Dr.Fujimoto, with adaptation to fit into this automonous driving training environment. You can find the original TD3 proposed by Scott Fujimoto at https://github.com/sfujim/TD3.

## Quick Intro
Autonomous driving is becoming the trend for future transportation. One of its most significant challenges is to recognize traffic signs and obey traffic rules specified by the signs. In this repository, the particular topic of optimal vehicle speed control whenever a vehicle reaches a speed limit sign is studied. This research is conducted in a longitudinal environment, which only has one dimension, the straight traffic lane, along which a vehicle will drive through. There will be multiple traffic signs in the traffic lane, which the vehicle can recognize 150 meters in advance. Three factors are taken into account during control optimization, that is, energy consumption of the vehicle, jerk (change of acceleration), and most importantly, speed of the vehicle. Methods of Q learning with the deep neural network are implemented in the research for speed limit control. More concisely, two deep learning methods are implemented, which are DDPG, TD3. Find more information about the environment in our paper.

## Install
```
cd Simple-Autonomous-Driving-with-TD3-and-DDPG
pip install -e .
```

## Initiate Environment
You can create an instance of the environment just like you do with other gym environments
```
import gym
gym.make('gym_longicontrol:DeterministicTrack-v0')
gym.make('gym_longicontrol:StochasticTrack-v0')
```
## Training
### Train the agent with TD3
```
cd Simple-Autonomous-Driving-with-TD3-and-DDPG\rl\pytorch\TD3
python main.py --save_model
```
The model will be saved in Simple-Autonomous-Driving-with-TD3-and-DDPG\rl\pytorch\TD3\models. A document named "trainprocess.txt" will be created in current directory to record the returns of the entire training process.
#### Visualize the performance
```
try_visualize.py
```
This block of code will automatically load the trained actor and critic network parameters and produce the rendered result in a MP4 video. The video can be found at Simple-Autonomous-Driving-with-TD3-and-DDPG\rl\pytorch\TD3\out. You may need to install the ffmpeg package before rendering the result.
```
conda install -c conda-forge ffmpeg
```
### Train the agent with DDPG
Just switch the directory to Simple-Autonomous-Driving-with-TD3-and-DDPG\rl\pytorch\DDPG and run the same code as you train the TD3.

Here is an example of the trained model provided by the author of the environment dynamik1703
<p align="center">
<img src="/img/trained_agent.gif" width=600 height=270>
</p>
