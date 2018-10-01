#!/usr/bin/env python

"""
Code to load an expert policy and generate roll-out data for behavioral cloning.
Example usage:
    python run_expert.py experts/Humanoid-v1.pkl Humanoid-v1 --render \
            --num_rollouts 20

Author of this script and included expert policies: Jonathan Ho (hoj@openai.com)
"""

import os
import pickle
import tensorflow as tf
import numpy as np
import tf_util
import gym
import load_policy
import argparse
from keras.models import Sequential
from keras.layers import Dense, Activation
import pdb

def main(iters=10):
    # run trained model
    parser = argparse.ArgumentParser()
    parser.add_argument('expert_policy_file', type=str)
    parser.add_argument('envname', type=str)
    parser.add_argument('--render', action='store_true')
    parser.add_argument("--max_timesteps", type=int)
    parser.add_argument('--num_rollouts', type=int, default=20,
                        help='Number of expert roll outs')
    args = parser.parse_args()

    print('loading and building expert policy')
    policy_fn = load_policy.load_policy(args.expert_policy_file)
    print('loaded and built')


    # load the expert data
    with open('expert_data/'+args.envname+'.pkl', 'rb') as f:
        data = pickle.load(f)
    
    observations_exp = data['observations']
    actions_exp_orig = data['actions']
    actions_exp = np.reshape(actions_exp_orig, (len(actions_exp_orig), actions_exp_orig.shape[-1]))
    
    n_obs = observations_exp.shape[-1]
    n_act = actions_exp.shape[-1]
    
    # build a keras model
    model = Sequential([
        Dense(64, input_shape=(n_obs,)), Activation('relu'),
        Dense(64), Activation('relu'),
        Dense(64), Activation('relu'),
        Dense(32), Activation('relu'),
        Dense(n_act)])
    
    # For a mean squared error regression problem
    model.compile(optimizer='rmsprop',
                  loss='mse')

    import gym
    means = []
    stds = []
    with tf.Session():
        tf_util.initialize()
        env = gym.make(args.envname)
        max_steps = args.max_timesteps or env.spec.timestep_limit
        
        for it in range(iters):
            # Fit the data
            model.fit(observations_exp, actions_exp, epochs=16, batch_size=32, validation_split=0.25)

            returns = []
            new_observations = []
            new_actions = []
            for i in range(args.num_rollouts):
                print('iter', i)
                obs = env.reset()
                done = False
                totalr = 0.
                steps = 0
                while not done:
                    action = model.predict(obs[None,:])
                    new_observations.append(obs)
                    new_actions.append([action])
                    obs, r, done, _ = env.step(action)
                    totalr += r
                    steps += 1
                    if args.render:
                        env.render()
                    if steps % 100 == 0: print("%i/%i"%(steps, max_steps))
                    if steps >= max_steps:
                        break
                returns.append(totalr)

            print('{0} interation'.format(it))
            print('returns', returns)
            print('mean return', np.mean(returns))
            print('std of return', np.std(returns))
            means.append(np.mean(returns))
            stds.append(np.std(returns))

            # add new observations and corresponding expert actions to the original data
            exp_actions = []
            for obs in new_observations:
                #observations_exp.append(obs)
                exp_actions.append(policy_fn(obs[None,:]))

            first_dim = np.array(exp_actions).shape[0]
            last_dim  = np.array(exp_actions).shape[2]
            observations_exp = np.concatenate((observations_exp, np.array(new_observations)), 0)
            actions_exp = np.concatenate((actions_exp, np.reshape(np.array(exp_actions), [first_dim, last_dim])), 0)
            print(means)
            print(stds)

if __name__ == '__main__':
    main()

