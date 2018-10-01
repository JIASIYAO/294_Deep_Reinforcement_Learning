import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Activation
import pickle
import gym
import argparse
import tf_util
import os
import pdb

# run trained model
parser = argparse.ArgumentParser()
parser.add_argument('envname', type=str)
parser.add_argument('--render', action='store_true')
parser.add_argument("--max_timesteps", type=int)
parser.add_argument('--num_rollouts', type=int, default=20,
                    help='Number of expert roll outs')
args = parser.parse_args()


# load the expert data
with open('expert_data/'+args.envname+'.pkl', 'rb') as f:
    data = pickle.load(f)

observations_exp = data['observations']
actions_exp_orig = data['actions']
actions_exp = np.reshape(actions_exp_orig, (len(actions_exp_orig), actions_exp_orig.shape[-1]))

n_obs = observations_exp.shape[-1]
n_act = actions_exp.shape[-1]
#pdb.set_Trace()

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


with tf.Session():
    tf_util.initialize()

    # Fit the data
    model.fit(observations_exp, actions_exp, epochs=16, batch_size=32, validation_split=0.25)

    env = gym.make(args.envname)
    max_steps = args.max_timesteps or env.spec.timestep_limit

    returns = []
    observations = []
    actions = []

    for i in range(args.num_rollouts):
        print('iter', i)
        obs = env.reset()
        done = False
        totalr = 0.
        steps = 0
        while not done:
            action = model.predict(obs[None,:])
            observations.append(obs)
            actions.append([action])
            obs, r, done, _ = env.step(action)
            totalr += r
            steps += 1
            if args.render:
                env.render()
            if steps % 100 == 0: print("%i/%i"%(steps, max_steps))
            if steps >= max_steps:
                break
        returns.append(totalr)
    print('returns', returns)
    print('mean return', np.mean(returns))
    print('std of return', np.std(returns))

    train_data = {'observations': np.concatenate(observations),
                   'actions': np.array(actions)}

    with open(os.path.join('train_data', args.envname + '.pkl'), 'wb') as f:
        pickle.dump(train_data, f, pickle.HIGHEST_PROTOCOL)


