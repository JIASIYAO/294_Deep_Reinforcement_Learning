import os
import pickle
import load_policy
import tensorflow as tf
import tf_util
import numpy as np

# loading and building expert policy
policy_fn = load_policy.load_policy("experts/Hopper-v2.pkl")

i = 1
iters = 3
while i<iters:
    # run expert to get the first {o1,a1,..., on,an}
    os.system("python run_expert.py experts/Hopper-v2.pkl Hopper-v2 --render --num_rollouts 20")
    
    # train the model based on {o1,a1,...,om,am}
    os.system("python behave_clone.py Hopper-v2 --render --num_rollouts 20")

    # use expert to label actions for {o1,..,om} from model
    # load observations from expert dataset
    with open('expert_data/Hopper-v2.pkl', 'rb') as f:
        data = pickle.load(f)
    observations = list(data['observations'])
    actions = list(data['actions'])
    # load observations from train dataset
    with open('train_data/Hopper-v2.pkl', 'rb') as f:
        data = pickle.load(f)
    observations_train = data['observations']
    
    with tf.Session():
        tf_util.initialize()
        for obs in observations:
            observations.append(obs)
            actions.append(policy_fn(obs[None,:]))
        # write the total dataset from expert and the current dataset
        expert_data = {'observations': np.array(observations),
                       'actions': np.array(actions)}

        with open(os.path.join('expert_data', args.envname + '.pkl'), 'wb') as f:
            pickle.dump(expert_data, f, pickle.HIGHEST_PROTOCOL)

    i += 1 
