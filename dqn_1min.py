import myenv  # これを読み込んでおく
import numpy as np
import gym
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Activation
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dropout

from keras.optimizers import Adam
Adam._name=""

from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory

ENV_NAME = 'myenv-v0'
env = gym.make(ENV_NAME)
nb_actions = env.action_space.n

loop_episode = 100

# Next, we build a very simple model.
model = Sequential()
model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_actions, activation='relu'))
model.add(Dense(nb_actions,activation='softmax'))
print(model.summary())

# Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
# even the metrics!
memory = SequentialMemory(limit=50000, window_length=1)
policy = LinearAnnealedPolicy(
    EpsGreedyQPolicy(),
    attr='eps',
    value_max=1.0,
    value_min=0.1,
    value_test=0.05,
    nb_steps=50000
)

dqn = DQNAgent(model=model,
               nb_actions=nb_actions,
               memory=memory,
               nb_steps_warmup=10,
               target_model_update=1e-2,
               policy=policy,
               test_policy=policy,
               enable_double_dqn=True,
               enable_dueling_network=True,
               dueling_type="avg",
               )
dqn.compile(Adam(lr=1e-3), metrics=['mae'])


# Okay, now it's time to learn something! We visualize the training here for show, but this
# slows down training quite a lot. You can always safely abort the training prematurely using
# Ctrl + C.

dqn.fit(env, nb_steps=env.episode_length*loop_episode, visualize=True, verbose=2)

# After training is done, we save the final weights.
dqn.save_weights('dqn_{}_weights.h5f'.format(ENV_NAME), overwrite=True)

# Finally, evaluate our algorithm for 5 episodes.
dqn.test(env, nb_episodes=5, visualize=True)




# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# flatten_1 (Flatten)          (None, 896)               0
# _________________________________________________________________
# dense_1 (Dense)              (None, 896)               803712
# _________________________________________________________________
# activation_1 (Activation)    (None, 896)               0
# _________________________________________________________________
# dense_2 (Dense)              (None, 896)               803712
# _________________________________________________________________
# activation_2 (Activation)    (None, 896)               0
# _________________________________________________________________
# dense_3 (Dense)              (None, 896)               803712
# _________________________________________________________________
# activation_3 (Activation)    (None, 896)               0
# _________________________________________________________________
# dense_4 (Dense)              (None, 3)                 2691
# =================================================================
