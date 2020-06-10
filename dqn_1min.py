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

# モデルを定義
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

memory = SequentialMemory(limit=50000, window_length=1)

# ε-Greedy法を使用
policy = LinearAnnealedPolicy(
    EpsGreedyQPolicy(),
    attr='eps',
    value_max=1.0,
    value_min=0.1,
    value_test=0.05,
    nb_steps=50000
)

# DDQNを利用
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

# ステップ数はcsvファイルの長さ×ループ数(chainerっぽい感じにした)
dqn.fit(env, nb_steps=env.episode_length*loop_episode, visualize=True, verbose=2)

# 学習器の出力先
dqn.save_weights('dqn_{}_weights.h5f'.format(ENV_NAME), overwrite=True)

dqn.test(env, nb_episodes=5, visualize=True)