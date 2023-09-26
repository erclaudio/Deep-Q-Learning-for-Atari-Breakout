from google.colab import drive

drive.mount('/content/drive')



from baselines.common.atari_wrappers import make_atari, wrap_deepmind
import numpy as np
import tensorflow as tf
from tensorflow import keras
import gym

seed = 42

model = keras.models.load_model("/content/drive/MyDrive/game_ai/model/")

env = make_atari("BreakoutNoFrameskip-v4")
env = wrap_deepmind(env, frame_stack=True, scale=True)
env.seed(seed)

    # Adding the monitor as a wrapper to the environment
env = gym.wrappers.Monitor(env, '/content/drive/MyDrive/game_ai/videos' , video_callable=lambda episode_id: True, force=True)

# setting the return parameters
n_episodes = 10
rewards = np.zeros(n_episodes, dtype=float)

for i in range(n_episodes):
    # Resetting the state for each episode
    state = np.array(env.reset())
    done = False

    while not done:
        # Choosing an action based on greedy policy
        state_tensor = tf.convert_to_tensor(state)
        state_tensor = tf.expand_dims(state_tensor, 0)
        action_values = model.predict(state_tensor)
        action = np.argmax(action_values)

        # Perform action and get next state, reward and done
        state_next, reward, done, _ = env.step(action)
        state = np.array(state_next)

        # Update the reward observed at episode i
        rewards[i] += reward

env.close()
print('Returns: {}'.format(returns))