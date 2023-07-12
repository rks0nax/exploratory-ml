from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from modules.gym_custom_env import CustomEnv

# Test the trained model
model = PPO.load("ppo_lunar_checkpoint")
# env_name = 'LunarLander-v2'
# model = PPO("MlpPolicy", CustomEnv(env_name), verbose=1)
env = CustomEnv('LunarLander-v2', mode='human')
obs, info = env.reset()
while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info, _ = env.step(action)
    env.render()
    if done:
        obs, _ = env.reset()

env.close()
