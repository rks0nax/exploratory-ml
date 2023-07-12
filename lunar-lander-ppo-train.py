# Description: Train a PPO agent on the Lunar Lander environment
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_checker import check_env
from modules.gym_custom_env import CustomEnv
from stable_baselines3.common.evaluation import evaluate_policy


env_name = 'LunarLander-v2'
policy_type = "MlpPolicy"

env = CustomEnv(env_name)

try:
    model = PPO.load("ppo_lunar_checkpoint", env=env)
except Exception as e:
    model = PPO(policy_type, env, verbose=1)

# Initialize model
# Number of steps to train on
total_steps = 1000

obs, info = env.reset()
for step in range(total_steps):
    action, _ = model.predict(obs, deterministic=False)

    # Add any custom modifications to the action here

    obs, reward, done, info, _ = env.step(action)

    # Add any custom modifications to the observation or reward here

    model.learn(total_timesteps=1, reset_num_timesteps=False)
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
    print(f"Mean reward: {mean_reward} +/- {std_reward}")

    if done:
        obs, info = env.reset()

    # Save the model checkpoint every 1000 steps
    if step % 10 == 0:
        model.save("ppo_lunar_checkpoint")

# # Save the model
model.save("ppo_lunar_checkpoint")
