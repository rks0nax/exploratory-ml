import gymnasium as gym

class CustomEnv(gym.Env):
    
    def __init__(self, env_name, mode='terminal'):
        super().__init__()
        self.env = gym.make(env_name, render_mode=mode)
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        return self.env.reset(seed=seed, options=options)

    def render(self):
        return self.env.render()

    def close(self):
        return self.env.close()
