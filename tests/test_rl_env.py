import pytest
import numpy as np
from src.rl_agent.env import OptimalExecutionEnv

def test_rl_env_spaces():
    env = OptimalExecutionEnv(total_volume=100.0, total_steps=10)
    obs, info = env.reset()
    
    # Check observation shape and type
    assert obs.shape == (5,)
    assert obs.dtype == np.float32
    
    # Initial state properties
    assert env.remaining_volume == 100.0
    assert env.current_step == 0
    
    # Action space
    assert env.action_space.shape == (1,)
    assert env.action_space.dtype == np.float32

def test_rl_env_step():
    env = OptimalExecutionEnv(total_volume=100.0, total_steps=5)
    obs, info = env.reset()
    
    # Take a step: sell 50% of remaining (which is 100.0 * 0.5 = 50.0 BTC)
    action = np.array([0.5], dtype=np.float32)
    next_obs, reward, terminated, truncated, info_dict = env.step(action)
    
    assert env.current_step == 1
    assert env.remaining_volume < 100.0
    assert not terminated
    
    # Take steps until end
    for _ in range(4):
        next_obs, reward, terminated, truncated, info_dict = env.step(np.array([0.5], dtype=np.float32))
        
    # After 5 steps (total_steps), it should be terminated
    assert terminated or env.current_step >= 5
    # Since it was the last step, remaining volume should be forced to 0
    assert env.remaining_volume == 0.0
