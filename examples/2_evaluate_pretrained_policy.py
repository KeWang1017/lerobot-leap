"""
This scripts demonstrates how to evaluate a pretrained policy from the HuggingFace Hub or from your local
training outputs directory. In the latter case, you might want to run examples/3_train_policy.py first.
"""

from pathlib import Path

import gym_pusht  # noqa: F401
import gymnasium as gym
import imageio
import numpy as np
import torch
from huggingface_hub import snapshot_download

from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.common.policies.act.modeling_act import ACTPolicy
# import custom environment
import gym_lowcostrobot
import time

# Create a directory to store the video of the evaluation
output_directory = Path("outputs/eval/lowcostrobot_liftcube_act")
output_directory.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda")

# Download the diffusion policy for pusht environment
# pretrained_policy_path = Path(snapshot_download("lerobot/diffusion_pusht"))
# OR uncomment the following to evaluate a policy from the local outputs/train folder.
# pretrained_policy_path = Path("outputs/train/2024-06-21/17-12-04_gym-lowcostrobot_act_default/checkpoints/last/pretrained_model")
# policy = ACTPolicy.from_pretrained(pretrained_policy_path)

pretrained_policy_path = Path("outputs/train/2024-06-25/17-03-40_gym-lowcostrobot_diffusion_default/checkpoints/last/pretrained_model")
policy = DiffusionPolicy.from_pretrained(pretrained_policy_path)
policy.eval()
policy.to(device)

# Initialize evaluation environment to render two observation types:
# an image of the scene and state/position of the agent. The environment
# also automatically stops running after 300 interactions/steps.
env = gym.make(
    "LiftCube-v0",
    disable_env_checker=True, 
    observation_mode="both",
    action_mode="ee",
    render_mode="rgb_array",
)

# Reset the policy and environmens to prepare for rollout
policy.reset()
numpy_observation, info = env.reset(seed=42)

# Prepare to collect every rewards and all the frames of the episode,
# from initial state to final state.
rewards = []
frames = []

# Render frame of the initial state
frames.append(env.render())

step = 0
done = False
time_list = []
while not done:
    start_time = time.time()
    # Prepare observation for the policy running in Pytorch
    state = torch.from_numpy(np.concatenate(
                    [np.array(numpy_observation["arm_qpos"]), np.array(numpy_observation["arm_qvel"]), np.array(numpy_observation["object_qpos"])],
                ))
    image_front = torch.from_numpy(numpy_observation["image_front"])
    image_top = torch.from_numpy(numpy_observation["image_top"])

    # Convert to float32 with image from channel first in [0,255]
    # to channel last in [0,1]
    state = state.to(torch.float32)
    image_front = image_front.to(torch.float32) / 255
    image_front = image_front.permute(2, 0, 1)
    image_top = image_top.to(torch.float32) / 255
    image_top = image_top.permute(2, 0, 1)

    # Send data tensors from CPU to GPU
    state = state.to(device, non_blocking=True)
    image_front = image_front.to(device, non_blocking=True)
    image_top = image_top.to(device, non_blocking=True)

    # Add extra (empty) batch dimension, required to forward the policy
    state = state.unsqueeze(0)
    image_front = image_front.unsqueeze(0)
    image_top = image_top.unsqueeze(0)

    # Create the policy input dictionary
    observation = {
        "observation.state": state,
        "observation.images.front": image_front,
        "observation.images.top": image_top,
    }

    # Predict the next action with respect to the current observation
    with torch.inference_mode():
        action = policy.select_action(observation)

    # Prepare the action for the environment
    numpy_action = action.squeeze(0).to("cpu").numpy()
    time_duration = time.time() - start_time
    time_list.append(time_duration)

    # Step through the environment and receive a new observation
    numpy_observation, reward, terminated, truncated, info = env.step(numpy_action)
    print(f"{step=} {reward=} {terminated=}")

    # Keep track of all the rewards and frames
    rewards.append(reward)
    frames.append(env.render())

    # The rollout is considered done when the success state is reach (i.e. terminated is True),
    # or the maximum number of iterations is reached (i.e. truncated is True)
    done = terminated | truncated | done
    step += 1

# if terminated:
if info["is_success"]:
    print("Success!")
else:
    print("Failure!")

print(f"Average time per action: {np.mean(time_list):.4f} s")
# Get the speed of environment (i.e. its number of frames per second).
fps = env.metadata["render_fps"]
# fps = 50
env.close() # close the rendering window

# Encode all frames into a mp4 video.
video_path = output_directory / "rollout.mp4"
imageio.mimsave(str(video_path), np.stack(frames), fps=fps)

print(f"Video of the evaluation is available in '{video_path}'.")
