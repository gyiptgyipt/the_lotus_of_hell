"""
Export a trained SB3 PPO policy to ONNX so uav_control/rl_inference_node.cpp
can load it with ONNX Runtime (no PyTorch dependency at inference time).

We export the deterministic action (the Gaussian distribution's mean, no
sampling) since that's what you want at inference. SB3's PPO does not
tanh-squash continuous actions itself -- it clips them to the action space
bounds after sampling -- so the exported network outputs an unbounded mean
and rl_inference_node.cpp clamps to [-1, 1] on the C++ side, matching what
SB3 does during rollout collection.

Usage:
  python3 export_onnx.py --model runs/exp1/ppo_intercept_final.zip \
      --out ../uav_intercept_ws/src/uav_control/models/policy.onnx
"""

import argparse

import torch
import torch.nn as nn
from stable_baselines3 import PPO

from ros_bridge import OBS_DIM


class DeterministicPolicyWrapper(nn.Module):
    """Runs SB3's feature extractor -> actor MLP -> action_net (mean), matching
    ActorCriticPolicy._predict(..., deterministic=True) for a Gaussian policy."""

    def __init__(self, sb3_policy):
        super().__init__()
        self.features_extractor = sb3_policy.features_extractor
        self.mlp_extractor = sb3_policy.mlp_extractor
        self.action_net = sb3_policy.action_net

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        features = self.features_extractor(obs)
        latent_pi, _ = self.mlp_extractor(features)
        mean_actions = self.action_net(latent_pi)
        return mean_actions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to SB3 .zip checkpoint")
    parser.add_argument("--out", type=str, required=True, help="Output .onnx path")
    args = parser.parse_args()

    model = PPO.load(args.model, device="cpu")
    wrapper = DeterministicPolicyWrapper(model.policy).eval()

    dummy_input = torch.zeros(1, OBS_DIM, dtype=torch.float32)

    torch.onnx.export(
        wrapper,
        dummy_input,
        args.out,
        input_names=["observation"],
        output_names=["action"],
        dynamic_axes={"observation": {0: "batch"}, "action": {0: "batch"}},
        opset_version=17,
    )
    print(f"Exported ONNX policy to {args.out}")

    # Sanity check: run the same input through torch and onnxruntime, compare.
    import onnxruntime as ort
    import numpy as np

    with torch.no_grad():
        torch_out = wrapper(dummy_input).numpy()

    sess = ort.InferenceSession(args.out, providers=["CPUExecutionProvider"])
    onnx_out = sess.run(None, {"observation": dummy_input.numpy()})[0]

    max_diff = np.max(np.abs(torch_out - onnx_out))
    print(f"Max abs diff (torch vs onnxruntime) on zero input: {max_diff:.6f}")
    if max_diff > 1e-4:
        print("WARNING: outputs diverge more than expected -- check opset/version compatibility.")


if __name__ == "__main__":
    main()
