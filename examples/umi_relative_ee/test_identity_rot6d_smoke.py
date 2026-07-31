"""Smoke test for the umi_rot6d_identity_norm A/B flag.

Verifies:
  1. The config flag exists and parses on act/smolvla/pi05.
  2. compute_umi_relative_ee_stats(identity_rot6d=True) leaves rot6d dims at
     identity stats while keeping pos/gripper at their real computed stats.
  3. A NormalizerProcessorStep built from those identity stats is an exact no-op
     on the rot6d dims (for MIN_MAX and QUANTILES modes).
"""
import numpy as np
import torch

from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature
from lerobot.processor.normalize_processor import NormalizerProcessorStep
from lerobot.datasets.umi_relative_ee_stats import compute_umi_relative_ee_stats


def make_fake_dataset(n_episodes=3, episode_len=20, seed=0):
    rng = np.random.default_rng(seed)
    actions, ep_idx = [], []
    for e in range(n_episodes):
        # small per-frame deltas so relative poses are near-identity, like real data
        base = rng.uniform(-0.05, 0.05, size=7)
        for t in range(episode_len):
            actions.append(base + 0.002 * t * rng.uniform(-1, 1, size=7))
            ep_idx.append(e)
    return {"action": np.asarray(actions, dtype=np.float32),
            "episode_index": np.asarray(ep_idx, dtype=np.int64)}


def main():
    # 1. Config flag parses.
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.policies.pi05.configuration_pi05 import PI05Config
    for cfg_cls, name in [(ACTConfig, "act"), (SmolVLAConfig, "smolvla"), (PI05Config, "pi05")]:
        c = cfg_cls(use_umi_relative_ee=True, umi_rot6d_identity_norm=True)
        assert getattr(c, "umi_rot6d_identity_norm") is True, name
        print(f"[ok] {name}: umi_rot6d_identity_norm flag present and settable")

    hf = make_fake_dataset()
    chunk_size = 10

    # 2a. Baseline (scaled) stats.
    scaled = compute_umi_relative_ee_stats(hf, chunk_size, identity_rot6d=False)
    # 2b. Identity-rot6d stats.
    ident = compute_umi_relative_ee_stats(hf, chunk_size, identity_rot6d=True)

    act_rot6d = slice(3, 9)
    # rot6d dims must be identity in `ident` but NOT in `scaled`.
    for stat, want in {"min": -1.0, "max": 1.0, "mean": 0.0, "std": 1.0,
                       "q01": -1.0, "q99": 1.0}.items():
        got = ident["action"][stat][act_rot6d]
        assert np.allclose(got, want), f"action {stat} rot6d = {got}, want {want}"
        # scaled should differ from identity on at least the min/max-ish stats
        if stat in ("min", "max", "q01", "q99"):
            sc = scaled["action"][stat][act_rot6d]
            assert not np.allclose(sc, want), f"scaled action {stat} unexpectedly identity"
    # pos dims [0:3] and gripper [9] must still be real (not identity) in `ident`.
    assert not np.allclose(ident["action"]["min"][:3], -1.0), "pos min should be real"
    assert not np.allclose(ident["action"]["max"][9], 1.0), "gripper max should be real"
    # state rot6d slices [3:9] and [13:19].
    for sl in (slice(3, 9), slice(13, 19)):
        assert np.allclose(ident["observation.state"]["min"][sl], -1.0)
        assert np.allclose(ident["observation.state"]["max"][sl], 1.0)
    print("[ok] stats: rot6d dims forced to identity; pos/gripper retain real stats")

    # 3. Normalizer no-op on rot6d for both MIN_MAX and QUANTILES.
    features = {"action": PolicyFeature(type=FeatureType.ACTION, shape=(10,))}
    x = torch.tensor([[0.3, -0.7, 0.05, 0.9, -0.1, 0.2, -0.4, 0.8, 0.0, 0.5]])
    for mode in (NormalizationMode.MIN_MAX, NormalizationMode.QUANTILES):
        norm = NormalizerProcessorStep(features=features, norm_map={FeatureType.ACTION: mode},
                                       stats={"action": ident["action"]})
        out = norm._apply_transform(x, "action", FeatureType.ACTION, inverse=False)
        rot6d_in = x[0, 3:9]
        rot6d_out = out[0, 3:9]
        assert torch.allclose(rot6d_in, rot6d_out, atol=1e-5), \
            f"{mode}: rot6d changed {rot6d_in} -> {rot6d_out}"
        # and unnormalizing the (unchanged) rot6d is also identity
        back = norm._apply_transform(out, "action", FeatureType.ACTION, inverse=True)
        assert torch.allclose(x[0, 3:9], back[0, 3:9], atol=1e-5), f"{mode}: unnormalize not identity"
        print(f"[ok] normalizer {mode.value}: rot6d dims are an exact no-op (fwd + inv)")

    print("\nALL SMOKE CHECKS PASSED")


if __name__ == "__main__":
    main()
