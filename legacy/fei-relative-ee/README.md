# Historical `fei` relative-EE source snapshot

This directory is a verbatim source snapshot from `fei@cbf41c1e`. It preserves
the former `RelativeEEDataset`, temporal ACT wrapper, Diffusion UMI prototype,
policy-specific processor factories, SO101 integration, validation code, and
associated tests.

It is intentionally outside `src/` and is not imported or registered by the v5
runtime. The maintained implementation uses the standard LeRobotDataset plus
`use_umi_relative_ee=true` for ACT, SmolVLA, and π0.5. See
`examples/umi_relative_ee/README.md`.
