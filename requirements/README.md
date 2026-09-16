# Dependency policy

The Vast server has a verified `torch 2.10.0+cu128` and `torchvision 0.25.0+cu128` build with `sm_120` support. Do not let model installation replace it.

- `constraints-cu128.txt` documents the working Torch pair.
- `shared.txt` contains model-independent tools.
- `glm.txt` follows the official GLM-OCR package floors while constraining the verified Torch pair.
- Freeze the exact resolved GLM package versions after the first successful server smoke test.
- Paddle dependencies are deferred while establishing the GLM baseline.
