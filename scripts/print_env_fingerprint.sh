#!/usr/bin/env bash
set -euo pipefail
python - <<'PY2'
import platform
import numpy as np
import torch
try:
    import minedojo
    minedojo_version = getattr(minedojo, "__version__", "unknown")
except Exception as exc:
    minedojo_version = f"unavailable: {exc}"
print("OS:", platform.platform())
print("Python:", platform.python_version())
print("NumPy:", np.__version__)
print("Torch:", torch.__version__)
print("CUDA runtime:", torch.version.cuda)
print("cuDNN:", torch.backends.cudnn.version())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
print("MineDojo:", minedojo_version)
PY2
java -version || true
nvidia-smi || true
