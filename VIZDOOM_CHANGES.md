# ViZDoom migration changes

- Added `envs/vizdoom.py`: Gymnasium ViZDoom -> legacy Dreamer interface.
- Added `vizdoom` config profile; MineCLIP path is fully zero-scaled.
- Updated `expr.py` and `test.py` to select MineDojo or ViZDoom by task prefix.
- Added five switchable ViZDoom scenarios.
- Added train/test/smoke-test scripts.
- Added `vizdoom>=1.3.0,<2.0` dependency.
- Original MineDojo environment path remains available.
