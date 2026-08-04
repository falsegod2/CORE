# Apply 5E patch on top of the 5D repository

Back up the 5D branch first. Then copy the patch contents into the repository
root, preserving paths.

```bash
unzip EXPERIMENT-5E-PATCH.zip -d /tmp/EXPERIMENT-5E-PATCH
cp -a /tmp/EXPERIMENT-5E-PATCH/. /gz-data/CORE/
cd /gz-data/CORE
python scripts/audit_outcomeaware5e.py
pytest -q tests/test_outcome_aware_stable5e.py
bash ./scripts/train_5e.sh harvest_log_in_plains 0 0
```

Do not resume a 5D checkpoint. Start 5E from a fresh log directory because the
training objective and optimizer trajectory differ.
