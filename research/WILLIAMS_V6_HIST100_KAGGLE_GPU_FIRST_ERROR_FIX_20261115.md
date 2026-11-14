# HIST100 GPU: causal diagnosis and safe retry

**Real Kaggle T4 GPU was allocated.** Original private GPU notebook failed
before the model inference stage: `HIST100_REAL_GPU_NO_UNIQUE_ATTACHED_PRIVATE_V5_TRAINING_OUTPUT`.

The worker assumed precisely one level between `/kaggle/input` and
`williams_v5_model/adapter`. Kaggle private notebook outputs may have
additional directories. This revision looks recursively for one matching
adapter, and **requires the exact original v5 weights SHA256 and authenticated
80-step receipt** before running any model. Nothing can be substituted.

The retry accepts only a previous terminal `ERROR` kernel, never overwrites
QUEUED/RUNNING/COMPLETE. Same 100 fixed charts, same original trained v5
adapter, no new training, no broker orders, no paid GitHub GPU runner.

Existing 10 CPU shards continue independently.
