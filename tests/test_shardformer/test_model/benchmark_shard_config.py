# whether to profile
profile = True

# processes in parallel
workers = 8

# model config
shard_config = {
    "tp_size": 8,
    "pp_size": 1,
    "num_microbatches": 1,
    "enable_sequence_parallelism": True,
    "sequence_parallelism_mode": "2",
    "use_lazy_init": True,
    "precision": "fp32",
    "initial_scale": 1,
}
