### setup

```sh
# Make sure `beaker` CLI is installed: https://beaker-docs.apps.allenai.org/start/install.html

uv sync

export TINKER_API_KEY=... # ensure it's set!
```

<details>
<summary>debugging</summary>

### test tinker decoder in `tb` + local container

```sh
tb run \
  --agent terminus-tinker \
  --agent-kwarg checkpoint_path=tinker://a4782131-a6c1-41bb-800d-af2f9b5a3db1/sampler_weights/000061 \
  --agent-kwarg model_name=Qwen/Qwen3-8B \
  --dataset terminal-bench-core==0.1.1 \
  --task-id hello-world \
  --n-concurrent 1 \
  --output-path ~/tmp/tbench
```

### test `minitb` + beaker container

```sh
minitb run \
  --agent terminus-tinker \
  --agent-kwarg checkpoint_path=tinker://a4782131-a6c1-41bb-800d-af2f9b5a3db1/sampler_weights/000061 \
  --agent-kwarg model_name=Qwen/Qwen3-8B \
  --dataset-path /Users/dhei/ai2/papergym/papers \
  --task-id n19-1119 \
  --n-concurrent 1 \
  --output-path ~/tmp/tbench \
  --log-level debug
```

### test trainer

```sh
# rl on hello world (TODO: make a command that works)
python tinking/trainer.py \
  model_name="Qwen/Qwen3-8B" \
  log_path=./logs \
  n_concurrent=1 \
  num_batches=5

# debugging scale
python tinking/trainer.py \
  model_name="Qwen/Qwen3-8B" \
  dataset_path=~/ai2/papergym/papers \
  log_path=./logs \
  n_concurrent=1 \
  num_batches=5
```

</details>

### rl on papergym

```sh
# Qwen/Qwen3-235B-A22B-Instruct-2507
# openai/gpt-oss-120b

# big run
python tinking/trainer.py \
  model_name="Qwen/Qwen3-235B-A22B-Instruct-2507" \
  dataset_path=~/ai2/papergym/papers \
  log_path=./logs \
  n_concurrent=1 \
  num_batches=100
```

Design ideas:

- Each turn samples using the above command, then pulls the output from the command
   - Then, it pulls the new model during training
- Each turn also uploads traces to transluce (labeled with train step)

- Only uses the existing images, which can be used without any need to use `papergym` logic