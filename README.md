### setup

```sh
# Grab a key: https://tinker-console.thinkingmachines.ai/keys
export TINKER_API_KEY=...
```

```sh
uv sync
```

<details>
<summary>debugging</summary>

```sh
# Make sure `beaker` CLI is installed: https://beaker-docs.apps.allenai.org/start/install.html
```

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
# RL on hello world
python tinking/trainer.py \
  model_name="Qwen/Qwen3-8B" \
  num_batches=5 \
  group_size=4 \
  wandb.enabled=False \
  minitb.dataset="terminal-bench-core==0.1.1" \
  minitb.task_id=hello-world

# RL on hello world + wandb
python tinking/trainer.py \
  model_name="Qwen/Qwen3-235B-A22B-Instruct-2507" \
  num_batches=20 \
  group_size=16 \
  wandb.run_name="debug-hello-world" \
  minitb.dataset="terminal-bench-core==0.1.1" \
  minitb.task_id=hello-world

# small scale run
python tinking/trainer.py \
  model_name="Qwen/Qwen3-8B" \
  num_batches=5 \
  group_size=4 \
  wandb.enabled=False \
  minitb.dataset_path=~/ai2/papergym/papers
```

</details>

### rl on papergym

```sh
# big run
python tinking/trainer.py \
  model_name="Qwen/Qwen3-235B-A22B-Instruct-2507" \
  num_batches=100 \
  group_size=16 \
  wandb.run_name="papergym" \
  minitb.dataset_path=~/ai2/papergym/papers

# openai/gpt-oss-120b
```

### rl on beaker

```sh
python tinking/beaker/launch.py \
  workspace=ai2/davidh \
  budget=ai2/oe-base \
  follow=true \
  allow_dirty=true \
  -- \
python tinking/trainer.py \
  model_name=Qwen/Qwen3-235B-A22B-Instruct-2507 \
  num_batches=100 \
  group_size=16 \
  wandb.run_name=papergym \
  minitb.dataset_path=~/ai2/papergym/papers
```

### plans

- [ ] Allow executing multiple task IDs at once
- [ ] Extract correct/incorrect from terminalbench (not helpful for papergym, but helpful for terminalbench)

- [X] Each turn samples using the above command, then pulls the output from the command
   - [X] Then, it pulls the new model during training
- [ ] Each turn also uploads traces to transluce (labeled with train step)
- [ ] Only uses the existing images, which can be used without any need to use `papergym` logic