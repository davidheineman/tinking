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
# RL on arithmetic + wandb
python tinking/trainer.py \
  model_name=meta-llama/Llama-3.2-1B \
  batch_size=100 \
  group_size=4 \
  optim.lr=1e-4 \
  env=MathConfig \
  env.dataset=arithmetic \
  env.max_tokens=5 \
  wandb.entity=ai2-llm \
  wandb.project=tinker \
  wandb.run_name=debug-arithmetic

# RL on MATH 500 + wandb
python tinking/trainer.py \
  model_name=Qwen/Qwen3-4B-Instruct-2507 \
  num_batches=5 \
  group_size=4 \
  wandb.enabled=True \
  wandb.entity=ai2-llm \
  wandb.project=tinker \
  wandb.run_name=debug-math-500 \
  env=MathConfig \
  env.dataset=math500 \
  env.max_tokens=2048

# Tinker Demo's MATH example + wandb
python tinking/trainer.py \
  model_name=Qwen/Qwen3-8B \
  group_size=16 \
  batch_size=64 \
  optim.lr=2e-5 \
  env=MathConfig \
  env.dataset=hendrycks \
  env.max_tokens=512 \
  wandb.entity=ai2-llm \
  wandb.project=tinker \
  wandb.run_name=debug-math-hendrycks

# RL on terminal hello world
python tinking/trainer.py \
  model_name=Qwen/Qwen3-8B \
  num_batches=5 \
  group_size=4 \
  wandb.enabled=False \
  minitb.dataset=terminal-bench-core==0.1.1 \
  minitb.task_id=hello-world

# RL on hello world + wandb
python tinking/trainer.py \
  model_name=Qwen/Qwen3-8B \
  num_batches=100 \
  group_size=16 \
  wandb.run_name=debug-hello-world \
  minitb.dataset=terminal-bench-core==0.1.1 \
  minitb.task_id=hello-world

# small scale run
python tinking/trainer.py \
  model_name=Qwen/Qwen3-8B \
  num_batches=100 \
  group_size=16 \
  wandb.enabled=False \
  minitb.dataset_path=~/ai2/papergym/papers
```

### build Dockerfile

```sh
# manually build and deploy
docker build --platform linux/amd64 -t tinking .
beaker image delete davidh/tinking || true
beaker image create --name tinking tinking
```

</details>

### rl on papergym

```sh
# big run
python tinking/trainer.py \
  model_name=Qwen/Qwen3-235B-A22B-Instruct-2507 \
  num_batches=100 \
  group_size=16 \
  wandb.run_name=papergym \
  minitb.dataset_path=~/ai2/papergym/papers

# openai/gpt-oss-120b
```

### rl on ttt-discovery

```sh
python tinking/trainer.py \
  model_name=openai/gpt-oss-20b \
  num_batches=50 \
  batch_size=512 \
  group_size=64 \
  lora_rank=32 \
  adv_estimator=entropic \
  adv_beta=2.0 \
  env=ErdosConfig \
  env.n_points=200 \
  env.buffer_size=16 \
  env.epsilon=0.125 \
  env.max_tokens=2048 \
  env.teacher_forcing=True \
  env.context_window_tokens=4096 \
  wandb.enabled=False

python tinking/beaker/launch.py \
  workspace=ai2/davidh \
  budget=ai2/oe-base \
  follow=true \
  allow_dirty=true \
  -- \
python tinking/trainer.py \
  model_name=openai/gpt-oss-20b \
  num_batches=50 \
  batch_size=512 \
  group_size=64 \
  lora_rank=32 \
  adv_estimator=entropic \
  adv_beta=2.0 \
  log_path=/results \
  env=ErdosConfig \
  env.n_points=200 \
  env.buffer_size=16 \
  env.epsilon=0.125 \
  env.max_tokens=26000 \
  env.teacher_forcing=True \
  env.context_window_tokens=32768 \
  wandb.enabled=True \
  wandb.entity=ai2-llm \
  wandb.project=tinker \
  wandb.run_name=erdos-ttt
```

### rl on beaker

```sh
# RL on hello world
python tinking/beaker/launch.py \
  workspace=ai2/davidh \
  budget=ai2/oe-base \
  follow=true \
  allow_dirty=true \
  -- \
python tinking/trainer.py \
  model_name=Qwen/Qwen3-8B \
  num_batches=100 \
  group_size=16 \
  log_path=/results \
  wandb.run_name=debug-hello-world \
  minitb.dataset="terminal-bench-core==0.1.1" \
  minitb.task_id=hello-world

# RL on Olmo 3 Math mix
python tinking/beaker/launch.py \
  workspace=ai2/davidh \
  budget=ai2/oe-base \
  follow=true \
  allow_dirty=true \
  -- \
python tinking/trainer.py \
  model_name=meta-llama/Llama-3.1-8B-Instruct \
  num_batches=100 \
  group_size=16 \
  wandb.enabled=True \
  wandb.entity=ai2-llm \
  wandb.project=tinker \
  wandb.run_name=debug-math-500 \
  env=MathConfig \
  env.dataset=polaris \
  env.max_tokens=16384

# big run
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
  log_path=/results \
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