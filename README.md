### setup

```bash
uv sync

export TINKER_API_KEY=... # ensure it's set!
```

### rl on tbench

```bash
tb run \
  --agent terminus-tinker \
  --agent-kwarg checkpoint_path=tinker://a4782131-a6c1-41bb-800d-af2f9b5a3db1/sampler_weights/000061 \
  --agent-kwarg model_name=Qwen/Qwen3-8B \
  --dataset terminal-bench-core==0.1.1 \
  --task-id hello-world \
  --n-concurrent 1 \
  --output-path ~/tmp/tbench
```

Design ideas:

- Each turn samples using the above command, then pulls the output from the command
   - Then, it pulls the new model during training
- Each turn also uploads traces to transluce (labeled with train step)

- Only uses the existing images, which can be used without any need to use `papergym` logic