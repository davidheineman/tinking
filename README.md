### setup

```bash
uv sync

export TINKER_API_KEY=... # ensure it's set!
```

### rl on code

```bash
python tinking/train_coding_rl.py \
    --model-name "meta-llama/Llama-3.2-1B" \
    --log-path ./logs \
    --num-batches 100 \
    --groups-per-batch 4 \
    --group-size 4 \
    --learning-rate 1e-4
```