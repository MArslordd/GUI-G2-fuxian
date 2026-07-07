# GUI-G2 Small Reproduction With verl

This repo keeps the official GUI-G2/open-r1 code, and adds a minimal verl path for practicing GRPO on GUI grounding.

## What Is Reproduced

GUI-G2 is mainly a reward-modeling method. The model predicts a GUI bbox, and the RL reward is:

- Gaussian point reward: predicted center is scored by a Gaussian centered on the target bbox center.
- Gaussian coverage reward: predicted and target bboxes are converted to 2D Gaussians and compared with the Bhattacharyya coefficient.
- Adaptive variance: sigma is proportional to bbox width/height, using `alpha=0.5`.

The verl implementation is in `gui_g2_verl/reward.py`.

## Dataset Choice

Use `likaixin/ScreenSpot-v2-variants` by default. It has 755 ScreenSpot-v2 rows, images, and multiple instruction styles, so it is small enough for a local reproduction while staying close to the paper's ScreenSpot-v2 evaluation setting.
The dataset only provides a `train` split, so the preparation script shuffles that split and then creates local `train.parquet` and `val.parquet` files.

Prepare a tiny parquet split:

```bash
python scripts/prepare_screenspot_v2_verl.py \
  --max-train-samples 64 \
  --max-val-samples 16 \
  --output-dir data/screenspot_v2_verl
```

For a smoke test, use `--max-train-samples 8 --max-val-samples 4`.

## Install

Use a verl-capable environment with CUDA, vLLM, PyTorch, and Qwen-VL dependencies. In a fresh machine, the practical path is usually:

```bash
git clone https://github.com/verl-project/verl.git ../verl
pip install -e ../verl
pip install -r requirements-verl.txt
```

Then run this repository from its root so `gui_g2_verl/reward.py` resolves correctly.

## Train

```bash
MODEL_PATH=Qwen/Qwen2.5-VL-3B-Instruct \
TRAIN_BATCH_SIZE=8 \
PPO_MINI_BATCH_SIZE=4 \
ROLLOUT_N=4 \
TOTAL_EPOCHS=1 \
bash scripts/run_verl_gui_g2_grpo.sh
```

For an even lighter run, set:

```bash
ROLLOUT_N=2 TRAIN_BATCH_SIZE=4 PPO_MINI_BATCH_SIZE=2 bash scripts/run_verl_gui_g2_grpo.sh
```

## Notes

- This is intentionally a small-data reproduction for learning verl. It should verify the training mechanics and reward behavior, not match the paper's reported numbers.
- The official paper used larger models, more data, `num_generations=8`, `beta=0.04`, bfloat16, flash attention, and high-resolution image settings.
- The original GUI-G2 training entry remains `src/gui_g2/run_grpo_gaussian.sh`; use it as a reference for the paper's non-verl implementation.
