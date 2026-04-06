# Roadmap Execution Plan With 3 RunPod Workers

This playbook turns ROADMAP.md into an execution schedule that uses three pods in parallel with minimal idle time.

## Pod inventory

Use the current exposed TCP endpoints:

- p1: root@213.173.108.102 -p 26317 (classic_peach_piranha)
- p2: root@213.173.108.102 -p 30412 (mid_emerald_goldfish)
- p3: root@157.157.221.29 -p 21099 (musical_red_coral)

If ports change, override with environment variables in scripts.

## Strategy

- Keep Tier 1 and Tier 2 code/paper edits on local workstation.
- Use pods primarily for Tier 3 training-heavy experiments.
- Partition by independent experiment family, not by random chunks, so each pod can run end-to-end and write interpretable logs.
- Start with fast smoke runs for every Tier 3 item, then full runs.

## Baseline setup on each pod

Run this once per pod.

1. Connect:
   bash scripts/connect_pods.sh p1

2. Bootstrap repo and env:
   cd /root
   if [ -d scaling-operator-learning ]; then cd scaling-operator-learning && git pull; else git clone https://github.com/lucasperrier/scaling-operator-learning.git && cd scaling-operator-learning; fi
   python -m venv .venv
   . .venv/bin/activate
   pip install -U pip
   pip install -e ".[dev]"

3. Sanity:
   python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"

## Pod assignment (optimal parallel layout)

### p1: Tier 3.1 data-axis densification

Reason: this is the highest-impact unresolved training block and easiest to parallelize.

Run all tasks with expanded N grid:

. .venv/bin/activate
python scripts/run_sweep.py \
  --task burgers_operator --config configs/burgers_operator.yaml \
  --models mlp_baseline,deeponet,fno \
  --dataset-sizes 50,100,150,200,300,500,750,1000,2000,3000,5000 \
  --resolutions 32,64,128,256 \
  --data-seeds 11,22,33 \
  --train-seeds 101,202,303 \
  --device cuda -j 1 | tee tier3_1_burgers.log

Then repeat for Darcy and Diffusion configs in sequence or in tmux windows.

### p2: Tier 3.3 widened resolution grid + Tier 3.4 frequency-filtered diffusion

Run widened resolution grid first:

. .venv/bin/activate
python scripts/run_sweep.py \
  --task diffusion --config configs/diffusion.yaml \
  --models mlp_baseline,deeponet,fno \
  --dataset-sizes 50,100,200,500,1000,2000,5000 \
  --resolutions 16,32,48,64,96,128,192,256,384,512 \
  --data-seeds 11,22,33 \
  --train-seeds 101,202,303 \
  --device cuda -j 1 | tee tier3_3_diffusion.log

Then run frequency-filter variants once code support exists for filtered inputs:

. .venv/bin/activate
python scripts/run_sweep.py \
  --task diffusion --config configs/diffusion.yaml \
  --models mlp_baseline,deeponet,fno \
  --dataset-sizes 500,1000,2000,5000 \
  --resolutions 128,256,512 \
  --data-seeds 11,22,33 \
  --train-seeds 101,202,303 \
  --device cuda -j 1 | tee tier3_4_filtered_diffusion.log

### p3: Tier 3.2 fixed-parameter MLP control + Tier 3.5 and Tier 3.6 ablations

This pod runs targeted ablations with smaller grids but multiple variants.

1) Fixed-parameter MLP (Diffusion):

. .venv/bin/activate
python scripts/run_sweep.py \
  --task diffusion --config configs/diffusion.yaml \
  --models mlp_baseline \
  --capacities tiny,small,small-med,medium,med-large,large,xlarge \
  --dataset-sizes 500,1000,2000,5000 \
  --resolutions 32,64,128,256 \
  --data-seeds 11,22,33 \
  --train-seeds 101,202,303 \
  --device cuda -j 1 | tee tier3_2_mlp_control.log

2) DeepONet interpolation ablation:

. .venv/bin/activate
python scripts/run_deeponet_ablation.py \
  --tasks burgers_operator,darcy,diffusion \
  --capacities medium,large,xlarge \
  --device cuda | tee tier3_5_deeponet_ablation.log

3) Early stopping and optimizer sensitivity:

. .venv/bin/activate
python scripts/run_optimizer_ablation.py \
  --config configs/burgers_operator.yaml \
  --models mlp_baseline,deeponet,fno \
  --capacities medium \
  --device cuda -j 1 | tee tier3_6_optim_burgers.log

python scripts/run_optimizer_ablation.py \
  --config configs/diffusion.yaml \
  --models mlp_baseline,deeponet,fno \
  --capacities medium \
  --device cuda -j 1 | tee tier3_6_optim_diffusion.log

## Tier 2 execution (existing data only)

Run locally after code changes are merged:

python scripts/run_holdout_contest.py
python scripts/run_within_slice_bootstrap.py
python scripts/aggregate_runs.py --runs-root runs
python scripts/run_multilaw_analysis.py
python scripts/plot_multilaw_figures.py

Then refresh paper tables/figures.

## Simplified execution (per pod)

Instead of running individual commands, use the master orchestration script:

# On pod 1:
bash scripts/run_tier3_all.sh --pod p1

# On pod 2:
bash scripts/run_tier3_all.sh --pod p2

# On pod 3:
bash scripts/run_tier3_all.sh --pod p3

# Dry run (print commands without executing):
bash scripts/run_tier3_all.sh --pod all --dry-run

## Monitoring

Use:

bash scripts/monitor_pods.sh

Optional watch loop:

watch -n 30 bash scripts/monitor_pods.sh

## Pull and merge workflow

Use one staging folder per pod, then merge into local runs.

rsync -avz -e "ssh -p 26317" root@213.173.108.102:/root/scaling-operator-learning/runs/ runs_p1/
rsync -avz -e "ssh -p 30412" root@213.173.108.102:/root/scaling-operator-learning/runs/ runs_p2/
rsync -avz -e "ssh -p 21099" root@157.157.221.29:/root/scaling-operator-learning/runs/ runs_p3/

rsync -av runs_p1/ runs/
rsync -av runs_p2/ runs/
rsync -av runs_p3/ runs/

## Completion checklist

- Tier 1 text fixes applied in paper/main.tex.
- Tier 2 held-out prediction and within-slice bootstrap implemented and reported.
- Tier 3.1, 3.3, 3.4, 3.2, 3.5, 3.6 completed with logs saved.
- Final aggregate + multilaw analysis + figures regenerated.
- ROADMAP.md checkboxes updated and submission-ready PDF rebuilt.
