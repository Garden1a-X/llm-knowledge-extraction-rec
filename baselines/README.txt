================================================================================
Baseline Models with RecBole
================================================================================

This directory contains scripts to run baseline recommendation models using
the RecBole library (https://recbole.io/).

RecBole provides implementations of 80+ recommendation algorithms, ensuring
correctness and fair comparison.

================================================================================
Available Scripts
================================================================================

Data Preparation:
  prepare_data_for_recbole.py    Convert MovieLens to RecBole format

Single Run:
  run_baseline.py                Run one model with specific parameters

Multiple Trials:
  run_5_trials.sh                Quick: Run 5 trials with default seeds
  run_multiple_trials.py         Flexible: Custom number of trials and seeds
  extract_results_from_json.py   Extract statistics from completed trials

Batch Operations:
  run_all_baselines.sh           Run all baseline models once

================================================================================
Setup
================================================================================

1. Install RecBole:

   pip install recbole

   Or with conda:
   conda install -c aibox recbole

2. Prepare data for RecBole:

   python baselines/prepare_data_for_recbole.py \
       --ml_data_dir /path/to/ml-1m \
       --output_dir data/recbole/ml-1m

   This converts MovieLens 1M to RecBole format (.inter, .user, .item files).

================================================================================
Baseline Models
================================================================================

We will run the following baseline models:

1. LightGCN (SIGIR 2020)
   - Pure collaborative filtering with graph convolution
   - No knowledge graph
   - Expected NDCG@10: 0.18-0.22 on ML-1M

2. BPR (UAI 2009)
   - Matrix factorization with BPR loss
   - Classic baseline
   - Expected NDCG@10: 0.12-0.16 on ML-1M

3. NGCF (SIGIR 2019)
   - Neural graph collaborative filtering
   - More complex GNN
   - Expected NDCG@10: 0.16-0.20 on ML-1M

4. KGAT (KDD 2019) [Requires KG preparation]
   - Knowledge graph attention network
   - Uses metadata as knowledge graph
   - Expected NDCG@10: 0.20-0.24 on ML-1M

5. RippleNet (CIKM 2018) [Requires KG preparation]
   - Preference propagation on KG
   - Expected NDCG@10: 0.18-0.22 on ML-1M

Note: KGAT and RippleNet require knowledge graph data, which we'll prepare
when implementing those baselines.

================================================================================
Usage
================================================================================

Single model:
-------------

python baselines/run_baseline.py \
    --model LightGCN \
    --dataset ml-1m \
    --data_path data/recbole \
    --device cuda \
    --epochs 300

All models (except KG-based):
-----------------------------

bash baselines/run_all_baselines.sh

Custom parameters:
------------------

python baselines/run_baseline.py \
    --model LightGCN \
    --dataset ml-1m \
    --epochs 100 \
    --batch_size 1024 \
    --lr 0.001 \
    --embedding_size 64 \
    --device cuda

================================================================================
Output
================================================================================

Results will be saved to: outputs/baselines/MODEL_DATASET_TIMESTAMP/
  - config.json: Configuration used
  - results.json: Test set performance
  - checkpoints/: Saved model checkpoints

Example output:
  outputs/baselines/LightGCN_ml-1m_20241226_120000/
    ├── config.json
    ├── results.json
    └── checkpoints/
        └── LightGCN-ml-1m.pth

================================================================================
Expected Results on MovieLens 1M
================================================================================

Based on RecBole documentation and papers:

Model         NDCG@10   Recall@10   Precision@10   Hit@10
-----------   -------   ---------   ------------   ------
BPR           0.12-0.16  0.20-0.25   0.08-0.12     0.40-0.50
NGCF          0.16-0.20  0.24-0.30   0.10-0.14     0.45-0.55
LightGCN      0.18-0.22  0.25-0.32   0.11-0.15     0.50-0.60
KGAT          0.20-0.24  0.28-0.34   0.12-0.16     0.52-0.62
RippleNet     0.18-0.22  0.26-0.32   0.11-0.15     0.50-0.60

Note: Exact values depend on:
- Data preprocessing (user/item filtering)
- Evaluation protocol (negative sampling vs full ranking)
- Hyperparameters
- Random seed

Our goal: Ours-Full should achieve NDCG@10 > 0.24 (better than KGAT)

================================================================================
Available Models in RecBole
================================================================================

General Recommendation:
- BPR, NeuMF, NGCF, LightGCN, SGL, SimGCL, NCL

Knowledge-aware:
- KGAT, KGIN, RippleNet, MKR, KTUP

Sequential:
- GRU4Rec, SASRec, BERT4Rec

Context-aware:
- FM, DeepFM, Wide&Deep, xDeepFM

See: https://recbole.io/model_list.html

================================================================================
References
================================================================================

RecBole:
  Zhao et al., RecBole: Towards a Unified, Comprehensive and Efficient
  Framework for Recommendation Algorithms, CIKM 2021

LightGCN:
  He et al., LightGCN: Simplifying and Powering Graph Convolution Network
  for Recommendation, SIGIR 2020

KGAT:
  Wang et al., KGAT: Knowledge Graph Attention Network for Recommendation,
  KDD 2019

RippleNet:
  Wang et al., RippleNet: Propagating User Preferences on the Knowledge
  Graph for Recommender Systems, CIKM 2018

================================================================================
Running Multiple Trials (RECOMMENDED for Papers)
================================================================================

For stable results with mean and standard deviation, run multiple trials with
different random seeds.

Quick start (5 trials):
-----------------------

bash baselines/run_5_trials.sh LightGCN

This will run 5 trials with seeds: [42, 2023, 2024, 2025, 12345]

Extract and analyze results:
---------------------------

python baselines/extract_results_from_json.py outputs/baselines

This will:
- Find all MODEL_DATASET_TIMESTAMP/results.json files
- Extract test_result from each trial
- Compute mean ± std for all metrics
- Output Markdown table (for README)
- Output LaTeX table (for paper)
- Save to MODEL_summary_Ntrials.json (e.g., LightGCN_summary_5trials.json)

Example output:
--------------

NDCG@10:   0.1263 ± 0.0045
Recall@10: 0.0578 ± 0.0023
...

Why multiple trials?
--------------------

1. Single runs can be unstable due to random initialization
2. Papers require reporting mean and standard deviation
3. Allows statistical significance testing
4. Ensures reproducibility

Recommended: Run 3-5 trials for each baseline and your proposed method.

Advanced usage:
---------------

Run custom trials with specific seeds:

python baselines/run_multiple_trials.py \
    --model LightGCN \
    --num_trials 5 \
    --seeds 42 2023 2024 2025 12345 \
    --epochs 300

================================================================================
Next Steps
================================================================================

After running baselines:

1. Run multiple trials for each baseline (3-5 trials each)
2. Compare results and establish baseline performance
3. For KGAT/RippleNet: Prepare KG from MovieLens metadata
4. Implement our innovation: LLM-based visual knowledge extraction
5. Run multiple trials for Ours-Visual, Ours-Interest, Ours-Full
6. Statistical comparison against baselines

================================================================================
