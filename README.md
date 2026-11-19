# Field-level PHI memorization

Code and synthetic data for **Not All Information is the Same: Characterizing Field-Level PHI Memorization in LLMs**, by Isha Agarwal, Krishnaveni Parvataneni, Cassandra Parent, and Marzyeh Ghassemi.

This release contains the methods in the active paper: matched synthetic PHI/benign documents, full-parameter Llama 3.1 8B Instruct fine-tuning, greedy prefix attacks, and first-data-token Jacobian-lens (J-Lens) analysis. It excludes autoregressive and template/oracle-lens experiments. Figure 3 was updated to match the reconstructed step-800 Figure 4 analysis: SSN/account uses mean RR across all tokens in the final four digits; other rows retain first-data-token RR. The original first-token-only heatmap is preserved under `results/archived/`.

## Contents

| Directory | Contents |
|---|---|
| `data/training/llama31_8b_balanced_2k_phi_benign_v2_luna_9f/` | Frozen synthetic corpus, name pools, structured pools, generation manifest |
| `src/memo_diff/dataset/` | Template rendering, field generation, name-pool procedure, validation |
| `src/memo_diff/training/` | Fine-tuning, document packing, prefix attacks |
| `src/memo_diff/analysis/` | Field scoring, token targets, Jacobian fitting and readout |
| `scripts/` | Dataset reproduction, training, scoring, statistics, plotting |
| `figures/` | The two main heatmaps and appendix aggregate-performance panel |
| `results/archived/` | Archived plot inputs and onset statistics |
| `tests/` | CPU checks for the retained methods |

The corpus has **1,000 matched pairs**: 800 training pairs (1,600 documents) and 200 held-out pairs (400 documents). Both members of each pair stay in the same split. All person/organization names and record identifiers are synthetic; geography is public reference data. SSN/account values retain the paper's `000-00-XXXX` or `999-00-XXXX` construction.

## Install

Use Python 3.11, preferably in a virtual environment:

```sh
python -m pip install -e '.[training,dev]'
python -m pytest -q
```

GPU experiments require access to the Llama 3.1 8B Instruct base weights and an appropriate CUDA/PyTorch environment. Model weights, fitted Jacobian tensors, optimizer states, credentials, and cluster-specific launch files are not distributed here. Historical runs used PyTorch 2.8.0 and Transformers 4.57.1; these core training versions are pinned in `requirements-research.txt` (`pip install -r requirements-research.txt`). They are not required for CPU-only plotting.

## Rebuild the released dataset

No API call is needed to reproduce the released documents from the frozen pools:

```sh
python scripts/reproduce_dataset.py --output-dir outputs/reproduced-data
```

This verifies that every generated training/held-out row exactly matches the released JSONL. The original optional name-pool generation procedure and prompts are retained in `dataset/luna_pools.py`; see [dataset details](docs/DATASET.md).

## Train and evaluate a fresh model

```sh
python scripts/train_llama_sft.py \
  --config configs/train_llama31_8b_balanced_2k_phi_benign_v2_luna_9f.yaml \
  --output-dir outputs/sft --no-wandb

python scripts/run_prefix_attack_single.py \
  --checkpoint outputs/sft/checkpoints/step_000800 --step 800 \
  --output-dir outputs/attack/step_800 \
  --samples-per-label 800 --membership train --add-special-tokens
```

Repeat the attack for the checkpoints of interest and combine their one-line `prefix_attack_by_checkpoint.jsonl` files before using `plot_prefix_attack_split_panels.py`. The evaluator uses greedy decoding, a 128-token continuation cap, complete-field matching in the corresponding field context, and eligible continuation-field denominators. **The retained current implementation splits near half the document's character length, aligned to a token boundary; it is not an exact half-token split.** See [reproduction limits](docs/REPRODUCIBILITY.md).

Fit and apply the first-token J-Lens:

```sh
python scripts/fit_paper_jacobian_checkpoint.py \
  --checkpoint outputs/sft/checkpoints/step_000800 --step 800 \
  --fit-texts data/jacobian_fit/fineweb_n1000.jsonl \
  --fit-prompts 1000 --max-seq-len 128 --skip-first 16 \
  --output-dir outputs/lens

python scripts/run_first_token_jlens.py \
  --checkpoint outputs/sft/checkpoints/step_000800 --step 800 \
  --jacobians outputs/lens/jacobians.pt --output-dir outputs/jlens \
  --samples-per-label 800 --ssn-last-four

python scripts/analyze_onsets.py \
  --input outputs/jlens/scores_shard_0.jsonl \
  --step 800 --output outputs/jlens/onsets.json
```

To plot a fresh numerical heatmap:

```sh
python scripts/aggregate_first_token_scores.py \
  --input outputs/jlens/scores_shard_0.jsonl --output outputs/jlens/summary.json
python scripts/replot_jlens_train_n800_mrr_heatmap.py \
  --summary outputs/jlens/summary.json --output outputs/jlens/mem_emergence.png
```

The fit and scoring commands support prompt/row sharding; use their `--help` output for the indices and shard counts. Every fit shard must complete before `merge_jacobian_fit_shards.py` is run. For scoring, concatenate disjoint `scores_shard_*.jsonl` files before onset analysis.

J-Lens scores the first data token given the gold preceding document. Literal `MRN-` and `HPB-` prefixes are supplied; the original default scores the SSN `000`/`999` prefix. Pass `--ssn-last-four` (as above) to use the current Figure 3 suffix metric instead. This measures conditional token accessibility, **not complete-field recall or causal memory storage**. Field-format and context shortcuts are described in [dataset details](docs/DATASET.md).

## Reproduce the paper figures

```sh
python scripts/reproduce_paper_figures.py --output-dir outputs/paper-figures
```

This rebuilds the archived prefix-attack heatmap, the **updated numerical Figure 3** (`mem_emergence.png/pdf`), and the archived appendix panel. Figure 3 reads `results/current/jlens_heatmap_inputs.json`, using the same 800 matched pairs, reconstructed step-800 checkpoint and metrics as the new Figure 4. All 22 plotted onset crossings are checked against that analysis. Its SSN/account row averages RR across both tokens in the random final four digits, given the gold prefix and earlier suffix tokens. Other rows use first-data-token RR.

To rebuild the historical first-token-only Figure 3, add `--archived-first-token`. Its preserved cell-color grids are **artwork, not recovered numerical scores**. The archived J-Lens filenames/caption refer to step 800 while the archived onset metadata says step 600; corresponding historical scores and checkpoint remain unrecovered. The updated Figure 3 is a fresh reconstructed-model analysis, not an exact replication or a one-row substitution into the old plot. The historical +17 SSN onset gap is not reproduced even by the same-run original-prefix control, so its disappearance cannot be attributed solely to suffix scoring.

The overview illustration (`fig_1.png`) is not in this code release. See [the artifact inventory and known discrepancies](docs/REPRODUCIBILITY.md).

## Citation and license

Citation metadata is in [CITATION.cff](CITATION.cff). Code is released under the [MIT license](LICENSE). Third-party data retain their own terms; see [THIRD_PARTY.md](THIRD_PARTY.md).
