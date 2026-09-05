# Attribute-Conditioned Hypernetworks for Subgroup Fairness in Medical Image Classification

Code for an MSc Individual Project at Imperial College London.

The project tests one specific claim: that feeding a patient's sensitive attribute
(sex, race, age, skin type) to a classifier through a **hypernetwork**, so that the
network's weights vary with the attribute, raises the performance of the **worst-off
subgroup** without giving up overall discrimination. Four public de-identified
datasets, eight training arms and three experimental frameworks, all evaluated
out-of-fold under 5-fold cross-validation, together with a set of mechanism
diagnostics read directly off the weights and the decision boundary. Those
diagnostics exist to separate two very different explanations of a null result:
the attribute never reaching the network, and the attribute reaching it but not
helping.

## Contents

- [Datasets and arms](#datasets-and-arms)
- [Repository layout](#repository-layout)
- [Setup](#setup)
- [Data and splits](#data-and-splits)
- [Framework 1: in-distribution evaluation](#framework-1-in-distribution-evaluation)
- [Framework 2: transfer between datasets](#framework-2-transfer-between-datasets)
- [Framework 3: where the attribute is injected](#framework-3-where-the-attribute-is-injected)
- [Two extensions of HyperAdapt](#two-extensions-of-hyperadapt)
- [Mechanism diagnostics and report figures](#mechanism-diagnostics-and-report-figures)
- [Output layout](#output-layout)

## Datasets and arms

| Dataset | n | Target | Prevalence | Attributes | Grouping unit |
|---|---:|---|---:|---|---|
| HAM10000 | 9,707 | malignant | 14.5% | sex, age (4 bins) | lesion |
| Fitzpatrick17k | 16,012 | malignant | 13.5% | skin type (I–VI) | image |
| MIMIC-CXR | 199,356 | No Finding | 30.2% | sex, race, age (2 bins) | patient |
| CheXpert | 138,644 | No Finding | 8.2% | sex, race, age (2 bins) | patient |

Binary targets, attribute definitions and age binning follow the MEDFAIR benchmark.
Eight arms are compared:

| Arm | Where the attribute is used | Needs the attribute at inference |
|---|---|---|
| ERM | not used | — |
| SWAD | not used (weight averaging, derived per fold from ERM) | — |
| GroupDRO | training loss reweighted by subgroup | — |
| HyperHead | conditions the classifier head | yes |
| HyperFusion | conditions one deep block | yes |
| HyperAdapt | conditions every layer but the stem | yes |
| HyperAdapt+SWAD | conditioning plus weight averaging | yes |
| HyperAdapt(pred) | attribute predicted from the image instead of given | — |

## Repository layout

```
src/
├── paths.py              single source of every absolute path (see Setup)
├── models/               ResNet-18 and four conditioned variants (incl. CondNet for the ablation)
├── datasets/             the four datasets plus the soft-attribute wrapper
├── training/
│   ├── harness/          shared training loop, run naming, hyperparameter grid,
│   │                     subgroup AUC, SWAD, GroupDRO, cluster bootstrap
│   ├── train_<dataset>_<method>.py   one training entry point per (dataset, method)
│   ├── train_condnet.py              single entry point for Framework 3
│   └── run_ood_cxr_*.py              cross-dataset evaluation drivers
├── utils/                per-dataset fairness definitions, threshold metrics, hyperplane tools
scripts/                  splits, configuration selection, per-framework analysis, plotting
slurm/                    SLURM jobs (all training and re-inference goes through these)
configs/                  base configuration per dataset
data/splits/              5-fold cross-validation indices, shipped with the repository
report/                   LaTeX source of the report
docs/                     working notes and experiment plans
```

> `docs/` is the full record of the research process. Some of the experiments
> recorded there (PAPILA, ROC post-processing, synthetic-attribute injection,
> information-gate estimation) did not enter the final report and their code has
> been removed from this repository; it remains recoverable from the git history.
> The report in `report/` and this file are the authoritative description.

## Setup

Python 3.11.

```bash
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121
```

Every script imports from the repository root, so set `PYTHONPATH` before running
anything:

```bash
cd /path/to/hypernetworks_MSc_IP
export PYTHONPATH="$(pwd)"
```

All paths live in [`src/paths.py`](src/paths.py) and are overridden by environment
variables, so moving to another machine needs no code change:

| Variable | Meaning | Default |
|---|---|---|
| `HN_WORK_ROOT` | root for produced artefacts | grandparent of the repository |
| `HN_OUTPUTS` / `HN_LOGS` | outputs / logs directly | `$HN_WORK_ROOT/{outputs,logs}` |
| `HN_CXR_ROOT` | CXR7-1M root (MIMIC-CXR and CheXpert images plus master CSV) | shared-drive location |
| `HN_CHEXPERT_META` | CheXpert-Plus metadata directory (`chexbert_labels.zip`) | shared-drive location |
| `HN_HAM_ROOT` | HAM10000 root | shared-drive location |
| `HN_FITZ_ROOT` | Fitzpatrick17k root | shared-drive location |

SLURM jobs read three more: `HN_REPO_ROOT` for the repository, and
`HN_CONDA_ACTIVATE` / `HN_CONDA_ENV` for the environment (the activate step is
skipped when the file does not exist, and the current interpreter is used). Job
scripts write logs to a relative path, so run `mkdir -p logs` in the directory you
submit from.

Training belongs on a GPU cluster. The analysis and plotting scripts run on CPU.

The conditioned models build one set of weights per sample, so `HyperAdapt` and the
`CondNet` cells need far more memory at a given batch size than the baselines do:
at the batch size of 128 used throughout they do not fit on a 4 GB card, which is
why the hypernetwork arms are submitted to `gpus48`. Lowering the batch size makes
them run anywhere, but it also breaks the equal-batch comparison against the
baselines, so keep it for smoke tests only.

## Data and splits

All four datasets are public and de-identified; obtain access yourself, place them
locally and point the environment variables at them. The 5-fold indices in
`data/splits/` ship with the repository, so reproducing the reported numbers does
not require rebuilding them. To rebuild from the raw data (folds are grouped by the
unit in the table above, so no lesion and no patient crosses a fold boundary):

```bash
python scripts/build_ham10000_splits.py
python scripts/build_mimic_splits_nofinding.py
python scripts/build_chexpert_splits_nofinding.py
python scripts/build_fitzpatrick_splits.py        # single split first
python scripts/build_fitzpatrick_cv_splits.py     # folds are derived from it
```

Each of these reproduces the shipped indices byte for byte.

## Framework 1: in-distribution evaluation

For every (dataset, method) pair: six hyperparameter configurations over five folds,
one configuration chosen on the fold-averaged validation worst-group AUC, then the
five disjoint test folds assembled into one out-of-fold result.

**1. Train.** One job script per dataset; `PY_SCRIPT` names the training entry point
and may be relative to the repository root.

```bash
# HAM10000, ERM (SWAD=1 also derives the SWAD baseline from the same run)
sbatch --partition=gpus24 --job-name=ham_erm_cvsearch \
       --output=logs/ham_erm_cvsearch.%N.%A_%a.log \
       --export=ALL,PY_SCRIPT=src/training/train_ham10000_resnet18.py,SWAD=1 \
       slurm/c1_ham_cv_search.sh

# HAM10000, HyperAdapt conditioned on sex+age; hypernetwork arms use gpus48 to keep
# the batch size identical to the baselines
sbatch --partition=gpus48 --job-name=ham_ha_cvsearch \
       --output=logs/ham_ha_cvsearch.%N.%A_%a.log \
       --export=ALL,PY_SCRIPT=src/training/train_ham10000_hyperadapt.py,BATCH=128,COND=sex_age \
       slurm/c1_ham_cv_search.sh

# GroupDRO reuses the same array script; its dataset arrives in GDRO_DATASET,
# because the array script calls $PY_SCRIPT with a fixed argument list
sbatch --partition=gpus24 --job-name=mimic_gdro_cvsearch \
       --output=logs/mimic_gdro_cvsearch.%N.%A_%a.log \
       --export=ALL,PY_SCRIPT=src/training/train_groupdro.py,GDRO_DATASET=mimic \
       slurm/c2_mimic_cv_search.sh
```

Swap the job script for the other datasets: `c2_mimic_cv_search.sh` for MIMIC,
`c3_chexpert_cv_search.sh` for CheXpert, `c4_fitz_cv_search.sh` for Fitzpatrick17k.

**2. Select, 3. evaluate, 4. test.**

```bash
python scripts/select_config_cv.py --dataset ham10000 --merge   # writes selected_configs.json
python scripts/build_oof_results_averaging.py                    # primary estimator
python scripts/id_holm_correction.py                             # Holm-corrected verdicts
# per-subgroup detail; --config-json is required and names the file written in step 2
python scripts/cv_oof_report.py --dataset ham10000 --section all \
       --config-json "${HN_OUTPUTS:-$PWD/../../outputs}/ham10000/cv5/selected_configs.json"
```

> The estimator is **averaging**: a metric is computed within each fold and the folds
> are then averaged. Pooling raw scores across folds is not used, because per-fold
> calibration drift then manufactures false positives. `cv_oof_report.py` is the one
> exception: it prints the pooled view as a diagnostic, so read its numbers as a
> cross-check on the per-subgroup detail rather than as the reported result.
> `--section all` runs the threshold-based metrics too and takes considerably longer
> than the other sections.

**5. Between-group gaps and per-group accuracy at a threshold** (secondary endpoints,
no retraining).

```bash
# One-off prerequisite: the SWAD arms only stored test predictions, so the validation
# predictions have to be recomputed from the averaged weights (needs a GPU).
# Submit once per dataset, DATASET in {ham10000, fitzpatrick, mimic, chexpert}
sbatch --job-name=swadval_ham --output=logs/swadval_ham.%N.%A_%a.log \
       --export=ALL,DATASET=ham10000 slurm/dump_swad_val.sh

python scripts/build_group_levels_averaging.py                   # levels, gaps, per-group accuracy
python scripts/build_group_tables.py                             # tables for docs/ plus LaTeX
python scripts/build_group_tables.py --rule fpr20_val            # same tables under another rule
```

> The threshold is chosen **per fold on that fold's validation split** (one global,
> group-independent threshold; Youden's J by default), applied to every subgroup in
> that fold, and the metrics are averaged across folds, so the threshold never sees
> the evaluation data and is never pooled across folds. The three rules
> (`youden_val`, `fpr20_val`, `half`) are a sensitivity check rather than a menu.
>
> This step covers the four in-distribution datasets and both transfer directions in
> one pass. On the transfer side it reuses the cross-dataset CV predictions and takes
> the threshold from the **source** validation split, since the target dataset has no
> validation split and a deployed model can only carry the operating point it was
> given. Both endpoints are secondary, so their p-values are not part of the Holm
> family of the primary endpoint and should be read as exploratory.

## Framework 2: transfer between datasets

MIMIC-CXR and CheXpert in both directions. Each fold is trained under three seeds,
giving fifteen models, and every model scores the **entire** target dataset. An effect
counts only when it clears both the paired cluster bootstrap interval and the
training-noise floor sigma_train, in **both** directions.

```bash
# 1. Add the extra seeds (fold and seed are decoupled here; trial 0 is the model
#    Framework 1 already trained and is not retrained)
sbatch --partition=gpus24 --job-name=oodv2_mimic_erm \
       --output=logs/oodv2_mimic_erm.%N.%A_%a.log \
       --export=ALL,PY_SCRIPT=src/training/train_mimic_resnet18.py,CONFIG_INDEX=2,SWAD=1 \
       slurm/ood_trial.sh

# 2. Score the whole target dataset, both directions
sbatch --partition=gpus24 --job-name=oodv2_ft \
       --output=logs/oodv2_ft.%N.%A_%a.log slurm/oodv2_full_target.sh

# 3. Variance components and sigma_train, once per endpoint, written into the mw/ and
#    ov/ subdirectories the confirmatory family reads
FAM="${HN_OUTPUTS:-$PWD/../../outputs}/ood_cxr/variance_decomposition_family"
ARMS=erm,swad,groupdro,hyperhead,hyperfusion,hyperadapt
python scripts/ood_variance_decomposition.py --metric marginal_worst --methods $ARMS --out-dir $FAM/mw
python scripts/ood_variance_decomposition.py --metric overall        --methods $ARMS --out-dir $FAM/ov

# 4. The confirmatory family: one call reports both directions, nine comparisons each,
#    with delta, CI, Holm-corrected p and the ratio to sigma_train
python scripts/ood_confirmatory_family.py --src $FAM
```

## Framework 3: where the attribute is injected

`CondNet` produces four nested cells from a single switch (ERM, conditioning the fc
layer only, fc plus layer4, fc plus layer1-4). All four share one deterministically
seeded base state, so a difference between cells can only come from the conditioning
scope.

```bash
# 1. Six-configuration search, one submission per cell (LOCATION in none/head/deep/full)
sbatch --partition=gpus48 --job-name=e1_ham_full_search \
       --output=logs/e1_ham_full_search.%N.%A_%a.log \
       --export=ALL,DATASET=ham10000,LOCATION=full slurm/e1_condnet_search.sh
python scripts/e1_select_config.py --dataset ham10000

# 2. Extend the selected configuration to three seeds
sbatch --partition=gpus48 --job-name=e1_ham_full_confirm \
       --output=logs/e1_ham_full_confirm.%N.%A_%a.log \
       --export=ALL,DATASET=ham10000,LOCATION=full,CONFIG_INDEX=2 slurm/e1_condnet_confirm.sh

# 3. In-distribution analysis (the two increments as one Holm family)
python scripts/e1_ham_analysis.py

# 4. Transfer side: full-target evaluation of MIMIC to CheXpert, then the analysis
sbatch --partition=gpus24 slurm/e2_ood_full_target.sh
python scripts/e2_ood_analysis.py
```

The three interventions of the ablation:

```bash
# 1. Alternative checkpoint: reread everything at the epoch with the best validation
#    worst-group AUC instead of the best validation Overall AUC
python scripts/e1_ham_analysis.py --checkpoint worstcase
python scripts/e1_rho_aggregate.py --checkpoint worstcase

# 2. Pathway knockout: with the weights fixed, disable the head or the convolutional
#    pathway and compare against permuting the attribute
python scripts/e2_knockout_permutation.py

# 3. Retrain with the head pathway switched off, leaving the convolutional pathway as
#    the only route the attribute can take
sbatch --export=ALL,DATASET=ham10000,LOCATION=full_nofc,CONFIG_INDEX=2 slurm/e3_condnet_nofc.sh
python scripts/e3_nofc_analysis.py
```

## Two extensions of HyperAdapt

**A predicted attribute.** The conditioning input comes from a logistic probe on
frozen ImageNet features instead of the ground truth, with two zero-information
control arms (`perm` shuffles the prediction, `const` replaces it by the training-set
marginal).

```bash
sbatch --export=ALL,DS=mimic slurm/p_attr_pred.sh              # fit the probe per fold
sbatch --partition=gpus48 \
       --export=ALL,PY_SCRIPT=src/training/train_fitzpatrick_hyperadapt_pred.py,ATTR_MODE=soft \
       slurm/p_pred_attr.sh                                     # ATTR_MODE in soft/hard/perm/const
python scripts/p_pred_attr_analysis.py --dataset fitzpatrick
python scripts/p_ood_analysis.py --direction m2c
```

**Weight averaging.** Train a HyperAdapt arm with `SWAD=1` (the run is named
`hyperadapt_swad`); batch-normalisation statistics are additionally re-estimated with
the official `update_bn` as a control.

```bash
sbatch --export=ALL,DATASET=ham10000,STAGE=id slurm/g_bn_update.sh
python scripts/g_fusion_analysis.py            # the 2x2 table and the interaction contrast
python scripts/g_mde_worstgroup.py             # minimum detectable effect on worst-group AUC
```

## Mechanism diagnostics and report figures

These answer whether the attribute reached the network at all. They read trained
weights or stored predictions; most need no images.

```bash
# Per-layer conditioning strength rho and rho_between
# (12 trained arms across 4 datasets, CPU only, no images read)
python scripts/named_hn_rho.py

# Per-layer rho for the ablation cells. Two ways to produce the per-run rho:
#   --mode rho  is CPU only and reads the checkpoints alone (enough for the rho figures)
#   the GPU job additionally builds the logit tables the knockout analysis needs
python scripts/e1_rho_and_permutation.py --mode rho     # CPU
sbatch slurm/e1_rho_logits.sh overall                   # GPU, also writes the logit tables
python scripts/e1_rho_aggregate.py                      # aggregate either of the two

# Counterfactual sweep of the decision hyperplane per subgroup
# (discriminative axis by between-group adjustment axis, with counterfactual AUC)
python scripts/viz_hyperplane_ham_sexage.py                     # HAM (the sex+age version used in the report)
python scripts/viz_hyperadapt_hyperplane_mimic.py               # MIMIC
python scripts/viz_hyperadapt_hyperplane_chexpert.py            # CheXpert
python scripts/viz_hyperadapt_hyperplane_fitzpatrick.py         # Fitzpatrick17k
python scripts/viz_blind_hyperplane.py --dataset ham --method erm   # attribute-blind baseline for contrast
# The transfer geometry reuses the in-distribution artefacts, so run the two chest
# X-ray scripts above first
python scripts/viz_ood_hyperplane_cxr.py --direction m2c

# Weight trajectories under an attribute sweep (no images needed)
python scripts/viz_hyperadapt_weight_trajectory.py              # HAM, age swept continuously
python scripts/viz_hyperadapt_trajectory_mimic.py               # MIMIC, three attributes

# Training curves, and the four figures of the results chapter
python scripts/plot_training_curves.py
python scripts/plot_report_ch5_figures.py                       # writes into report/figures/
```

The hyperplane scripts default to the fold-0 checkpoint of each dataset's selected
configuration; `--ckpt` points them at another one.

## Output layout

Checkpoints, per-fold predictions and analysis JSON are written outside the
repository, under `$HN_OUTPUTS` (by default `../../outputs`), and are not version
controlled:

```
outputs/
├── <dataset>/cv5/                checkpoints, validation logs, predictions/, selected_configs.json
├── conditioning_ablation/        the four cells of Framework 3, primary-estimator JSON,
│                                 and the per-group level / gap / accuracy JSON
├── ood_cxr/{m2c,c2m}/            cross-dataset predictions and variance decomposition
└── analysis/                     rho, Holm verdicts, hyperplane and trajectory artefacts
```

A run is identified by the triple `(method, config_tag, seed)`, and checkpoints and
validation logs share that name, so six configurations, several seeds and five folds
never overwrite one another. In distribution the seed follows the fold
(`seed = 42 + fold`); the transfer framework adds two independent seeds
(`seed = 42 + fold + 10 * trial`) so that training randomness can be separated from
the split.
