# When Repeated Measurements Hurt Tabular Foundation Models

Code, configurations and saved results for the paper

> **When Repeated Measurements Hurt Tabular Foundation Models**
> Yixiang Liu\*, Ruizhe Deng\*, Bibhas Chakraborty, Qiong Zhang, Yan Shuo Tan
> (\* equal contribution)
> arXiv: link to be added

Tabular foundation models (TFMs) are usually evaluated on tables whose rows are independent
samples. Scientific data often record several responses at the same input. This repository
reproduces our study of what happens when such repeated readings are given to a TFM as
separate rows: predictions can improve at the measured inputs while getting worse elsewhere.

## Contents

Everything is in [`repeated-measurements/`](repeated-measurements/):

| Path | What it holds |
|---|---|
| `reproduce.py` | single entry point (`figures`, `smoke`, `run`) |
| `experiments/` | experiment scripts (main grid, real data, count interventions, theory checks) |
| `analysis/` | scripts that turn saved results into the paper's figures and tables |
| `config/` | model versions, experiment manifests, data sources and split specifications |
| `data/` | saved results and predictions used by the figures |
| `reproduced/` | the figures and tables as regenerated from `data/` |
| `environments/` | one environment specification per model backend |
| `SHA256SUMS` | checksums of the saved results |

## Quick start

Use Python 3.13 in a fresh environment:

```sh
cd repeated-measurements
python -m pip install -r requirements.txt
python reproduce.py figures   # rebuilds the figures and summaries from saved results (CPU, no model weights)
python reproduce.py smoke     # checks the synthetic data generation
```

## Re-running the models

Install the relevant environment from `environments/` and obtain the official weights listed in
`config/models.json`, then for example:

```sh
python reproduce.py run --experiment main --backend tabpfn3 -- --families synth1d --outdir runs/main
```

Set `CKPT_V3` for the TabPFN-v3 checkpoint, `LIMIX_REPO` for the LimiX checkout, and `HF_TOKEN`
where required. Experiment-specific settings are in `config/` and in the experiment scripts.

For the real-data experiments, obtain the datasets from the sources listed in
`config/data_sources.csv` and use the split specifications supplied in `config/`.

## Citation

```bibtex
@article{liu2026repeated,
  title   = {When Repeated Measurements Hurt Tabular Foundation Models},
  author  = {Liu, Yixiang and Deng, Ruizhe and Chakraborty, Bibhas and Zhang, Qiong and Tan, Yan Shuo},
  note    = {Preprint},
  year    = {2026}
}
```

## License

The code is released under the MIT License (see [`LICENSE`](LICENSE)). Third-party code, models
and data remain subject to their original licenses.
