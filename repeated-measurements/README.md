# Repeated measurements: reproduction materials

Code, configurations and saved results for the paper and appendix.

Use Python 3.13 in a fresh environment:

```sh
python -m pip install -r requirements.txt
python reproduce.py figures
python reproduce.py smoke
```

`figures` reconstructs twelve figures and statistical summaries from saved data on CPU, without model weights. `smoke` checks synthetic data generation.

For model runs, install the relevant environment from `environments/` and obtain the official weights listed in `config/models.json`:

```sh
python reproduce.py run --experiment main --backend tabpfn3 -- --families synth1d --outdir runs/main
```

Set `CKPT_V3` for the TabPFN-3 checkpoint, `LIMIX_REPO` for the LimiX checkout, and `HF_TOKEN` where required. Experiment-specific settings and training configurations are included in `config/` and the experiment scripts.

For real-data experiments, obtain the data from the sources listed in `config/data_sources.csv` and use the supplied split specifications in `config/`.

Third-party code, models and data remain subject to their original licenses.
