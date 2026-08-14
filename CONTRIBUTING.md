# Contributing

Thanks for helping improve PJM DART Spike Prediction. Contributions that strengthen reproducibility, point-in-time correctness, tests, documentation, or market assumptions are especially welcome.

## Before opening a change

- Search existing issues and pull requests.
- Open an issue first for methodological changes, new datasets, model changes, or changes to reported results.
- Keep pull requests focused on one concern.

## Local setup

```bash
git clone https://github.com/whoIsClownHere/pjm-dart-prediction.git
cd pjm-dart-prediction
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Data collection requires a GridStatus API key:

```dotenv
GRIDSTATUS_API_KEY=your_key
```

Store it in a local `.env` file. Never commit credentials, raw API responses, or files under `input_data/`.

To build the local datasets and features:

```bash
mkdir -p input_data
python -m scripts.data.data_collection
python -m scripts.data.features
```

## Research integrity requirements

Every contribution must preserve these invariants:

- A feature may use only information available before the 11:00 EPT day-ahead gate for its delivery day.
- Feature availability must be demonstrated with timestamps, not inferred from row order.
- Labels must be built from raw, unclipped DART values.
- Training, validation, and test periods must remain chronological.
- The test period must not influence model, feature, label, or threshold selection.
- Changes affecting results must regenerate and explain the reported metrics and figures.
- Trading assumptions, costs, missing data, and exclusions must be explicit.

## Tests

Run:

```bash
pytest -q
```

Some existing integration tests require locally collected data. New unit tests should prefer small synthetic fixtures and must not require network access or secrets.

For time-sensitive features, include a regression test proving that every source observation predates the applicable market cutoff.

## Code and documentation

- Prefer small functions with clear inputs and outputs.
- Add type hints and docstrings where they clarify behavior.
- Keep configuration in `scripts/config.py` rather than duplicating constants.
- Use UTC internally and convert to Eastern Prevailing Time only where market rules require it.
- Update the README when commands, assumptions, or public results change.
- Do not commit notebook checkpoints, virtual environments, or generated private data.

## Pull requests

A pull request should describe:

- what changed and why;
- the affected data and date ranges;
- how point-in-time availability was verified;
- tests and reproduction commands run;
- whether public metrics or figures changed.

By contributing, you agree that your work will be licensed under the repository's MIT License.
