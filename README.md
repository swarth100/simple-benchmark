# Simple Benchmark

Simple benchmarking utilities to config and run benchmarks for given user modules.

The benchmark config must be expressed in `benchmark_config.yaml`.

The config is validated against the Pydantic schema in `config_validation.py`.

### Installing and running using Poetry

**NOTE**: Must have `poetry` installed: https://python-poetry.org/docs/#installation

```shell
poetry install
```

To run:

```shell
poetry run python runner.py
```

### Installing and running manually

**NOTE**: You must be in the directory where this project was cloned/downloaded

```shell
python -m pip install .
```

To run:

```shell
python runner.py
```