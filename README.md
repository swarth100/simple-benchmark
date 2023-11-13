# Simple Benchmark

Simple benchmarking utilities to config and run benchmarks for given user modules.
The runner can also be hosted as a webserver to give interactive user feedback on progress.
Currently the runner only supports python code for benchmarking.

The benchmark config must be expressed in `benchmark_config.yaml`.

The config is validated against the Pydantic schema in `config_validation.py`.

### System requirements

* `python` version 3.9 (can be managed via virtual environments)
* `libsqlite3-dev`, (or equivalent) usually installed via package manager

### Installing and running using Poetry

**NOTE**: Must have `poetry` installed: https://python-poetry.org/docs/#installation

```shell
poetry install
```

To run:

```shell
poetry run python runner.py
```

To show all run options and flags:

```shell
poetry run python runner.py --help
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

To show all run options and flags:

```shell
python runner.py --help
```