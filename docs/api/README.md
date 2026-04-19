# API Documentation

This directory contains the auto-generated API documentation for the internal Python packages used in the VLSI Decoder ML Optimizer project.

## Modules Covered
* `data_processing`: ETL and feature engineering pipelines.
* `models`: Surrogate model architectures and training scripts.
* `experiments`: Configuration parsing and logging utilities.

To generate the latest documentation, run:
```bash
sphinx-build -b html docs/api/source docs/api/build
```
