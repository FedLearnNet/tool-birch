# tool-birch

Clusters numeric tabular data with BIRCH and exports sample labels, subcluster centers, and a summary report.

## Input and outputs

Provide a CSV table as `features`. When the first column is named `id`
(case-insensitive), it is retained as an identifier and excluded from fitting.
All remaining feature columns must be numeric.

- `labels`: CSV with `id,label`, or `index,label` when no ID column is supplied.
  Returned when `compute_labels` is enabled.
- `subcluster_centers`: CSV containing the CF-tree subcluster centers.
- `report`: HTML summary of parameters, sample and feature counts, subclusters,
  and the number of output clusters.
- `cluster_size` metrics report the number of samples in each output cluster.

With `standardize` enabled, the centers use the standardized feature space.

## Configuration

- `threshold`: CF-tree radius threshold; default `0.5`.
- `branching_factor`: maximum subclusters per CF-tree node; default `50`.
- `use_global_clustering`: enable the final clustering step; default `true`.
- `n_clusters`: target cluster count for the final step; default `3`.
- `compute_labels`: generate labels and cluster-size metrics; default `true`.
- `standardize`: scale features to zero mean and unit variance; default `true`.

The repository includes the original Iris example tables in `data/`.
Use the numeric feature columns from those tables as inputs.

## Development

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env
# Set APP_ID and platform URLs in .env before connecting.
python main.py
```

The Docker base image provides the FL-Net API. `requirements.txt` contains only
additional runtime dependencies; `requirements-dev.txt` includes the API for local development.

Build with `docker build -t tool-birch .`.
