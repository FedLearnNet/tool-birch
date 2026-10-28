from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np
import pandas as pd
from pydantic.dataclasses import dataclass
from pyfedappwrap.learning.app_types import BaseSelfLearnedApp
from pyfedappwrap.learning.run_runfig import AppConfig, AppInputConfig, AppOutputConfig
from sklearn.cluster import Birch
from sklearn.preprocessing import StandardScaler


@dataclass
class BirchAppConfig(AppConfig):
    threshold: float = 0.5
    branching_factor: int = 50
    use_global_clustering: bool = 1
    n_clusters: int = 3
    compute_labels: bool = 1
    copy: bool = 1
    standardize: bool = 1


@dataclass
class BirchAppInputConfig(AppInputConfig):
    features: Any = None


@dataclass
class BirchAppOutputConfig(AppOutputConfig):
    labels: Path | None = None
    subcluster_centers: Path | None = None
    report: str | None = None


class BirchAPP(BaseSelfLearnedApp[BirchAppConfig, BirchAppInputConfig, BirchAppOutputConfig]):

    def run_algorithm(self, data: BirchAppInputConfig) -> BirchAppOutputConfig:
        df: pd.DataFrame = data.features
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input 'features' must be a pandas DataFrame.")

        ids, feats = _split_id_and_features(df)
        X = _ensure_numeric_matrix(feats)
        X = _maybe_standardize(X, self.config.standardize)

        n_clusters = self.config.n_clusters if bool(self.config.use_global_clustering) else None

        model = Birch(
            threshold=float(self.config.threshold),
            branching_factor=int(self.config.branching_factor),
            n_clusters=n_clusters,
            compute_labels=bool(self.config.compute_labels),
        )

        self.logger.info(f"Fitting Birch on {X.shape[0]} samples with {X.shape[1]} features")
        model.fit(X)
        self.logger.info("Model fitting complete")

        # Outputs
        out_labels_path = Path("labels.csv")
        out_centers_path = Path("subcluster_centers.csv")

        labels = None
        if bool(self.config.compute_labels):
            labels = model.labels_
            _write_labels_csv(out_labels_path, ids, labels)

        centers = getattr(model, "subcluster_centers_", None)
        if centers is not None:
            centers_df = pd.DataFrame(centers, columns=list(feats.columns))
            centers_df.to_csv(out_centers_path, index=False)

        # Simple metrics (optional)
        if labels is not None:
            unique, counts = np.unique(labels, return_counts=True)
            for k, c in zip(unique.tolist(), counts.tolist()):
                self.send_metric("cluster_size", float(c), float(k))

        report = _html_report(
            "BIRCH Report",
            params={
                "threshold": self.config.threshold,
                "branching_factor": self.config.branching_factor,
                "use_global_clustering": bool(self.config.use_global_clustering),
                "n_clusters": n_clusters,
                "compute_labels": bool(self.config.compute_labels),
                "copy": bool(self.config.copy),
                "standardize": bool(self.config.standardize),
            },
            extra={
                "n_samples": X.shape[0],
                "n_features": X.shape[1],
                "n_subclusters": None if centers is None else int(np.asarray(centers).shape[0]),
                "n_output_clusters": None if labels is None else int(len(np.unique(labels))),
            },
        )

        return BirchAppOutputConfig(
            labels=out_labels_path if bool(self.config.compute_labels) else None,
            subcluster_centers=out_centers_path if centers is not None else None,
            report=report,
        )


def _split_id_and_features(df: pd.DataFrame) -> Tuple[Optional[pd.Series], pd.DataFrame]:
    """
    If first column is named 'id' (case-insensitive), treat it as identifier.
    Otherwise return None and full df as features.
    """
    if df.shape[1] == 0:
        raise ValueError("Input dataframe has no columns.")

    first_col = str(df.columns[0]).strip().lower()
    if first_col == "id":
        ids = df.iloc[:, 0]
        feats = df.iloc[:, 1:]
        return ids, feats
    return None, df


def _ensure_numeric_matrix(feats: pd.DataFrame) -> np.ndarray:
    # Will raise if cannot convert
    X = feats.apply(pd.to_numeric, errors="raise").to_numpy(dtype=float)
    if X.ndim != 2 or X.shape[1] == 0:
        raise ValueError("Feature matrix must be 2D with at least one feature column.")
    return X


def _maybe_standardize(X: np.ndarray, standardize: bool) -> np.ndarray:
    if bool(standardize):
        return StandardScaler().fit_transform(X)
    return X


def _write_labels_csv(path: Path, ids: Optional[pd.Series], labels: np.ndarray) -> None:
    if ids is None:
        out = pd.DataFrame({"index": np.arange(len(labels)), "label": labels})
    else:
        out = pd.DataFrame({"id": ids.values, "label": labels})
    out.to_csv(path, index=False)


def _html_report(title: str, params: dict, extra: dict) -> str:
    def _rows(d: dict) -> str:
        return "\n".join(
            f"<tr><td><code>{k}</code></td><td>{v}</td></tr>"
            for k, v in d.items()
        )

    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>{title}</title>
  <style>
    body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; margin: 24px; }}
    table {{ border-collapse: collapse; width: 100%; max-width: 980px; }}
    td, th {{ border: 1px solid #ddd; padding: 8px; vertical-align: top; }}
    th {{ background: #f6f6f6; text-align: left; }}
    code {{ background: #f1f1f1; padding: 1px 4px; border-radius: 4px; }}
  </style>
</head>
<body>
  <h2>{title}</h2>

  <h3>Parameters</h3>
  <table>
    <tr><th>Key</th><th>Value</th></tr>
    {_rows(params)}
  </table>

  <h3>Summary</h3>
  <table>
    <tr><th>Key</th><th>Value</th></tr>
    {_rows(extra)}
  </table>
</body>
</html>
"""
