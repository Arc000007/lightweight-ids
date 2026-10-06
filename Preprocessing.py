"""
Preprocessing module for IDS pipeline.
Handles all datasets with auto-detection from dataset_paths.py

Supported datasets and their known structures
─────────────────────────────────────────────
CICIoT2023      : subdirs  train/ test/ validation/  *.csv
TON_IoT         : Train_Test_datasets/Train_Test_IoT_dataset/  *.csv
IoT-23          : flat directory  *.csv
CICIDS2017      : flat directory  *.pcap_ISCX.csv   (label col has leading space)
CSE-CIC-IDS2018 : flat directory  *.csv
Edge-IIoTset    : Edge-IIoTset dataset/
                    ├── Selected dataset for ML and DL/  ← preferred
                    ├── Attack traffic/
                    └── Normal traffic/
"""

import os
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils import resample

logger = logging.getLogger(__name__)

try:
    from dataset_paths import DATASET_PATHS
except ImportError:
    logger.warning("dataset_paths.py not found — will rely on data_path argument only")
    DATASET_PATHS = {}


# ─────────────────────────────────────────────────────────────────────────────
# Per-dataset configuration
# ─────────────────────────────────────────────────────────────────────────────

DATASET_CONFIGS: Dict[str, Dict] = {
    "CICIoT2023": {
        "label_col": "label",
        "benign_label": "BenignTraffic",
        "drop_cols": ["flow_id", "src_ip", "dst_ip", "src_mac", "dst_mac", "timestamp"],
        "strip_label": False,
    },
    "TON_IoT": {
        "label_col": "type",
        "benign_label": "normal",
        "drop_cols": ["ts", "uid", "id.orig_h", "id.resp_h"],
        "strip_label": False,
    },
    "IoT-23": {
        "label_col": "label",
        "benign_label": "Benign",
        "drop_cols": ["ts", "uid", "id.orig_h", "id.resp_h"],
        "strip_label": False,
    },
    # CICIDS2017 label column is " Label" — note the leading space produced by
    # CICFlowMeter.  We strip whitespace from both column names and values.
    "CICIDS2017": {
        "label_col": "Label",          # matched after strip()
        "benign_label": "BENIGN",
        "drop_cols": [],               # no index column in the pcap_ISCX files
        "strip_label": True,           # strip whitespace from col names & values
    },
    "CSE-CIC-IDS2018": {
        "label_col": "Label",
        "benign_label": "Benign",
        "drop_cols": ["Timestamp"],
        "strip_label": False,
    },
    # Edge-IIoTset "Selected dataset for ML and DL" files already have
    # Attack_type as the label column; Normal traffic rows are labelled "Normal".
    "Edge-IIoTset": {
        "label_col": "Attack_type",
        "benign_label": "Normal",
        "drop_cols": [
            "frame.time", "ip.src_host", "ip.dst_host",
            "arp.src.proto_ipv4", "arp.dst.proto_ipv4",
            "http.file_data", "http.request.full_uri",
            "icmp.transmit_timestamp", "dns.qry.name",
        ],
        "strip_label": False,
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# DataPreprocessor
# ─────────────────────────────────────────────────────────────────────────────

class DataPreprocessor:
    """Unified preprocessor for all supported IDS datasets."""

    def __init__(self, dataset_name: str, data_path: str,
                 test_size: float = 0.2, val_size: float = 0.1,
                 random_state: int = 42):
        if dataset_name not in DATASET_CONFIGS:
            raise ValueError(
                f"Unknown dataset '{dataset_name}'. "
                f"Supported: {list(DATASET_CONFIGS.keys())}"
            )
        self.dataset_name  = dataset_name
        self.data_path     = data_path
        self.cfg           = DATASET_CONFIGS[dataset_name]
        self.test_size     = test_size
        self.val_size      = val_size
        self.random_state  = random_state

        self.scaler             = StandardScaler()
        self.label_encoder      = LabelEncoder()
        self.feature_names: List[str] = []
        self.n_features: int    = 0
        self.class_distribution: Dict = {}

    # ── I/O ──────────────────────────────────────────────────────────────────

    def load_data(self, max_rows: Optional[int] = None) -> pd.DataFrame:
        """Detect whether data_path is a file or directory and dispatch."""
        path = self.data_path

        if os.path.isfile(path):
            return self._load_file(path, max_rows)

        if os.path.isdir(path):
            return self._load_directory(path, max_rows)

        raise FileNotFoundError(f"data_path not found: {path}")

    def _load_file(self, path: str, max_rows: Optional[int]) -> pd.DataFrame:
        ext = Path(path).suffix.lower()
        if ext == ".parquet":
            df = pd.read_parquet(path)
            if max_rows:
                df = df.head(max_rows)
        elif ext in (".csv", ".tsv"):
            df = pd.read_csv(path, nrows=max_rows, low_memory=False)
        else:
            raise ValueError(f"Unsupported file extension: {ext}")
        logger.info(f"Loaded {len(df):,} rows from {path}")
        return df

    def _load_directory(self, dir_path: str, max_rows: Optional[int]) -> pd.DataFrame:
        """Route to dataset-specific directory loader."""
        loaders = {
            "CICIoT2023":      self._load_ciciot2023,
            "TON_IoT":         self._load_ton_iot,
            "CICIDS2017":      self._load_cicids2017,
            "CSE-CIC-IDS2018": self._load_flat_csvs,
            "IoT-23":          self._load_flat_csvs,
            "Edge-IIoTset":    self._load_edge_iiotset,
        }
        loader = loaders.get(self.dataset_name, self._load_recursive)
        frames = loader(dir_path)

        if not frames:
            raise FileNotFoundError(
                f"No CSV/Parquet files found under {dir_path}"
            )

        df = pd.concat(frames, ignore_index=True)
        if max_rows:
            df = df.head(max_rows)
        logger.info(
            f"Loaded {len(df):,} rows from directory '{dir_path}' "
            f"({len(frames)} file(s))"
        )
        return df

    # ── Dataset-specific loaders ─────────────────────────────────────────────

    def _load_ciciot2023(self, root: str) -> List[pd.DataFrame]:
        frames = []
        for subdir in ("train", "test", "validation"):
            p = os.path.join(root, subdir)
            if os.path.isdir(p):
                for fname in sorted(os.listdir(p)):
                    if fname.endswith(".csv"):
                        frames.append(self._read_csv(os.path.join(p, fname)))
        if not frames:                       # flat layout fallback
            frames = self._load_flat_csvs(root)
        return frames

    def _load_ton_iot(self, root: str) -> List[pd.DataFrame]:
        candidate = os.path.join(
            root, "Train_Test_datasets", "Train_Test_IoT_dataset"
        )
        if os.path.isdir(candidate):
            frames = [
                self._read_csv(os.path.join(candidate, f))
                for f in sorted(os.listdir(candidate))
                if f.endswith(".csv")
            ]
            if frames:
                return frames
        return self._load_recursive(root)

    def _load_cicids2017(self, root: str) -> List[pd.DataFrame]:
        """
        CICIDS2017 files live directly in the folder and are named
        <Day>-<Details>.pcap_ISCX.csv  (or just *.csv after export).
        The label column produced by CICFlowMeter has a leading space:
        ' Label' → we normalise column names after loading.
        """
        frames = []
        for fname in sorted(os.listdir(root)):
            fpath = os.path.join(root, fname)
            if not os.path.isfile(fpath):
                continue
            # Accept both raw CICFlowMeter exports and renamed copies
            if fname.endswith(".csv") or fname.endswith(".pcap_ISCX.csv"):
                df = self._read_csv(fpath)
                # Normalise column names — strip surrounding whitespace
                df.columns = [c.strip() for c in df.columns]
                frames.append(df)
        return frames

    def _load_edge_iiotset(self, root: str) -> List[pd.DataFrame]:
        """
        Edge-IIoTset directory layout
        ──────────────────────────────
        Edge-IIoTset/
        └── Edge-IIoTset dataset/
            ├── Selected dataset for ML and DL/   ← best choice
            │   ├── DNN-EdgeIIoT-dataset.csv
            │   └── ...
            ├── Attack traffic/
            │   ├── DDoS attacks/
            │   └── ...  (nested)
            └── Normal traffic/
                └── ...

        We prefer the pre-selected ML/DL directory.  If absent we combine
        Attack traffic + Normal traffic directories.
        """
        base = os.path.join(root, "Edge-IIoTset dataset")
        if not os.path.isdir(base):
            base = root          # user pointed directly at the inner folder

        ml_dir = os.path.join(base, "Selected dataset for ML and DL")
        if os.path.isdir(ml_dir):
            logger.info("Edge-IIoTset: using 'Selected dataset for ML and DL'")
            frames = self._load_recursive(ml_dir)
            if frames:
                return frames

        # Fallback: combine Attack + Normal traffic trees
        logger.info(
            "Edge-IIoTset: 'Selected dataset' not found — "
            "combining Attack traffic + Normal traffic"
        )
        frames = []
        for subdir in ("Attack traffic", "Normal traffic"):
            p = os.path.join(base, subdir)
            if os.path.isdir(p):
                sub_frames = self._load_recursive(p)
                logger.info(
                    f"  {subdir}: loaded {len(sub_frames)} file(s)"
                )
                frames.extend(sub_frames)
        return frames

    def _load_flat_csvs(self, root: str) -> List[pd.DataFrame]:
        """Load all CSV files in a single flat directory (non-recursive)."""
        frames = []
        for fname in sorted(os.listdir(root)):
            fpath = os.path.join(root, fname)
            if os.path.isfile(fpath) and fname.endswith(".csv"):
                frames.append(self._read_csv(fpath))
        return frames

    def _load_recursive(self, root: str) -> List[pd.DataFrame]:
        """Walk the full directory tree and load every CSV/Parquet."""
        frames = []
        for dirpath, _, files in os.walk(root):
            for fname in sorted(files):
                fpath = os.path.join(dirpath, fname)
                if fname.endswith(".parquet"):
                    try:
                        frames.append(pd.read_parquet(fpath))
                        logger.info(f"  {os.path.relpath(fpath, root)}")
                    except Exception as e:
                        logger.warning(f"  skip {fpath}: {e}")
                elif fname.endswith(".csv"):
                    frames.append(self._read_csv(fpath))
        return frames

    @staticmethod
    def _read_csv(fpath: str) -> pd.DataFrame:
        try:
            df = pd.read_csv(fpath, low_memory=False)
            logger.info(f"  {os.path.basename(fpath)}: {len(df):,} rows")
            return df
        except Exception as e:
            logger.warning(f"  skip {fpath}: {e}")
            return pd.DataFrame()

    # ── Cleaning ─────────────────────────────────────────────────────────────

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        1. Strip whitespace from column names (fixes CICIDS2017 ' Label').
        2. Drop identifier / metadata columns.
        3. Replace ±inf → NaN, fill NaN with column median.
        4. Drop zero-variance columns.
        5. Drop exact duplicate rows.
        """
        # 1. Normalise column names
        df.columns = [c.strip() for c in df.columns]

        # 2. Drop configured columns
        drop = [c for c in self.cfg["drop_cols"] if c in df.columns]
        if drop:
            df = df.drop(columns=drop)

        # 3. Fix infinities / NaNs in numeric columns
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        num_cols = df.select_dtypes(include=[np.number]).columns
        df[num_cols] = df[num_cols].fillna(df[num_cols].median())

        # 4. Drop constant columns
        const = [c for c in num_cols if c in df.columns and df[c].std() == 0]
        if const:
            logger.info(f"Dropping {len(const)} constant columns")
            df.drop(columns=const, inplace=True)

        # 5. Deduplicate
        before = len(df)
        df.drop_duplicates(inplace=True)
        removed = before - len(df)
        if removed:
            logger.info(f"Removed {removed:,} duplicate rows")

        return df

    # ── Label encoding ────────────────────────────────────────────────────────

    def encode_labels(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Find the label column, binary-encode it (0=benign, 1=attack),
        and drop it from the feature frame.

        Handles leading/trailing whitespace in both column names and values
        (needed for CICIDS2017 and similar CICFlowMeter exports).
        """
        target = self.cfg["label_col"]
        strip  = self.cfg.get("strip_label", False)

        # Strip all column names first (clean() already does this, but be safe)
        df.columns = [c.strip() for c in df.columns]

        if target not in df.columns:
            # Last-ditch: case-insensitive match
            match = {c.lower(): c for c in df.columns}
            target = match.get(target.lower(), target)

        if target not in df.columns:
            raise ValueError(
                f"Label column '{self.cfg['label_col']}' not found. "
                f"Available columns: {list(df.columns[:10])} ..."
            )

        raw_labels = df[target].astype(str)
        if strip:
            raw_labels = raw_labels.str.strip()

        benign = self.cfg["benign_label"]
        y = (raw_labels != benign).astype(np.int32).values   # 0=benign 1=attack

        self.class_distribution = raw_labels.value_counts().to_dict()
        logger.info(
            f"Class distribution (top 5): "
            f"{ {k: v for k, v in list(self.class_distribution.items())[:5]} }"
        )

        df = df.drop(columns=[target], errors="ignore")
        return df, y

    # ── Categorical encoding ──────────────────────────────────────────────────

    def encode_categoricals(self, df: pd.DataFrame) -> pd.DataFrame:
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        if cat_cols:
            logger.info(f"One-hot encoding {len(cat_cols)} categorical column(s)")
            df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
        return df

    # ── Scaling ───────────────────────────────────────────────────────────────

    def fit_scale(self, X: np.ndarray) -> np.ndarray:
        """Fit scaler on X and return transformed X."""
        return self.scaler.fit_transform(X)

    def transform_scale(self, X: np.ndarray) -> np.ndarray:
        """Transform X using the already-fitted scaler."""
        return self.scaler.transform(X)

    # ── Splitting ─────────────────────────────────────────────────────────────

    def split(self, X: np.ndarray, y: np.ndarray
              ) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
                         np.ndarray, np.ndarray, np.ndarray]:
        """Stratified Train / Val / Test split."""
        X_tv, X_test, y_tv, y_test = train_test_split(
            X, y, test_size=self.test_size,
            random_state=self.random_state, stratify=y
        )
        rel_val = self.val_size / (1 - self.test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_tv, y_tv, test_size=rel_val,
            random_state=self.random_state, stratify=y_tv
        )
        return X_train, X_val, X_test, y_train, y_val, y_test

    # ── Class imbalance ───────────────────────────────────────────────────────

    def handle_class_imbalance(self, X: np.ndarray, y: np.ndarray,
                               strategy: str = "oversample"
                               ) -> Tuple[np.ndarray, np.ndarray]:
        n_majority = int((y == 0).sum())
        n_minority = int((y == 1).sum())

        if strategy == "oversample" and n_minority < n_majority:
            X_min, y_min = resample(
                X[y == 1], y[y == 1],
                n_samples=n_majority,
                random_state=self.random_state,
            )
            X = np.vstack([X[y == 0], X_min])
            y = np.hstack([y[y == 0], y_min])

        elif strategy == "undersample" and n_majority > n_minority:
            X_maj, y_maj = resample(
                X[y == 0], y[y == 0],
                n_samples=n_minority,
                random_state=self.random_state,
            )
            X = np.vstack([X_maj, X[y == 1]])
            y = np.hstack([y_maj, y[y == 1]])

        logger.info(f"After rebalancing: {np.bincount(y).tolist()}")
        return X, y

    # ── Master pipeline ───────────────────────────────────────────────────────

    def preprocess(self, max_rows: Optional[int] = None
                   ) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
                               np.ndarray, np.ndarray, np.ndarray]:
        """
        Full preprocessing pipeline.
        Returns: X_train, X_val, X_test, y_train, y_val, y_test
        """
        df = self.load_data(max_rows=max_rows)
        df = self.clean(df)
        df, y = self.encode_labels(df)
        df = self.encode_categoricals(df)

        X = df.select_dtypes(include=[np.number]).astype(np.float32).values
        self.feature_names = df.select_dtypes(include=[np.number]).columns.tolist()
        self.n_features    = X.shape[1]

        logger.info(f"Feature count: {self.n_features}")

        X_train, X_val, X_test, y_train, y_val, y_test = self.split(X, y)

        X_train = self.fit_scale(X_train)
        X_val   = self.transform_scale(X_val)
        X_test  = self.transform_scale(X_test)

        logger.info(
            f"Preprocessing complete — "
            f"train: {X_train.shape}  val: {X_val.shape}  test: {X_test.shape}"
        )
        return X_train, X_val, X_test, y_train, y_val, y_test

    def run(self, max_rows: Optional[int] = None) -> Dict[str, Any]:
        """
        Same as preprocess() but returns a dict instead of a tuple.
        Useful for interactive / notebook use.
        """
        X_train, X_val, X_test, y_train, y_val, y_test = self.preprocess(max_rows)
        return {
            "X_train": X_train, "X_val": X_val, "X_test": X_test,
            "y_train": y_train, "y_val": y_val, "y_test": y_test,
            "feature_names": self.feature_names,
            "n_features": self.n_features,
            "class_distribution": self.class_distribution,
            "scaler": self.scaler,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic data generator (dev / demo — no real dataset needed)
# ─────────────────────────────────────────────────────────────────────────────

def generate_synthetic_data(n_samples: int = 10_000, n_features: int = 40,
                             attack_ratio: float = 0.3,
                             random_state: int = 42) -> Dict[str, Any]:
    """
    Benign traffic  ~ N(0, 1)
    Attack traffic  ~ N(2, 1.5²)
    """
    rng      = np.random.RandomState(random_state)
    n_benign = int(n_samples * (1 - attack_ratio))
    n_attack = n_samples - n_benign

    X = np.vstack([
        rng.randn(n_benign, n_features).astype(np.float32),
        (rng.randn(n_attack, n_features) * 1.5 + 2.0).astype(np.float32),
    ])
    y = np.array([0] * n_benign + [1] * n_attack, dtype=np.int32)

    idx  = rng.permutation(len(X))
    X, y = X[idx], y[idx]

    scaler  = StandardScaler()
    X_s     = scaler.fit_transform(X)
    split1  = int(0.70 * n_samples)
    split2  = int(0.85 * n_samples)

    return {
        "X_train": X_s[:split1],      "y_train": y[:split1],
        "X_val":   X_s[split1:split2],"y_val":   y[split1:split2],
        "X_test":  X_s[split2:],      "y_test":  y[split2:],
        "feature_names": [f"feature_{i:03d}" for i in range(n_features)],
        "n_features": n_features,
        "class_distribution": {"Normal": n_benign, "Attack": n_attack},
        "scaler": scaler,
    }