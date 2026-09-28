"""Builds the feature matrix + labels for the representation-risk experiment, and
performs a deterministic train/test split.

No target leakage: `extract_features` (see representation_risk/features.py) never
reads `enrollment_race_distribution` -- only the label derivation in
representation_risk/labels.py does, and only on the synthetic historical records,
never on the trial being predicted for.
"""
from dataclasses import dataclass

from sklearn.model_selection import train_test_split

from representation_risk.benchmark_fixture import FIXTURE_BENCHMARK
from representation_risk.features import FEATURE_NAMES, extract_features
from representation_risk.labels import representation_gap, risk_level_from_gap
from representation_risk.synthetic_fixture import generate_historical_trials

SPLIT_SEED = 42
TEST_SIZE = 0.25


@dataclass
class Dataset:
    trial_ids: list[str]
    X: list[list[float]]
    gap: list[float]
    risk_level: list[str]

    def __len__(self) -> int:
        return len(self.trial_ids)


def build_dataset() -> Dataset:
    records = generate_historical_trials()
    trial_ids, X, gaps, levels = [], [], [], []
    for trial, observed in records:
        features = extract_features(trial)
        gap = representation_gap(observed, FIXTURE_BENCHMARK.distribution)
        trial_ids.append(trial.nct_id)
        X.append([features[name] for name in FEATURE_NAMES])
        gaps.append(gap)
        levels.append(risk_level_from_gap(gap))
    return Dataset(trial_ids=trial_ids, X=X, gap=gaps, risk_level=levels)


def _subset(dataset: Dataset, indices: list[int]) -> Dataset:
    return Dataset(
        trial_ids=[dataset.trial_ids[i] for i in indices],
        X=[dataset.X[i] for i in indices],
        gap=[dataset.gap[i] for i in indices],
        risk_level=[dataset.risk_level[i] for i in indices],
    )


def train_test_split_dataset(
    dataset: Dataset, test_size: float = TEST_SIZE, seed: int = SPLIT_SEED
) -> tuple[Dataset, Dataset]:
    indices = list(range(len(dataset)))
    try:
        train_idx, test_idx = train_test_split(
            indices, test_size=test_size, random_state=seed, stratify=dataset.risk_level
        )
    except ValueError:
        # A class with too few members to stratify -- fall back to a plain deterministic split.
        train_idx, test_idx = train_test_split(indices, test_size=test_size, random_state=seed)
    return _subset(dataset, train_idx), _subset(dataset, test_idx)
