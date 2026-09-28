# Representation Risk Experiment (EXPERIMENTAL, NOT VALIDATED)

Trained and evaluated on **synthetic fixture data only** -- see `representation_risk/README.md`.

Dataset: 60 synthetic historical trials (45 train / 15 test).

## Evaluation (no hard-coded winner)

| model | macro F1 | balanced accuracy |
| --- | --- | --- |
| logistic_regression | 0.4306 | 0.5333 |
| random_forest | 0.5556 | 0.6000 |

### Confusion matrix -- logistic_regression (labels: ['low', 'moderate', 'high'])

- [5, 0, 0]
- [4, 0, 1]
- [2, 0, 3]

### Confusion matrix -- random_forest (labels: ['low', 'moderate', 'high'])

- [5, 0, 0]
- [3, 1, 1]
- [1, 1, 3]

## Feature importance (top 5 per model)

- **logistic_regression**: num_sites=0.749, sex_inclusive=0.675, decentralized_access=0.550, num_regions=0.408, age_range_breadth_years=0.371
- **random_forest**: num_sites=0.190, num_regions=0.188, target_enrollment_log=0.099, sex_inclusive=0.098, eligibility_criteria_line_count=0.096
