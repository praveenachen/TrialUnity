# TrialUnity Retrieval Evaluation

16 benchmark cases, K=[3, 5].

## Aggregate metrics (mean over all cases)

| system | mrr | recall@3 | ndcg@3 | recall@5 | ndcg@5 |
| --- | --- | --- | --- | --- | --- |
| lexical | 1.0000 | 0.9479 | 0.8803 | 1.0000 | 0.9018 |
| dense | 1.0000 | 1.0000 | 0.8806 | 1.0000 | 0.8806 |
| hybrid | 1.0000 | 0.9688 | 0.9586 | 1.0000 | 0.9751 |

## Per-category results

| case | category | lexical nDCG@5 | dense nDCG@5 | hybrid nDCG@5 |
| --- | --- | --- | --- | --- |
| exact-condition-breast-cancer | exact_condition | 1.0000 | 1.0000 | 1.0000 |
| paraphrase-nsclc-acronym | related_terminology_low_lexical_overlap | 0.8772 | 0.9197 | 0.8772 |
| paraphrase-heart-attack | related_terminology_low_lexical_overlap | 0.8213 | 0.7579 | 0.8213 |
| intervention-preference-immunotherapy | treatment_preference | 0.8213 | 0.7579 | 1.0000 |
| location-preference-toronto | location_preference | 1.0000 | 1.0000 | 1.0000 |
| phase-preference-phase3 | phase_preference | 0.8213 | 0.7579 | 1.0000 |
| eligibility-age-restricted | eligibility_incompatibility | 1.0000 | 0.9514 | 0.9514 |
| distractor-heavy-diabetes | distractor | 0.7967 | 1.0000 | 1.0000 |
| acronym-egfr | related_terminology_low_lexical_overlap | 0.9143 | 1.0000 | 1.0000 |
| combined-condition-location-phase | phase_preference | 1.0000 | 0.8213 | 1.0000 |
| rare-disease-low-overlap | related_terminology_low_lexical_overlap | 1.0000 | 1.0000 | 1.0000 |
| intervention-preference-drugname | treatment_preference | 0.8213 | 0.8213 | 1.0000 |
| location-remote-decentralized | location_preference | 1.0000 | 0.7967 | 1.0000 |
| phase-preference-early-phase1 | phase_preference | 0.7967 | 0.7967 | 1.0000 |
| eligibility-sex-restricted | eligibility_incompatibility | 1.0000 | 0.9514 | 0.9514 |
| multi-signal-alignment | treatment_preference | 0.7579 | 0.7579 | 1.0000 |

## ESR sanity checks (separate from IR quality -- see evaluation/README.md)

- Deterministic: True
- Missing evidence lowers coverage, not score: True / True
- Does not change clinical retrieval ordering: True
