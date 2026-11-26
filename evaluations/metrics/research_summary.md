# Main experiment findings

All values and gains below are measured on the untouched chronological test set. They describe association, not causation.

Precision, recall, and F1 are computed for the minority fraud class only (`isFraud = 1`; scikit-learn `average="binary"`, `pos_label=1`). Majority-class, macro-averaged, and weighted-averaged scores are never reported, because the legitimate class would mask fraud-class performance at a 0.13% fraud rate.

## Sampled-graph findings

1. **Do sampled custom-graph features improve fraud detection?** Mean sampled-graph-minus-tabular PR-AUC across model families: -0.012915 (0/4 models improved).
2. **Does recent activity add signal beyond sampled graph structure?** Mean sampled-temporal-minus-sampled-graph PR-AUC: -0.004162 (3/4 models improved).
3. **Which model benefits most from custom graph and temporal features?** MLP has the largest sampled-temporal-minus-tabular PR-AUC gain (+0.042258).
4. **Do graph features primarily improve fraud-class precision, recall, or both?** For sampled-temporal versus sampled-tabular predictions at their validation-selected thresholds, mean fraud precision change is -0.001108 and mean fraud recall change is -0.040335; absolute changes are larger for recall more than precision.
5. **How should these results be interpreted?** They describe this deterministic sampled training graph and full chronological validation/test periods; see the full-graph findings for estimates from every training edge.

## Full-graph findings

The full graph is built from all 6,082,007 training transactions (8,669,206 accounts); every model is fit on all training-period fraud rows plus a seeded subsample of legitimate training rows (the same rows for every feature set). Precision, recall, and F1 are fraud-class (minority-class) scores.

6. **Do full-network structural features move the needle for the fraud class?** Mean full-graph-minus-tabular change: PR-AUC -0.000665, fraud precision +0.006448, fraud recall -0.004593, fraud F1 +0.000820 (1/4 models improved fraud F1).
7. **Do full-graph features add fraud-class signal beyond 24-hour activity?** Mean full-temporal-minus-temporal-only change: PR-AUC -0.007404, fraud precision -0.032712, fraud recall +0.005192, fraud F1 -0.016681 (0/4 models improved fraud F1).
8. **Does scaling beyond the sampled graph change the conclusion?** Sampled graph-minus-tabular mean change: PR-AUC -0.012915, fraud precision +0.009812, fraud recall -0.027955, fraud F1 -0.015939 (2/4 models improved fraud F1). Full graph-minus-tabular mean change: PR-AUC -0.000665, fraud precision +0.006448, fraud recall -0.004593, fraud F1 +0.000820 (1/4 models improved fraud F1). The two designs also differ in training-set size, so compare each gain with its own baseline.

Per-model results and ablation gains are in `main_results.csv`, `ablation_gains.csv`, and `fraud_class_gains.csv`. No outcome is assumed in advance.
