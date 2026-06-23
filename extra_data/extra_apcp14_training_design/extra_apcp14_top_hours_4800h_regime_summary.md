# APCP1.4 Top-rainfall 4800h Extra Samples Rainfall Regime Summary

## 1. Data source

This summary analyzes the extra training manifest:

```text
/home/huanghui/data/ParFlow-transformer/extra_data/extra_apcp14_training_design/extra_apcp14_top_hours_4800h_2020_2021.csv
```

The manifest contains selected APCP1.4 extra training samples from 2020 and 2021. Each row gives one 24h training sample start time (`t0`). These samples are selected from high-rainfall periods and are intended to supplement the normal training data.

## 2. Sample count by year

The manifest contains 800 extra 24h samples in total.

| Year | Sample count |
| --- | ---: |
| 2020 | 432 |
| 2021 | 368 |
| Total | 800 |

All 800 samples can be matched to the APCP1.4 24h stride-6 rainfall-regime statistics.

## 3. Rainfall-regime distribution

### 3.1 Rainfall intensity regimes

The rainfall-intensity regimes are mutually exclusive.

| Regime | Sample count | Percent |
| --- | ---: | ---: |
| dry | 0 | 0.00% |
| light | 296 | 37.00% |
| moderate | 390 | 48.75% |
| heavy | 114 | 14.25% |

The selected extra samples are mainly concentrated in moderate and heavy rainfall conditions:

```text
moderate + heavy = 504 samples, accounting for 63.00% of all selected extra samples.
```

No dry samples are included, indicating that this extra dataset is strongly biased toward wet and high-rainfall conditions.

### 3.2 Process-based rainfall regimes

The following regimes can overlap, so their percentages do not sum to 100%.

| Regime | Sample count | Percent |
| --- | ---: | ---: |
| strong_6h | 207 | 25.87% |
| persistent_wet | 704 | 88.00% |
| dry_to_wet | 63 | 7.88% |
| wet_to_dry | 66 | 8.25% |

The most dominant process-based regime is `persistent_wet`, which accounts for 88.00% of the selected samples. Therefore, the top-rainfall extra samples mainly represent sustained wet conditions rather than only short-duration extreme rainfall events.

## 4. Rainfall statistics of selected samples

| Metric | Value |
| --- | ---: |
| R24_total mean | 15.63 mm |
| R24_total std | 10.87 mm |
| R24_total min | 5.12 mm |
| R24_total median | 12.17 mm |
| R24_total max | 78.95 mm |
| R6_max mean | 8.42 mm |
| rain_hours mean | 18.12 h |

These statistics show that the selected extra samples have three major characteristics:

```text
1. Higher 24h accumulated rainfall;
2. Stronger rainfall variability;
3. Longer rainfall duration within the 24h sample window.
```

## 5. Comparison with normal training data

The normal training data refers to APCP1 samples from 2020 and 2021. The selected extra data refers to APCP1.4 top-rainfall 4800h samples.

| Regime | Normal train 2020+2021 | Extra top4800h |
| --- | ---: | ---: |
| dry | 49.62% | 0.00% |
| light | 38.76% | 37.00% |
| moderate | 9.90% | 48.75% |
| heavy | 1.71% | 14.25% |
| strong_6h | 3.56% | 25.87% |
| persistent_wet | 23.78% | 88.00% |
| dry_to_wet | 1.71% | 7.88% |
| wet_to_dry | 2.09% | 8.25% |

Compared with the normal training set, the selected APCP1.4 extra samples substantially increase the representation of hydrologically active regimes:

```text
moderate rainfall: 9.90% -> 48.75%
heavy rainfall: 1.71% -> 14.25%
strong_6h: 3.56% -> 25.87%
persistent_wet: 23.78% -> 88.00%
dry_to_wet: 1.71% -> 7.88%
wet_to_dry: 2.09% -> 8.25%
```

At the same time, dry samples are removed from the extra set:

```text
dry: 49.62% -> 0.00%
```

## 6. Main interpretation

The APCP1.4 top-rainfall 4800h extra dataset is not a balanced supplement to all rainfall regimes. Instead, it is a targeted high-rainfall supplement enriched in hydrologically active conditions.

Its main contribution is to increase the number of samples representing:

```text
1. Moderate rainfall;
2. Heavy rainfall;
3. Strong 6h rainfall;
4. Persistent wet conditions;
5. Dry-to-wet and wet-to-dry transition processes.
```

This makes the dataset suitable for testing whether additional high-rainfall and wet-condition samples can improve model robustness under hydrologically active conditions.

However, because the selected samples are dominated by persistent_wet and moderate/heavy rainfall regimes, this dataset may also shift the training distribution away from dry or weak-response conditions. Therefore, its effect should be evaluated not only by annual-average metrics, but also by rainfall-regime-specific WTD error metrics.

## 7. Suggested paper-level statement

The following statement can be used in the paper or presentation:

```text
The selected APCP1.4 top-rainfall samples are strongly enriched in hydrologically active regimes. Among the 800 selected 24h samples, 48.75% are moderate rainfall, 14.25% are heavy rainfall, 25.87% satisfy the strong_6h condition, and 88.00% correspond to persistent wet conditions, while no dry samples are included. Compared with the normal 2020-2021 training set, this extra dataset substantially increases the representation of wet, strong-rainfall, and transition regimes. Therefore, it is suitable for testing whether targeted high-rainfall augmentation improves surrogate-model robustness under active hydrologic conditions.
```

Chinese version:

```text
这批 APCP1.4 top-rainfall 额外样本明显富集于水文活跃情景。800 个样本中，moderate 占 48.75%，heavy 占 14.25%，strong_6h 占 25.87%，persistent_wet 占 88.00%，而 dry 样本为 0。相比普通 2020-2021 训练集，它显著提高了中强降雨、持续湿润和干湿转换样本比例，因此更适合用于检验强降雨/湿润条件下的模型鲁棒性是否能通过定向样本补充得到改善。
```
