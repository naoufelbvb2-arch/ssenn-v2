# R02 measurement tables

All quality and force summaries use the primary pass. Costs describe unprofiled single-thread CPU runs unless labeled allocator measurements. All 27 matched historical validation trajectories reproduced exactly, including the full learning curve for lightweight DFA versus the R01 ablation.

## R01 diagnosis

| Setting | Steps | Final PASS | Val CE | Val first move % | Test CE | CE ≤ .60 hits | Median hit step¹ | Median hit s¹ | Mean update µs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| light | 1000 | 9/9 | 0.56618 | 100.00 | — | 9/9 | 825 | 0.2426 | 294.5 |
| anchor_0.05 | 1000 | 9/9 | 0.55590 | 99.74 | — | 9/9 | 775 | 0.2881 | 395.1 |
| anchor_0.01 | 1000 | 9/9 | 0.56728 | 99.54 | — | 9/9 | 875 | 0.3529 | 395.6 |
| anchor_0.2 | 1000 | 9/9 | 0.55759 | 99.74 | — | 9/9 | 775 | 0.3037 | 384.3 |
| reference | 1000 | 9/9 | 0.61023 | 97.42 | — | 2/9 | 1000.0 | 0.5052 | 540.7 |
| slow_anchor | 1000 | 1/9 | 0.89730 | 54.83 | — | 0/9 | — | — | 519.1 |
| fast_anchor | 1000 | 9/9 | 0.56102 | 99.74 | — | 9/9 | 800 | 0.4405 | 527.2 |
| slow_growth | 1000 | 9/9 | 0.56067 | 99.54 | — | 9/9 | 800 | 0.4096 | 513.9 |
| no_growth | 1000 | 9/9 | 0.55590 | 99.74 | — | 9/9 | 775 | 0.4157 | 514.2 |
| weak_restore | 1000 | 9/9 | 0.56323 | 99.47 | — | 9/9 | 825 | 0.4450 | 577.9 |
| no_restore | 1000 | 9/9 | 0.55681 | 99.54 | — | 9/9 | 800 | 0.4364 | 537.5 |
| adamw | 1000 | 9/9 | 0.52602 | 100.00 | — | 9/9 | 550 | 0.3108 | 552.7 |

## Fresh confirmation

| Setting | Steps | Final PASS | Val CE | Val first move % | Test CE | CE ≤ .60 hits | Median hit step¹ | Median hit s¹ | Mean update µs |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| light | 1000 | 9/9 | 0.55838 | 99.93 | 0.55292 | 9/9 | 725 | 0.3809 | 419.8 |
| light | 2000 | 9/9 | 0.52167 | 100.00 | 0.52174 | 9/9 | 725 | 0.3809 | 413.3 |
| anchor_0.05 | 1000 | 9/9 | 0.54529 | 99.80 | 0.53847 | 9/9 | 725 | 0.4616 | 565.0 |
| anchor_0.05 | 2000 | 9/9 | 0.50921 | 100.00 | 0.50919 | 9/9 | 725 | 0.4616 | 575.5 |
| reference | 1000 | 9/9 | 0.59428 | 98.21 | 0.58620 | 7/9 | 925 | 0.8622 | 817.5 |
| reference | 2000 | 9/9 | 0.52147 | 100.00 | 0.52012 | 9/9 | 950 | 0.8779 | 840.6 |
| slow_growth | 1000 | 9/9 | 0.54950 | 99.80 | 0.54251 | 9/9 | 725 | 0.6785 | 835.4 |
| slow_growth | 2000 | 9/9 | 0.50975 | 100.00 | 0.50960 | 9/9 | 725 | 0.6785 | 826.4 |
| adamw | 1000 | 9/9 | 0.51736 | 100.00 | 0.51638 | 9/9 | 525 | 0.4979 | 805.0 |
| adamw | 2000 | 9/9 | 0.50907 | 100.00 | 0.51397 | 9/9 | 525 | 0.4979 | 818.4 |

¹ Hit medians are conditional on reaching the target by that horizon. The historical accuracy gate and cost of each hit appear in the per-seed CSV.

## Locked spring minus moving anchor at 1,000 steps

A method missing a target loses to one that hits. Both missing is a tie. Numerical differences use only pairs with both values, with coverage shown per corpus.

| Target | Measure | Spring wins | Ties | Losses | Mean paired Δ | 10009 | 14033 | 19001 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| validation | ce | 0 | 0 | 9 | +0.00422 | +0.00415 (3/3) | +0.00380 (3/3) | +0.00470 (3/3) |
| ce | step | 0 | 2 | 7 | +19.44444 | +16.66667 (3/3) | +25.00000 (3/3) | +16.66667 (3/3) |
| ce | seconds | 0 | 0 | 9 | +0.23208 | +0.26279 (3/3) | +0.25207 (3/3) | +0.18139 (3/3) |
| accuracy | step | 1 | 8 | 0 | -5.55556 | +0.00000 (3/3) | -16.66667 (3/3) | +0.00000 (3/3) |

## Force evolution on fresh corpora

Magnitudes are mean absolute coefficient updates. Ratios and cosines use flattened L2 vectors. Positive Δ CE means restoration worsened fast-weight validation CE versus the same drive-only step. The last column restricts this comparison to drive-only steps that improved CE from the current state.

| Setting | Step | Mean k | Displacement | Elastic magnitude | Drive magnitude | Ratio | Cosine | Δ CE | Helpful drive attenuated |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| reference | 0 | 0 | 0 | 0 | 0.0058517 | 0 | — | +0.000000 | 0/9 |
| reference | 25 | 0.013991 | 0.036082 | 9.3557e-05 | 0.0043674 | 0.036047 | -0.35349 | +0.001031 | 9/9 |
| reference | 100 | 0.053109 | 0.016814 | 0.00015125 | 0.0033123 | 0.071789 | -0.15738 | +0.001067 | 7/7 |
| reference | 250 | 0.081708 | 0.007246 | 8.3186e-05 | 0.0029472 | 0.038104 | 0.049829 | +0.000032 | 5/6 |
| reference | 500 | 0.11167 | 0.0061054 | 8.9703e-05 | 0.0026896 | 0.039303 | 0.13331 | -0.000200 | 0/5 |
| reference | 750 | 0.13805 | 0.0056585 | 0.00010533 | 0.00278 | 0.045977 | 0.090049 | -0.000185 | 1/4 |
| reference | 1000 | 0.16104 | 0.0050828 | 0.00011417 | 0.002315 | 0.059047 | 0.13958 | -0.000444 | 0/5 |
| reference | 1500 | 0.19895 | 0.0044694 | 0.00012451 | 0.0025499 | 0.053711 | 0.2148 | -0.000391 | 2/4 |
| reference | 2000 | 0.23084 | 0.0041857 | 0.00013847 | 0.0025001 | 0.062001 | 0.16388 | -0.000511 | 0/4 |
| slow_growth | 0 | 0 | 0 | 0 | 0.0058517 | 0 | — | +0.000000 | 0/9 |
| slow_growth | 25 | 0.0014176 | 0.03617 | 9.5928e-06 | 0.0043485 | 0.0038356 | -0.33897 | +0.000104 | 9/9 |
| slow_growth | 100 | 0.0055329 | 0.016237 | 1.5721e-05 | 0.0032484 | 0.008171 | -0.12352 | +0.000095 | 6/6 |
| slow_growth | 250 | 0.0085358 | 0.006946 | 8.3119e-06 | 0.0029373 | 0.0039966 | 0.074727 | -0.000004 | 1/6 |
| slow_growth | 500 | 0.011928 | 0.00621 | 1.0071e-05 | 0.0026223 | 0.0048142 | 0.12276 | -0.000021 | 0/5 |
| slow_growth | 750 | 0.015058 | 0.0055014 | 1.1749e-05 | 0.0026742 | 0.0058455 | 0.091978 | -0.000020 | 1/4 |
| slow_growth | 1000 | 0.017783 | 0.0047974 | 1.25e-05 | 0.0022004 | 0.0071915 | 0.15611 | -0.000060 | 0/6 |
| slow_growth | 1500 | 0.022541 | 0.0043074 | 1.4477e-05 | 0.0025256 | 0.0066399 | 0.24419 | -0.000059 | 2/4 |
| slow_growth | 2000 | 0.026987 | 0.0042609 | 1.7984e-05 | 0.002495 | 0.0088136 | 0.15532 | -0.000082 | 0/3 |

## Memory per warmed CPU update

| Setting | Deployed bytes | Training-only bytes | Peak above entry (min–max bytes) |
| --- | --- | --- | --- |
| light | 8868 | 1728 | 63500–63500 |
| anchor_0.05 | 8868 | 10596 | 63500–63500 |
| reference | 8868 | 19464 | 93968–93968 |
| slow_growth | 8868 | 19464 | 93968–93968 |
| adamw | 8868 | 17744 | 50924–50924 |
