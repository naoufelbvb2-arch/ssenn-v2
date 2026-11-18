# R01 measurement tables

Quality and cost are arithmetic means across corpus/model pairs. Time to target is the median among observed hits. A run can hit transiently and finish below target.

| Corpora | Method | Rate × | Final PASS | Ever hit | Hit step | Val CE | Test acc % | Test CE | Train s |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| historical | ablation | 1 | 0/3 | 1 | 700 | 0.7252 | 65.55 | 0.7333 | 0.345 |
| historical | adamw | 1 | 3/3 | 3 | 250 | 0.5122 | 74.15 | 0.5179 | 0.340 |
| historical | spring | 1 | 1/3 | 1 | 975 | 0.7426 | 65.82 | 0.7525 | 0.340 |
| new | ablation | 1 | 0/9 | 5 | 700 | 0.7537 | 65.41 | 0.7288 | 0.366 |
| new | ablation | 4 | 9/9 | 9 | 225 | 0.5662 | 74.00 | 0.5542 | 0.360 |
| new | adamw | 1 | 9/9 | 9 | 200 | 0.5260 | 73.65 | 0.5196 | 0.382 |
| new | anchor_only | 1 | 2/9 | 6 | 612.5 | 0.7529 | 65.28 | 0.7284 | 0.368 |
| new | anchor_only | 4 | 9/9 | 9 | 275 | 0.5559 | 73.42 | 0.5481 | 0.390 |
| new | restore_only | 1 | 0/9 | 0 | — | 2.1186 | 25.04 | 2.1199 | 0.374 |
| new | restore_only | 4 | 0/9 | 0 | — | 1.9273 | 43.20 | 1.9275 | 0.338 |
| new | spring | 1 | 0/9 | 6 | 837.5 | 0.7738 | 65.08 | 0.7479 | 0.342 |
| new | spring | 4 | 9/9 | 9 | 425 | 0.6102 | 73.07 | 0.5979 | 0.349 |

## Per-corpus outcomes for the original methods

| Corpus | Method | Rate × | Final PASS / 3 | Val CE | Direction test acc % |
| --- | --- | --- | --- | --- | --- |
| 3109 | ablation | 1.0 | 0 | 0.7467 | 59.92 |
| 5471 | ablation | 1.0 | 0 | 0.7416 | 57.54 |
| 7907 | ablation | 1.0 | 0 | 0.7727 | 54.56 |
| 3109 | ablation | 4.0 | 3 | 0.5606 | 100.00 |
| 5471 | ablation | 4.0 | 3 | 0.5570 | 100.00 |
| 7907 | ablation | 4.0 | 3 | 0.5810 | 100.00 |
| 3109 | adamw | 1.0 | 3 | 0.5214 | 100.00 |
| 5471 | adamw | 1.0 | 3 | 0.5228 | 100.00 |
| 7907 | adamw | 1.0 | 3 | 0.5338 | 100.00 |
| 3109 | spring | 1.0 | 0 | 0.7670 | 56.15 |
| 5471 | spring | 1.0 | 0 | 0.7627 | 57.14 |
| 7907 | spring | 1.0 | 0 | 0.7918 | 54.56 |
| 3109 | spring | 4.0 | 3 | 0.6057 | 94.84 |
| 5471 | spring | 4.0 | 3 | 0.6078 | 99.21 |
| 7907 | spring | 4.0 | 3 | 0.6172 | 99.01 |

## Paired spring component contrasts

Each difference is left minus right. A negative CE difference favors the left state. Counts indicate lower/equal/higher CE. The last three columns average over model seeds within each corpus.

| Rate × | Test comparison | Δ CE | Lower/equal/higher | Direction Δ pp | 3109 | 5471 | 7907 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1.0 | anchor_only deployed − ablation fast | -0.0003 | 4/0/5 | -0.93 | -0.0001 | +0.0015 | -0.0024 |
| 1.0 | anchor_only fast − ablation fast | +0.0000 | 0/9/0 | +0.00 | +0.0000 | +0.0000 | +0.0000 |
| 1.0 | restore_only fast − ablation fast | +1.3911 | 0/0/9 | -33.60 | +1.4066 | +1.3891 | +1.3776 |
| 1.0 | spring fast − anchor_only fast | +0.0187 | 0/0/9 | -1.59 | +0.0176 | +0.0195 | +0.0189 |
| 1.0 | spring deployed − anchor_only deployed | +0.0195 | 0/0/9 | -0.46 | +0.0186 | +0.0202 | +0.0197 |
| 1.0 | spring deployed − spring fast | +0.0005 | 4/0/5 | +0.20 | +0.0008 | +0.0021 | -0.0016 |
| 1.0 | spring deployed − ablation fast | +0.0191 | 0/0/9 | -1.39 | +0.0184 | +0.0216 | +0.0173 |
| 4.0 | anchor_only deployed − ablation fast | -0.0061 | 8/0/1 | -0.26 | -0.0051 | -0.0002 | -0.0129 |
| 4.0 | anchor_only fast − ablation fast | +0.0000 | 0/9/0 | +0.00 | +0.0000 | +0.0000 | +0.0000 |
| 4.0 | restore_only fast − ablation fast | +1.3734 | 0/0/9 | -59.52 | +1.3835 | +1.3758 | +1.3608 |
| 4.0 | spring fast − anchor_only fast | +0.0479 | 0/0/9 | -3.90 | +0.0474 | +0.0494 | +0.0468 |
| 4.0 | spring deployed − anchor_only deployed | +0.0498 | 0/0/9 | -2.05 | +0.0494 | +0.0514 | +0.0487 |
| 4.0 | spring deployed − spring fast | -0.0041 | 7/0/2 | +1.59 | -0.0031 | +0.0018 | -0.0110 |
| 4.0 | spring deployed − ablation fast | +0.0438 | 0/0/9 | -2.31 | +0.0443 | +0.0512 | +0.0358 |

## All clause positions

Each cell gives mean accuracy and cross-entropy in nats. Positions are zero-based. Historical means cover three model seeds. New-corpus means cover nine paired runs.

| Corpora | Method | Rate × | Split | Readout | 0 direction | 1 start | 2 first move | 3 second move | 4 third move |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| historical | ablation | 1 | validation | deployed | 47.02% / 0.769 | 18.06% / 2.022 | 51.98% / 0.753 | 100.00% / 0.095 | 100.00% / 0.080 |
| historical | ablation | 1 | test | deployed | 50.79% / 0.742 | 19.64% / 2.105 | 52.38% / 0.744 | 100.00% / 0.092 | 100.00% / 0.078 |
| historical | ablation | 1 | validation | fast | 47.02% / 0.769 | 18.06% / 2.022 | 51.98% / 0.753 | 100.00% / 0.095 | 100.00% / 0.080 |
| historical | ablation | 1 | test | fast | 50.79% / 0.742 | 19.64% / 2.105 | 52.38% / 0.744 | 100.00% / 0.092 | 100.00% / 0.078 |
| historical | adamw | 1 | validation | deployed | 45.04% / 0.749 | 18.65% / 1.837 | 100.00% / 0.032 | 100.00% / 0.008 | 100.00% / 0.007 |
| historical | adamw | 1 | test | deployed | 50.20% / 0.706 | 16.87% / 1.908 | 100.00% / 0.033 | 100.00% / 0.008 | 100.00% / 0.007 |
| historical | adamw | 1 | validation | fast | 45.04% / 0.749 | 18.65% / 1.837 | 100.00% / 0.032 | 100.00% / 0.008 | 100.00% / 0.007 |
| historical | adamw | 1 | test | fast | 50.20% / 0.706 | 16.87% / 1.908 | 100.00% / 0.033 | 100.00% / 0.008 | 100.00% / 0.007 |
| historical | spring | 1 | validation | deployed | 47.22% / 0.789 | 19.05% / 2.022 | 51.39% / 0.784 | 100.00% / 0.119 | 100.00% / 0.092 |
| historical | spring | 1 | test | deployed | 50.99% / 0.767 | 19.25% / 2.110 | 53.97% / 0.774 | 100.00% / 0.117 | 100.00% / 0.090 |
| historical | spring | 1 | validation | fast | 47.02% / 0.789 | 18.25% / 2.031 | 49.60% / 0.782 | 100.00% / 0.117 | 100.00% / 0.092 |
| historical | spring | 1 | test | fast | 50.79% / 0.764 | 19.44% / 2.112 | 50.60% / 0.773 | 100.00% / 0.115 | 100.00% / 0.089 |
| new | ablation | 1 | validation | deployed | 47.82% / 0.751 | 14.02% / 2.190 | 58.27% / 0.728 | 100.00% / 0.103 | 100.00% / 0.092 |
| new | ablation | 1 | test | deployed | 47.69% / 0.753 | 17.06% / 2.056 | 57.34% / 0.731 | 100.00% / 0.102 | 100.00% / 0.093 |
| new | ablation | 1 | validation | fast | 47.82% / 0.751 | 14.02% / 2.190 | 58.27% / 0.728 | 100.00% / 0.103 | 100.00% / 0.092 |
| new | ablation | 1 | test | fast | 47.69% / 0.753 | 17.06% / 2.056 | 57.34% / 0.731 | 100.00% / 0.102 | 100.00% / 0.093 |
| new | ablation | 4 | validation | deployed | 49.14% / 0.737 | 15.54% / 1.969 | 100.00% / 0.150 | 100.00% / 0.024 | 100.00% / 0.028 |
| new | ablation | 4 | test | deployed | 49.54% / 0.729 | 16.73% / 1.914 | 100.00% / 0.151 | 100.00% / 0.024 | 100.00% / 0.028 |
| new | ablation | 4 | validation | fast | 49.14% / 0.737 | 15.54% / 1.969 | 100.00% / 0.150 | 100.00% / 0.024 | 100.00% / 0.028 |
| new | ablation | 4 | test | fast | 49.54% / 0.729 | 16.73% / 1.914 | 100.00% / 0.151 | 100.00% / 0.024 | 100.00% / 0.028 |
| new | adamw | 1 | validation | deployed | 48.08% / 0.723 | 15.08% / 1.929 | 100.00% / 0.038 | 100.00% / 0.007 | 100.00% / 0.007 |
| new | adamw | 1 | test | deployed | 48.02% / 0.725 | 16.47% / 1.893 | 100.00% / 0.038 | 100.00% / 0.007 | 100.00% / 0.007 |
| new | adamw | 1 | validation | fast | 48.08% / 0.723 | 15.08% / 1.929 | 100.00% / 0.038 | 100.00% / 0.007 | 100.00% / 0.007 |
| new | adamw | 1 | test | fast | 48.02% / 0.725 | 16.47% / 1.893 | 100.00% / 0.038 | 100.00% / 0.007 | 100.00% / 0.007 |
| new | anchor_only | 1 | validation | deployed | 48.08% / 0.750 | 13.69% / 2.187 | 58.73% / 0.728 | 100.00% / 0.104 | 100.00% / 0.091 |
| new | anchor_only | 1 | test | deployed | 47.55% / 0.755 | 17.46% / 2.052 | 56.42% / 0.730 | 100.00% / 0.103 | 100.00% / 0.092 |
| new | anchor_only | 1 | validation | fast | 47.82% / 0.751 | 14.02% / 2.190 | 58.27% / 0.728 | 100.00% / 0.103 | 100.00% / 0.092 |
| new | anchor_only | 1 | test | fast | 47.69% / 0.753 | 17.06% / 2.056 | 57.34% / 0.731 | 100.00% / 0.102 | 100.00% / 0.093 |
| new | anchor_only | 4 | validation | deployed | 48.94% / 0.717 | 14.62% / 1.934 | 99.74% / 0.152 | 100.00% / 0.024 | 100.00% / 0.028 |
| new | anchor_only | 4 | test | deployed | 47.69% / 0.723 | 15.87% / 1.885 | 99.74% / 0.155 | 100.00% / 0.024 | 100.00% / 0.028 |
| new | anchor_only | 4 | validation | fast | 49.14% / 0.737 | 15.54% / 1.969 | 100.00% / 0.150 | 100.00% / 0.024 | 100.00% / 0.028 |
| new | anchor_only | 4 | test | fast | 49.54% / 0.729 | 16.73% / 1.914 | 100.00% / 0.151 | 100.00% / 0.024 | 100.00% / 0.028 |
| new | restore_only | 1 | validation | deployed | 0.00% / 2.282 | 15.48% / 2.133 | 24.67% / 2.099 | 34.66% / 2.070 | 47.05% / 2.022 |
| new | restore_only | 1 | test | deployed | 0.00% / 2.282 | 16.60% / 2.134 | 23.74% / 2.102 | 36.51% / 2.064 | 45.43% / 2.031 |
| new | restore_only | 1 | validation | fast | 0.00% / 2.282 | 15.48% / 2.133 | 24.67% / 2.099 | 34.66% / 2.070 | 47.05% / 2.022 |
| new | restore_only | 1 | test | fast | 0.00% / 2.282 | 16.60% / 2.134 | 23.74% / 2.102 | 36.51% / 2.064 | 45.43% / 2.031 |
| new | restore_only | 4 | validation | deployed | 0.33% / 2.264 | 15.21% / 2.111 | 39.55% / 1.907 | 63.89% / 1.778 | 87.73% / 1.620 |
| new | restore_only | 4 | test | deployed | 0.46% / 2.261 | 17.39% / 2.102 | 40.48% / 1.908 | 63.96% / 1.774 | 87.38% / 1.634 |
| new | restore_only | 4 | validation | fast | 0.33% / 2.264 | 15.21% / 2.111 | 39.55% / 1.907 | 63.89% / 1.778 | 87.73% / 1.620 |
| new | restore_only | 4 | test | fast | 0.46% / 2.261 | 17.39% / 2.102 | 40.48% / 1.908 | 63.96% / 1.774 | 87.38% / 1.634 |
| new | spring | 1 | validation | deployed | 48.68% / 0.774 | 13.69% / 2.199 | 56.48% / 0.761 | 100.00% / 0.129 | 100.00% / 0.103 |
| new | spring | 1 | test | deployed | 47.42% / 0.779 | 17.06% / 2.065 | 55.95% / 0.757 | 100.00% / 0.127 | 100.00% / 0.104 |
| new | spring | 1 | validation | fast | 48.28% / 0.773 | 14.02% / 2.202 | 55.62% / 0.759 | 100.00% / 0.127 | 100.00% / 0.102 |
| new | spring | 1 | test | fast | 47.62% / 0.776 | 17.26% / 2.068 | 55.75% / 0.757 | 100.00% / 0.125 | 100.00% / 0.104 |
| new | spring | 4 | validation | deployed | 48.88% / 0.721 | 14.48% / 1.991 | 97.42% / 0.334 | 100.00% / 0.040 | 100.00% / 0.046 |
| new | spring | 4 | test | deployed | 47.22% / 0.726 | 16.60% / 1.916 | 97.69% / 0.339 | 100.00% / 0.040 | 100.00% / 0.046 |
| new | spring | 4 | validation | fast | 49.07% / 0.740 | 15.48% / 2.021 | 96.03% / 0.326 | 100.00% / 0.039 | 100.00% / 0.046 |
| new | spring | 4 | test | fast | 49.54% / 0.732 | 16.14% / 1.941 | 96.10% / 0.330 | 100.00% / 0.040 | 100.00% / 0.047 |
