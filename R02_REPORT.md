# R02 spring dynamics decision

**Disable adaptive spring restoration in the next training architecture.** On three previously unused corpora, the best validation-selected full spring had higher validation cross-entropy and took longer to reach equivalent quality than a moving anchor in **all nine paired runs**. A moving anchor remains a measurable quality–cost trade-off against lightweight DFA. The next phase can use those two implementations as its efficiency baselines.

## Independent decision

The grid, endpoints and decision rule were committed before training. The R01 corpus seeds were 3109, 5471 and 7907. Confirmation used 10009, 14033 and 19001, each crossed with model seeds 11, 29 and 47. All models shared the original architecture, initialization and minibatches for each pair. Selection used deployed validation CE at 1,000 steps. Fresh corpora were constructed after the selections were written. Their document hashes were disjoint from each other and all C01/R01 documents. [Registered protocol](R02_PROTOCOL.md) · [Validation selection](results/r02/primary/selection.json) · [Fresh document fingerprints](results/r02/primary/new_corpus_manifest.json)

| Fresh confirmation, 1,000 steps | Validation CE | Test CE | First-move test accuracy | Hits at CE ≤ 0.60 | Median hit step / CPU seconds¹ | Mean update µs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Lightweight DFA | 0.55838 | 0.55292 | 99.93% | 9/9 | 725 / 0.381 | 420 |
| DFA + moving anchor | **0.54529** | **0.53847** | 99.87% | 9/9 | 725 / 0.462 | 565 |
| Full spring, selected | 0.54951 | 0.54251 | 99.87% | 9/9 | 725 / 0.679 | 835 |
| Full spring, reference | 0.59428 | 0.58620 | 98.28% | 7/9 | 925 / 0.862 | 817 |
| AdamW | 0.51736 | 0.51638 | 100% | 9/9 | 525 / 0.498 | 805 |

¹ Hit medians cover successful runs only. Elapsed time includes batch selection and preparation plus updates. The update column excludes batch preparation. Both exclude evaluation, probes, setup and warm-up. All CPU timings describe this implementation and host.

The common median of 725 steps hides a paired difference. Selected spring reached CE ≤ 0.60 later than the anchor in seven pairs and tied in two. Its mean validation CE penalty was +0.00415, +0.00380 and +0.00470 nats on the three corpora, respectively. At the historical 66% accuracy threshold, it reached the target earlier in one pair and tied in eight. It met none of the registered repeatable-advantage criteria. The moving anchor itself improved mean validation CE by 0.01309 over lightweight DFA, at a higher update cost. AdamW reached the CE threshold in fewer steps than either DFA baseline. [Paired comparisons and per-horizon tables](results/r02/tables.md) · [Complete per-seed endpoints](results/r02/per_seed.csv)

## What each coefficient changed

The diagnosis fixed the fast rate at 0.72 and changed one coefficient at a time from the R01 reference. The entries below are mean validation CE over the nine R01 corpus/model pairs. The original three equations and their order were used for every full-spring row.

| Intervention | Settings | Validation CE |
| --- | --- | ---: |
| Reference | η₀ = 0.05, βₖ = 0.02, γ = 0.10 | 0.61023 |
| Change anchor rate η₀ | 0.01 / 0.20 | 0.89730 / 0.56102 |
| Change stiffness growth βₖ | 0.002 / 0 | 0.56067 / 0.55590 |
| Change restoring strength γ | 0.01 / 0 | 0.56323 / 0.55681 |

Reducing stiffness growth tenfold produced the best configuration with active restoration. The reference anchor rate of 0.05 also won the separate moving-anchor comparison. With βₖ = 0, stiffness stayed zero and the full control matched the lightweight moving anchor. With γ = 0, stiffness continued to slow anchor consolidation, which raised mean CE slightly. The slow-anchor full-spring setting failed the historical quality gate in eight of nine runs. Across the diagnosis there were **100 PASS and 8 FAIL**. All 45 independent confirmation runs passed the historical gate, while two reference runs missed the stricter CE target at 1,000 steps. Both reached it during the prespecified extension.

## Did restoration oppose useful updates?

**Early in training, yes. Later, the effect changed sign.** At each probe the code held the weights, next training batch and validation examples fixed. It compared the validation CE after applying the local drive alone with CE after applying the drive plus the signed restoring term. This measures the immediate effect on fast weights. The independent quality comparison above measures the accumulated effect on the deployed anchors.

On fresh corpora at step 25, the reference force had 3.6% of the drive's L2 magnitude and a mean cosine of −0.354 with it. Restoration reduced the benefit of **all nine** validation-improving drives. At step 100 the ratio was 7.2%, and restoration reduced the benefit of all seven improving drives. At step 1,000 the mean cosine was +0.140 and adding restoration *lowered* immediate CE by 0.00044 on average. Reducing βₖ cut mean stiffness at step 1,000 from 0.161 to 0.0178 and the force/drive ratio from 5.9% to 0.72%. [Force magnitudes, displacement, signed alignment and CE effects at every probe](results/r02/tables.md)

At step 1,000, mean absolute displacement was similar for reference and selected spring (0.00508 and 0.00480). Their mean absolute restoring contributions differed ninefold (0.000114 and 0.0000125), while their drive magnitudes were 0.00231 and 0.00220. The reduced force primarily reflects slower stiffening.

The reference validation CE gap to the moving anchor narrowed from 0.0490 at 1,000 steps to 0.0123 at 2,000. For selected spring it narrowed from 0.00422 to 0.00053. All five methods reached 100% first-move test accuracy at 2,000 steps. Together, the probes and extension support an early loss of useful progress followed by substantial catch-up. They provide no evidence of a durable quality gain over the simpler anchor within the measured horizon.

## Implementation, memory and reproducibility

The lightweight learner keeps two fast matrices and fixed feedback. The optional moving anchor adds two matrices. Its 1,000-step unit test checks bitwise equality of fast weights and anchors against the corresponding zero-stiffness control. All nine lightweight DFA validation curves matched the archived fast-only ablation exactly, and the nine moving-anchor and nine full-spring reference curves also matched R01. [Lightweight implementation](ssenn/light_dfa.py) · [Verification tests](tests/test_r02.py)

All methods deploy 8,868 bytes. Temporary memory was measured separately as peak live CPU tensor allocation above the footprint at entry to one warmed update, including overlap while replacing weights. The measurement excludes Python objects, allocator reserves and process RSS. All three measured updates per setting returned to their entry footprint. [Allocator measurements in bytes](results/r02/memory.json)

| Learner | Additional persistent bytes | Temporary peak bytes above entry |
| --- | ---: | ---: |
| Lightweight DFA | 1,728 | 63,500 |
| DFA + moving anchor | 10,596 | 63,500 |
| Full spring, either configuration | 19,464 | 93,968 |
| AdamW | 17,744 | 50,924 |

**Software and replay.** All 23 software tests passed. All 108 diagnosis and 45 confirmation trajectories reproduced exactly on replay, including probes, metrics and first-hit steps. Absolute timings varied materially: replay median seconds to CE ≤ 0.60 were 1.052 for the anchor and 1.642 for selected spring. The anchor was faster in all nine pairs in both passes. [Replay audit](results/r02/audit.json)

R02 used one CPU thread, three independent confirmation corpora from one synthetic grammar and two fixed horizons. Decisions about wider networks, other tasks or GPU training cost require separate evidence. No GPU resources were used.

From the repository root, reproduce the software checks, measurements and tables with:

```powershell
python -m unittest discover -s tests -v
python -m benchmarks.r02 --output r02.local
python -m benchmarks.r02_tables r02.local
```
