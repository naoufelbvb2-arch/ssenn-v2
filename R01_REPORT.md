# C01-R01 learning failure diagnosis

**Recommendation: improve spring dynamics.** Within the fixed 1,000-step T02 budget, increasing the fast learning rate resolved most of the direction error. Fast-weight DFA alone then predicted every held-out direction-dependent move correctly. Adding stiffness and restoring force increased test cross-entropy for every matched corpus/model pair at both rates. A moving anchor alone offered a small readout benefit. These results support investigating the restoring term before spending GPU time on C02.

## Independent replication and convergence

Corpus seeds 3109, 5471 and 7907 were each crossed with model seeds 11, 29 and 47. Each learner received the same initial forward weights, batches, 48-unit architecture and evaluation schedule for its pair. Across the new and historical corpora, all 576 document hashes were distinct. Corpus generation has a separate RNG from initialization, feedback and batch selection. The manifest records all four fingerprints. [Protocol committed before experimentation](R01_PROTOCOL.md) · [Seed and data manifest](results/r01/manifest.json)

Each original method received 27 search runs: three learning rates on the same nine pairs. Final mean validation CE selected 0.72 for both DFA learners and 0.005 for AdamW. At the half, original and quadruple rates, validation CE was 0.877/0.774/0.610 for spring, 0.834/0.754/0.566 for ablation and 0.604/0.526/0.535 for AdamW. Selections were saved before test scoring. The highest DFA rate was the edge of the grid. [Validation-only decisions](results/r01/primary/selection.json)

| New corpora, deployed readout | Final PASS | Ever hit 66% | Median first-hit step / seconds¹ | Mean seconds for 1,000 steps | Final val CE | Test CE | First-move test accuracy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Spring, original | 0/9 | 6/9 | 837.5 / 0.286 | 0.342 | 0.774 | 0.748 | 55.95% |
| DFA ablation, original | 0/9 | 5/9 | 700 / 0.242 | 0.366 | 0.754 | 0.729 | 57.34% |
| Spring, selected | 9/9 | 9/9 | 425 / 0.151 | 0.349 | 0.610 | 0.598 | 97.69% |
| DFA ablation, selected | 9/9 | 9/9 | 225 / 0.080 | 0.360 | 0.566 | 0.554 | 100% |
| AdamW, original and selected | 9/9 | 9/9 | 200 / 0.077 | 0.382 | 0.526 | 0.520 | 100% |

¹ Medians use only runs that reached the target. Accuracy was sampled every 25 steps. A final PASS requires at least 66% validation accuracy and a CE improvement from initialization. All nine original-rate spring replications finished **FAIL**, including the six transient hits. These CPU timings include batch sampling and updates. Evaluation, probes, setup and warm-up are excluded.

## Where prediction failed

The historical controls reproduced the C01 first-move test accuracy of 53.97% for spring and 100% for AdamW. The table gives mean test **accuracy / CE in nats** for each position on the three new corpora. The first two targets are randomly drawn. The final three are deterministic given the available context.

| Learner | Direction | Start symbol | First move | Second move | Third move |
| --- | --- | --- | --- | --- | --- |
| Spring, original | 47.42% / .779 | 17.06% / 2.065 | 55.95% / .757 | 100% / .127 | 100% / .104 |
| DFA ablation, original | 47.69% / .753 | 17.06% / 2.056 | 57.34% / .731 | 100% / .102 | 100% / .093 |
| Spring, selected | 47.22% / .726 | 16.60% / 1.916 | 97.69% / .339 | 100% / .040 | 100% / .046 |
| DFA ablation, selected | 49.54% / .729 | 16.73% / 1.914 | 100% / .151 | 100% / .024 | 100% / .028 |
| AdamW, selected | 48.02% / .725 | 16.47% / 1.893 | 100% / .038 | 100% / .007 | 100% / .007 |

**Learning rate and capacity.** Changing only the fast rate moved spring first-move accuracy from 55.95% to 97.69%. In a paired validation probe, only the direction token was flipped while the start and preceding history stayed fixed. Spring predicted *both* answers correctly on 20.50% of pairs at the original rate and 93.19% at the selected rate. Selected DFA ablation and AdamW each scored 100%. These interventions show that the network can represent the direction/start interaction and the original rate learned it too slowly for the budget.

**Credit assignment and stability.** Mean cosine similarity between the DFA first-layer drive and the analytic descent direction on first-move validation examples rose from 0.022 at initialization to 0.840 for original-rate spring and 0.808 for selected spring. A positive aggregate cosine alone did not predict mastery. Every trial stayed finite. Transient target crossings at the original rate show sensitivity to the stopping checkpoint. The experiments establish a rate effect and a spring effect. They leave open how much of AdamW's CE advantage comes from backpropagation versus adaptive optimization.

## What the springs contributed

At the selected rate, an anchor with stiffness disabled followed exactly the fast-only DFA trajectory. Reading its consolidated weights lowered mean test CE by **0.0061** versus fast-only DFA, with lower CE in 8/9 pairs. Its first-move accuracy was 99.74%. At the original rate the readout benefit was mixed, with lower CE in 4/9 pairs.

Enabling the coupled stiffness and restoring term on top of consolidation raised deployed test CE by **0.0498** at the selected rate and **0.0195** at the original rate, in 9/9 pairs each. With the full spring dynamics, reading the anchor instead of its own fast weights recovered 0.0041 CE on average at the selected rate. Full spring still finished 0.0438 CE worse than fast-only DFA. Restoring toward a fixed initial anchor was the weakest control, reaching the target in 0/9 pairs at either rate. This isolates a penalty from the coupled mechanism under these settings. Separating stiffness growth from restoring strength would require another controlled experiment. [Per-corpus effects and all validation/test position scores](results/r01/tables.md)

## Engineering verification and cost

- **Software PASS.** All 14 historical tests and five R01 tests passed. The R01 tests exercise seed independence, position indexing and direction flips, component controls, the read-only credit diagnostic and test exclusion from search.
- **Reproducibility PASS.** The replay exactly matched all 81 search runs and 90 confirmation runs on learning outputs, including curves, probes, final metrics and target status. All nine historical controls matched archived validation curves, test CE and position accuracies. [Replay audit](results/r01/reproducibility.json)
- **Scientific outcomes.** The primary search recorded 45 PASS and 36 FAIL. The confirmation set, including controls and historical replays, recorded 42 PASS and 48 FAIL. These sets contain repeated configurations. A replay adds no independent scientific observations. [Validation search runs](results/r01/primary/search.json) · [Per-seed confirmation CSV](results/r01/per_seed.csv) · [Curves, probes and timing for every confirmation](results/r01/primary/confirmation.json)
- **CPU cost.** All methods deploy 8,868 bytes of float32 weights. Each local variant allocated another 19,464 bytes of persistent training state; AdamW allocated 17,744. The ablation retains C01's three-array allocation. Sixteen profiled updates at the selected settings measured 622,848 dense matrix FLOPs per local step and 624,000 for AdamW. Summed profiler self-CPU time per step was 756 µs for spring, 765 µs for ablation and 592 µs for AdamW. Profiling introduces overhead. Unprofiled time to target appears above. [CPU operator profile](results/r01/profile.json)

The inference is limited to three independent corpora from one grammar, three model seeds, one training horizon and a small rate grid. Corpus/model pairs share both factors. Timing varies with host load. GPU time, natural-language generalization and scaling remain unmeasured.

## Reproduction

From the repository root, run the following in the C01 Python/PyTorch environment. The diagnosis command writes a fresh primary run, replay and audit. Its exit code reports reproducibility. Scientific decisions are stored per trial.

```powershell
python -m unittest discover -s tests -v
python -m benchmarks.r01 --output r01.local
python -m benchmarks.r01_tables r01.local
```
