# C01 engineering and experiment report

The local DFA network learned from random weights on both tasks. All three frozen spring models solved XOR. On held-out token documents, mean validation cross-entropy fell from 2.227 to 0.743 nats and accuracy rose from 7.48% to 64.55%. Only one spring seed met the preset 66% validation accuracy target. AdamW met it for all three seeds. The ablation achieved lower mean test cross-entropy than the full spring model. These trials establish local learning and leave the benefit of adaptive springs unresolved.

## Protocol

The spring recurrence follows the original `SpringLayer` update order. A single dense tanh hidden layer receives the output residual through a fixed random projection. Both affine layers, including biases, use the three-state rule. Each seed shares initial forward weights, input encoding, batches and evaluation points across all methods. Spring evaluation uses consolidated anchors, ablation evaluation uses fast weights, and AdamW uses its current parameters. The ablation retains the same state allocation and update code with both slow rates set to zero. The raw results also include fast-weight validation scores. [Learning equations and source map](README.md) · [Experiment configuration](configs/c01.json)

**T01** uses all four balanced XOR cases. The hidden width is 16, the batch size is four and the budget is 800 updates. The target is four correct classifications. This measures fitting the truth table. The validation and test entries refer to the same four cases.

**T02** uses 144 unique 40-token documents split 96/24/24 before fitting the vocabulary on the training documents. Each clause chooses a random direction and starting symbol, then walks around a six-symbol ring. The predictor receives the previous four tokens from the same document. The three splits contain 3,456/864/864 next-token targets. The classifier has nine outputs including an unknown-token class. The hidden width is 48. Each seed receives 1,000 batches of 64 training windows sampled with replacement. Repeated local patterns across documents are intentional. Validation is evaluated every 25 steps and at initialization. Test scores use the last step.

## Results

Values are arithmetic means over seeds 11, 29 and 47. Perplexity is computed per run before averaging. The step columns give the first observed target hit in seed order. A dash means the target was never observed. [Per-seed results, time to target and validation curves](results/c01.json)

| Task | Method | Initial val CE | Final val CE | Val PPL | Val accuracy | Test CE | Test PPL | Test accuracy | Steps to target |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| XOR | Spring | 0.704 | 0.0073 | 1.007 | 100% | 0.0073 | 1.007 | 100% | 40 / 40 / 60 |
| XOR | Ablation | 0.704 | 0.0031 | 1.003 | 100% | 0.0031 | 1.003 | 100% | 20 / 20 / 40 |
| XOR | AdamW | 0.704 | 0.0002 | 1.000 | 100% | 0.0002 | 1.000 | 100% | 20 / 20 / 20 |
| Tokens | Spring | 2.227 | 0.743 | 2.101 | 64.55% | 0.753 | 2.122 | 65.82% | — / 975 / — |
| Tokens | Ablation | 2.227 | 0.725 | 2.065 | 64.43% | 0.733 | 2.082 | 65.55% | — / 700 / — |
| Tokens | AdamW | 2.227 | 0.512 | 1.669 | 73.50% | 0.518 | 1.679 | 74.15% | 250 / 300 / 175 |

The final spring validation accuracies on T02 were 64.12%, 66.09% and 63.43%. Ablation seed 29 crossed 66% at step 700 and finished at 65.62%. The other two ablation seeds never crossed. A failed gate remains in the results even when cross-entropy improved.

The largest test error difference was on the **first symbol predicted from a direction and starting symbol**. Spring accuracy there was 53.97%, ablation 52.38% and AdamW 100%. All three methods predicted both later deterministic symbols with 100% accuracy. The random direction and starting symbol have expected best accuracies of 50% and 16.67%. Given the mix of evaluated positions, a predictor that knows the generator has expected overall accuracy of 74.07%. Spring fast-weight validation CE averaged 0.744, close to the deployed anchors' 0.743. Freezing alone does not explain the gap.

## CPU measurements

Each trial ran in a fresh CPU process with one PyTorch thread and deterministic algorithms. Timing includes batch selection and learning updates. Data construction, a separate model's warm-up step, validation and test evaluation are outside the timer. There was no explicit compilation. Rates count XOR samples or predicted tokens. Peak memory is the process lifetime high-water working set, including imports, setup and evaluation. State counts are persistent tensor elements between steps and exclude activations, temporary update tensors and cleared gradients.

| Task | Method | Deployed elements | Training-only elements | Mean training seconds | Mean items/s | Mean peak MiB |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| XOR | Spring | 82 | 196 | 0.193 | 16,546 | 287.4 |
| XOR | Ablation | 82 | 196 | 0.235 | 14,286 | 286.9 |
| XOR | AdamW | 82 | 166 | 0.193 | 16,608 | 287.2 |
| Tokens | Spring | 2,217 | 4,866 | 0.382 | 168,339 | 291.1 |
| Tokens | Ablation | 2,217 | 4,866 | 0.405 | 163,229 | 290.8 |
| Tokens | AdamW | 2,217 | 4,436 | 0.329 | 194,588 | 291.5 |

The spring counts include two extra synaptic arrays and the fixed feedback matrix. AdamW counts include its two moment arrays and a step scalar for each weight matrix. The timings describe short, unoptimized CPU runs. A separate CPU profile counted five dense matrix multiplies per method per step. On T02 these accounted for 622,848 FLOPs for either DFA variant and 624,000 for AdamW. The profile excludes elementwise updates, checks, activation, softmax and memory traffic. [Measured matrix FLOPs](results/matrix_flops.json)

## Verification and review

| Check | Outcome |
| --- | --- |
| Equation ordering, finite updates, saturation, frozen snapshot and ablation | PASS |
| Fixed DFA, manual hidden credit and absence of Autograd in the local path | PASS |
| Seeded replay, matched initial predictions, state counts and dataset isolation | PASS |
| Independent repeat of all 18 trials, including every validation point, terminal test score and first target step | PASS, exact match |
| CPU matrix-FLOP profile | PASS, all six task/method combinations |
| T01 final target and reduced CE | PASS, 9 of 9 |
| T02 final target and reduced CE | PASS, 4 of 9. FAIL, 5 of 9 |
| GPU time to target and GPU cost | NOT_RUN, deferred to C02 |

**Scope and limitations.** Fourteen automated tests exercised the software. The three model seeds share one synthetic corpus. The experiment supports a claim about this grammar and this budget. It provides little evidence about natural language, deeper networks or scaling. C01 uses the original three synaptic laws with dense tanh units, fixed-window one-hot inputs and bias connections. The legacy production kernels, sleep, growth, checkpointing and distributed trainer were outside scope. The original repository was accessed read-only at revision `b220f892`. [Reference revision](https://github.com/naoufelbvb2-arch/next-23-MTL/tree/b220f8922f3d7988190707130e17d7e4494c1d7f)

**C02 recommendation.** Keep GPU spending deferred pending review. First repeat T02 on independent corpus seeds with a prespecified, matched tuning budget for all three learners. Track the first direction-dependent prediction separately and require repeatable attainment of the same validation target before comparing GPU time to quality.
