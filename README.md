# SS-ENN V2

C01 tests whether a dense, locally trained network can learn XOR and predict the next token in held-out synthetic documents. It includes a fixed-feedback ablation and a capacity-matched AdamW comparator. See the [engineering report](REPORT.md) for outcomes and failed quality targets.

## Reproduce C01

From the repository root, use Python 3.11+ with a CPU build of PyTorch. The recorded environment used Python 3.14.4, PyTorch 2.12.0+cpu, and psutil 7.2.2 on Windows. Windows peak working set collection uses psutil when available. The package runs directly from this checkout.

```powershell
python -m pip install torch==2.12.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install psutil==7.2.2
python -m benchmarks.verify
```

The last command runs the software tests, all 18 scientific trials and a separate CPU matrix-FLOP profile. It writes per-seed results and validation curves to `results.local.json`. Exit code 1 signals a software failure, a trial error, an unmet final quality gate, or a profiler failure. The recorded configuration returns 1 because five trials miss their final accuracy target.

The test and benchmark entry points can also run independently.

```powershell
python -m unittest discover -s tests -v
python -m benchmarks.c01 --config configs/c01.json --output results.local.json
python -m benchmarks.flops
```

## Learning rule

The local path uses CPU float32 tensors with gradient recording disabled. For each batch, let `E = one_hot(y) - softmax(logits)`. Output units receive `E`. Hidden units receive `(E @ B) * (1 - h²)`, where `B` is a fixed random matrix independent of the forward weights and `h` is the hidden tanh activation. The manual derivative applies to the receiving unit only. At both layers the drive `D` is the batch mean of input × local error. A constant input 1 supplies each bias.

```text
w_next  = w + (eta_w * D - gamma * k * (w - w0))
stretch = w_next - w0
w0_next = w0 + eta_0 * (1 - k) * stretch
k_next  = clip(k + beta_k * abs(stretch) * (1 - k), 0, 1)
```

Both slow laws use the old anchor and stiffness. Initially `w0 = w` and `k = 0`. Updates require matching matrix dimensions and reject nonfinite candidates before replacing either layer. The frozen predictor takes independent copies of the two anchor matrices. In the ablation, setting `eta_0 = beta_k = 0` keeps stiffness zero and the anchor fixed. Its deployed predictor uses the learned fast weights. This choice makes the ablation a test of DFA alone. [Reference spring implementation](https://github.com/naoufelbvb2-arch/next-23-MTL/blob/b220f8922f3d7988190707130e17d7e4494c1d7f/engine/spring_layer.py)

## Source map

| File | Responsibility |
| --- | --- |
| `ssenn/dynamics.py`, `ssenn/credit.py` | Three-state update and replaceable hidden credit projection |
| `ssenn/network.py`, `ssenn/local.py` | Shared forward pass, local learning and frozen inference |
| `ssenn/conventional.py` | PyTorch Autograd and AdamW |
| `ssenn/data.py`, `ssenn/experiment.py` | Document splitting, tokenizer fitting, training and measurement |
| `configs/c01.json`, `benchmarks/`, `tests/` | Experiment settings, reproducible entry points and verification |
