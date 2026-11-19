# C02-A decoder and local credit assignment

The model is a single pre-normalized decoder block with 32 residual channels, two 16-channel attention heads, a 64-channel tanh feed-forward layer and a 16-token context. It uses trainable token embeddings, fixed sinusoidal positions, three trainable RMS normalization gain vectors, two residual additions and an untied language-model head. The 11 trainable tensors include all projections. The affine maps use no biases. A single forward implementation serves local training, anchor inference and AdamW. [Decoder source](ssenn/decoder.py)

For input indices (x), the forward equations are:

```text
X = embedding[x] + position
A = rms(X) * norm_attention
Q, K, V = split_heads(A Wq), split_heads(A Wk), split_heads(A Wv)
P = softmax(causal_mask(Q Kᵀ / sqrt(head_width)))
M = join_heads(P V)
U = X + M Wattention_out
F = rms(U) * norm_ff
H = tanh(F Wff_in)
L = U + H Wff_out
Z = rms(L) * norm_final
logits = Z Whead
```

The mask admits keys at positions up to and including the query position. RMS uses the channel axis within each individual token. A logit at position (i) predicts token (i+1). The model has no operation that computes normalization statistics over future positions or other examples.

## Local learning

Let `E = (one_hot(next_token) − softmax(logits)) / number_of_batch_targets`. Each direct-feedback site has its own fixed, independently sampled matrix `B_site` of shape vocabulary by receiving width. Write `R_site = E B_site`. All drives use the same pre-update weights and activations. An outer product below sums over both batch and sequence positions. [Local update implementation](ssenn/decoder_local.py)

| Trainable tensor | Local additive drive |
| --- | --- |
| Token embedding | Scatter-add `R_embedding` to each observed input token's row. Repeated tokens accumulate. |
| Attention, FF and final norm gains | For each gain, sum its unit-RMS input times its own `R_norm` over batch and positions. |
| Attention output projection | Outer product of concatenated attended values `M` and `R_attention_out`. |
| FF input projection | Outer product of `F` and `R_ff_in * (1 − H²)`. |
| FF output projection | Outer product of `H` and `R_ff_out`. |
| LM head | Outer product of final normalized features `Z` and `E`. |
| Q, K and V projections | Outer product of `A` with each rejoined head signal derived below. |

For attention, split `R_attention_mix` into heads to obtain `G`. Treat this signal as fixed at the output of the attention mix `P V`. The only derivative through the attention operation is its **local** derivative with this direct signal:

```text
C  = G Vᵀ
S  = P * (C − rowsum(P * C)) / sqrt(head_width)
dQ = S K
dK = Sᵀ Q
dV = Pᵀ G
```

The zero probabilities at masked coordinates give zero score drives. Earlier keys and values can receive learning signals from later queries that attended to them during teacher-forced training. During prediction each query has access only to its prefix. None of these drives uses an error propagated through the head, residual stream or feed-forward stack. Autograd is disabled throughout local training, and local model parameters have gradient recording disabled. Tests compare Q/K/V drives to finite differences of the direct-signal attention objective and check for forbidden differentiation calls. [Causality and credit tests](tests/test_decoder.py)

**Lightweight DFA** applies `W ← W + η drive`. **Moving anchor** uses the same fast update and then applies `A ← A + η₀ (W − A)` to every tensor, including embeddings and norm gains. It evaluates with the anchors. Both validate all proposed tensors before committing any update. **AdamW** trains the same forward graph on next-token CE using PyTorch Autograd. The local variants allocate fixed projections; the anchor variant adds one full set of forward tensors. AdamW retains optimizer moments and step counters. Fixed positions and the mask are reported separately. [Conventional comparison](ssenn/decoder_adam.py)

## Data and checkpoint boundaries

Structured symbols and real-text characters each have their own vocabulary. IDs for beginning, end and unknown are reserved. Fit the remaining vocabulary only on the training documents. Sample full windows uniformly from the set of eligible training starts without crossing document boundaries. Evaluate with consecutive document-local blocks, score each next token including EOS once, and ignore padding in short tails. The context restarts at each evaluation block. Real-text PPL has units of characters (plus EOS); structured PPL uses grammar symbols (plus EOS).

Checkpoints replace a file atomically at completed update boundaries. They store the step, model tensors, projections, anchors or optimizer state, and minibatch generator state. The benchmark binds checkpoints to a hash of the run configuration, corpus manifest, method and seed. The replay reconstructs the learner and sampler at step 300 and checks its recorded trajectory against the uninterrupted run. [Checkpoint implementation](ssenn/lm_training.py)
