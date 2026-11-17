# Extension: grokking under minibatch SGD

## Question

In *To Grok Grokking* (Xu, Vardi and Safran, 2026), full-batch gradient descent fits the training data quickly (t1). It generalizes much later (t2), because only weight decay removes the part of the initialization the data cannot see. Grokking therefore runs on a weight-decay clock: t2 ≈ ln((m−n)ν²/(mε)) / (2ηλ).

This extension asks what minibatch SGD does to that clock. Minibatching could matter in three ways:

1. it changes the clock itself (A1, B1);
2. it changes which learning rates are usable, and so the number of examples needed to grok (A2);
3. its gradient noise hides generalization (A3).

B2 tests whether the paper's mechanism explains grokking in the two-layer network at all.

## Settings

**Setting A: ridge regression (paper Fig. 2).**
- **Data:** inputs x ∈ ℝ^m with x ~ N(0, I_m/m), n training points, teacher θ* a random unit vector, labels y = θ*ᵀx + σξ with ξ ~ N(0, 1). The noise σ is 0 except in A3. Test labels are always clean.
- **Model:** linear, prediction θᵀx, initialized as θ_0 ~ N(0, ν² I_m).
- **Defaults:** n = 100, m = 1000, λ = 1e-4, η = 1, ν² = 1.
- **Test loss:** the population loss (1/m)‖θ − θ*‖², computed exactly.

**Setting B: two-layer ReLU network (paper Fig. 4).**
- **Model:** f(x) = Σ_j a_j ReLU(w_jᵀx), with W ∈ ℝ^{m×d} and weight decay on both W and a.
- **Data:** zero teacher (all labels 0); data and initialization as in `configs/fig4.yaml`.
- **Defaults:** n = 50, m = 1000, d = 50, λ = 0.05, η = 1e-4, ν² = 1.
- **Test loss:** estimated on 5000 test points.

## Notation and definitions

| Symbol | Meaning |
|---|---|
| b | Batch size. b = n is full-batch GD |
| t | Optimization steps |
| E = b·t | Examples processed. Epochs are E/n |
| K = XXᵀ | n×n kernel of the training inputs in setting A |
| κ_min, κ_max | Smallest and largest eigenvalues of K |
| P, P⊥ | Orthogonal projections onto span{x_1, …, x_n} and onto its complement |
| W_⊥ = W(I − P_X) | Part of the first-layer weights orthogonal to the training inputs (setting B) |
| t1 | Last step with training loss ≥ ε (`grok/metrics.py`, `last_above`) |
| t2 | First step with test loss ≤ c (`first_at_or_below`) |
| ε, c | Thresholds, 0.01 by default |

Loss conventions follow the replication (`REPRODUCTION_NOTES.md`, U2).

**SGD update (setting A).** B_t holds b indices drawn uniformly with replacement, unless an arm says otherwise:

θ_{t+1} = (1 − ηλ) θ_t − (η/b) Σ_{i∈B_t} x_i (x_iᵀθ_t − y_i)

## Facts the design rests on

**Fact 1 (exact, setting A).** Every x_i lies in span{x_1, …, x_n}. So for any batch size, sampling scheme and batch sequence,

P⊥θ_t = (1 − ηλ)^t P⊥θ_0.

The test loss splits into two orthogonal terms:

L_test(t) = (1/m)‖P(θ_t − θ*)‖² + (1/m)‖(1 − ηλ)^t P⊥θ_0 − P⊥θ*‖²

The second term, the **null-space loss**, is the same for GD and for every SGD run with the same η and λ.

**Fact 2 (exact, setting B).** The gradient of each w_j is a combination of the x_i, and the training outputs depend only on W P_X. So W_⊥ decays exactly as (1 − ηλ)^t under any batching. It changes only the test outputs.

**Approximations used for predictions.** These are not proven; A0 checks them numerically before any prediction is frozen.

- GD is stable for η < 2/(κ_max/n + λ). This bound is exact.
- SGD with b = 1 is stable for η below about 2/max_i ‖x_i‖².
- At the ridge solution, SGD noise scales with the training residual r = nλ(K + nλI)⁻¹y. This residual is close to zero when nλ ≪ κ_min.

## Common protocol

- **Paired seeds.** All arms of an experiment share the data, initialization and teacher; only the batch-sampling generator differs. Results are reported as within-seed differences against GD, with bootstrap 95% confidence intervals. Unpaired spreads are also reported.
- **Exact baseline.** In setting A the GD baseline is the closed form in `grok/spectral.py`.
- **Logged quantities.** Training loss on the full training set and test loss, never the minibatch loss. In setting A also the null-space loss and the row-space loss; in setting B also ‖W_⊥‖, ‖W P_X‖ and ‖a‖. Times are logged on the replication's grid and t1, t2 are refined by bisection between logged points.
- **Crossing times under SGD.** Report t2 as the first crossing and as the sustained crossing: the first t after which the test loss stays ≤ c at every later logged time. t1 is already a sustained definition.
- **Thresholds.** Sweep ε and c over {0.005, 0.01, 0.02, 0.05}. The curves are stored, so this needs no new training.
- **Step cap and censoring.** Each run is capped at 10× its predicted t2. Seeds that never cross a threshold are reported as a censored fraction, never dropped.
- **Frozen predictions.** Each experiment's predictions and decision rules go into a file that is frozen before the full runs, as `anchors/anchors.json` was for the replication.
- **Implementation of setting A.** θ = Qc + θ_⊥, where Q is an orthonormal basis of span{x_i}. Only the n coordinates c are simulated, at a cost of O(bn) per step. θ_⊥ follows Fact 1 in closed form. The test loss is exact.

## Experiments

### A0. Correctness checks (before any full run)

- **SGD at b = n against the closed form.** Over 10^4 steps, the relative difference in training and test loss must be below 1e-10.
- **Fact 1.** For b < n, ‖P⊥θ_t − (1−ηλ)^t P⊥θ_0‖ / ‖P⊥θ_0‖ must be below 1e-10.
- **Fact 2.** Run the same check for W_⊥ in setting B with n < d.
- **The approximations above.** On 10 seeds, check:
  - κ_min and κ_max;
  - the measured η_max for b = 1 and b = n;
  - the training residual at the ridge solution.

  If an approximation fails, revise the prediction that uses it before freezing it.

### A1. Does minibatching change the weight-decay clock?

**Design.** Setting A defaults, 50 paired seeds.

| Arm | b | η | λ | Sampling |
|---|---|---|---|---|
| GD (closed form) | 100 | 1 | 1e-4 | – |
| Matched steps | 1, 10, 25 | 1 | 1e-4 | with replacement |
| Matched epochs | 1, 10, 25 | b/n | 1e-4 | with replacement |
| Reshuffling | 10 | 1 | 1e-4 | without replacement, one pass per epoch |
| No weight decay | 1, 100 | 1 | 0 | with replacement |

In the matched-epochs arm, each epoch applies a decay of (1 − bλ/n)^{n/b} ≈ (1 − λ) and makes the same expected progress as one GD step.

**Measure.** t1, t2, t2 − t1 in steps and in examples; the null-space and row-space losses over time.

**Predictions.**
1. The null-space loss equals GD's to within 1e-10 relative (Fact 1).
2. Matched steps: t2 in steps is within 1% of GD's for every b.
3. Matched epochs: t2 in epochs is within 1% of GD's.
4. t1 in steps is about 2× GD's at b = 1 and approaches GD's as b grows. This is a mean-square rate argument, so it is approximate.
5. No weight decay: the test loss stays above (1/m)‖P⊥(θ_0 − θ*)‖² ≈ ν²(m−n)/m, and t2 is censored for every seed and every b.

**Decision rule.** A t2 gap that the row-space loss does not account for means a bug or a wrong argument, and stops the study until it is explained. For t1, only the order is tested: the prediction fails if the ratio to GD exceeds 3.

### A2. What does grokking cost in data at each batch size's best learning rate?

**Design.**
- Setting A, m = 1000, n ∈ {50, 100, 300}, so n/m ∈ {0.05, 0.1, 0.3}.
- b ∈ {1, 4, 16, 64, n}.
- For each (n, b), find η_max(b) by bisection on a log grid. η counts as stable if, for all 10 probe seeds, the training loss after 10^4 steps is below its initial value.
- Run at η = η_max(b)/2 with 50 paired seeds. As a robustness check, also run η_max(b)/4.

**Measure.** Examples to grok, E2 = b·t2; ηλ·t2; t1; and t2/t1.

**Predictions.**
1. ηλ·t2 is the same in every arm, to within 2%. This follows from Fact 1 once the row space has converged.
2. η_max(n) = 2/(κ_max/n + λ), and η_max(1) ≈ 2/max_i ‖x_i‖².
3. E2(GD)/E2(b = 1) ≈ κ_max / max_i ‖x_i‖², computed per seed from the data. This is about (1 + √(n/m))²: roughly 1.5, 1.7 and 2.4 for the three values of n/m.
4. E2 is flat in b up to a critical batch size, then grows linearly in b.

**Decision rule.** Prediction 3 fails if the median per-seed ratio of measured to predicted is outside [0.85, 1.15] for any n/m.

### A3. Can SGD noise delay grokking in setting A?

**Design.**
- Setting A, n = 100, m = 1000.
- λ chosen so that nλ/κ̄_min ∈ {0.01, 0.1, 1, 10}, where κ̄_min ≈ (1 − √(n/m))² ≈ 0.47.
- σ ∈ {0, 0.1}; b ∈ {1, n}; η = 1; 50 paired seeds.
- t2 is measured relative to GD's exact limit L_∞^GD: the first t with L_test ≤ L_∞^GD + c. This removes the noise floor GD itself has when σ > 0.

**Measure.**
- The extra stationary loss from SGD, ΔL: the mean of L_test^SGD − L_∞^GD over the last 2/(ηλ) steps.
- t1 and t2 for GD and SGD.
- GD's ratio t2/t1.

**Predictions.**
1. ΔL < c whenever GD's t2/t1 ≥ 10.
2. ΔL reaches c only where nλ/κ_min ≳ 1.
3. SGD changes t2 by less than 5% wherever grokking is pronounced (t2/t1 ≥ 10).

**Decision rule.** The claim "SGD noise can only matter where grokking has already disappeared" is rejected if any condition with GD t2/t1 ≥ 10 shows SGD changing t2 by more than 5%.

### B1. Does the setting A result carry over to the network?

**Design.** Setting B defaults; b ∈ {1, 5, 10, 25, 50} with b = 50 being GD; η = 1e-4; λ = 0.05; 50 paired seeds, or at least 20 if compute is short.

**Measure.** t1 and t2 in steps and in examples; ‖W P_X‖ and ‖a‖ over time.

**Prediction.** t2 in steps is within 5% of GD's for every b, because the gradient noise is small at η = 1e-4.

**Decision rule.** A gap above 5% whose 95% confidence interval excludes 0 means minibatching changes the network's clock. If so, B3 becomes a priority.

### B2. Does the paper's linear mechanism explain grokking in the network?

**Design.** Setting B, n ∈ {10, 20, 40} (n < d) and n ∈ {50, 80, 160} (n ≥ d); GD and b = 1; η = 1e-4; 20 paired seeds.

**Measure.**
- ‖W_⊥‖, ‖W P_X‖ and ‖a‖ over time.
- **Ablated test loss**: the test loss of the network with W replaced by W P_X, that is, with the null-space part removed. Its crossing time is t2^∥.

**Predictions.**
- For n < d: ‖W_⊥‖ follows (1 − ηλ)^t exactly under both GD and SGD (Fact 2).
- Whether that decay is what delays generalization is the open question this experiment answers.
- For n ≥ d: W_⊥ = 0, so any delay must come from elsewhere.

**Decision rule.** At a given n, the mechanism explains the delay if both of the following hold:
1. t2^∥ ≤ 0.1·t2;
2. t2 matches, to within 20%, the time at which the null-space part of the test output falls below c, using the measured decay of ‖W_⊥‖ and ‖a‖.

Otherwise, report what does track t2 among ‖a‖, ‖W P_X‖ and the activation pattern.

### B3. Larger learning rates in the network (optional)

**Design.** η ∈ {1e-4, 1e-3, 1e-2}, keeping only stable values; b ∈ {1, 50}; 20 paired seeds.

**Prediction.** ηλ·t2 is the same across η and b, to within 10%.

Run this only if B1 or B2 shows a deviation from the linear picture.

## Out of scope

- The Fig. 2 sweeps repeated under SGD. A1 implies their outcome.
- Rescaling λ with the batch size. It changes the objective; the matched-epochs arm of A1 replaces it.
- The network with no weight decay. It tests a different mechanism and needs a literature check first.
- An evaluation-set check in setting A, where the test loss is exact.

## Compute (estimate)

| Experiment | Where | Estimate |
|---|---|---|
| A0–A3 | CPU or Colab | Under 1 hour in total, using the n-dimensional simulation |
| B1 | Colab | About 10 CPU-equivalent hours |
| B2 | Colab | About 10 CPU-equivalent hours (n = 160 is the most expensive) |
| B3 | Colab | About 10 CPU-equivalent hours, only if triggered |

The setting B figures are scaled from CPU step times measured for the replication. GPU speed is not yet measured.
