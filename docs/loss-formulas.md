# Loss Formulas

All diffusion losses map from LLM training via one key insight:
- **LLM:** `avg_log_prob(sequence)` — higher = model prefers this
- **Diffusion:** `-MSE(pred, target)` — lower MSE = model prefers this

So `chosen_logps - rejected_logps` in Grimoire becomes `loss_rejected - loss_chosen` in Atelier.

## Epsilon / SFT (EpsilonLoss)

Standard denoising training. Works with any adapter — the adapter determines the target.

```
L = mean(MSE(model_pred, target))

# Epsilon prediction (DDPM): target = noise
# Flow matching: target = noise - latents
# V-prediction: target = sigma * noise - (1 - sigma) * latents
```

## Flow Matching (FlowMatchingLoss)

Specialized for flow matching models (Qwen-Image-Edit, SD3, FLUX) with latent normalization and packing.

```
noisy_input = (1 - sigma) * latents + sigma * noise
target = noise - latents

L = weighted_mean(MSE(model_pred, target))
```

**Timestep sampling:** Density-based (logit-normal) rather than uniform.

**Weighting schemes:** `none` (default), `sigma_sqrt` (weight by sigma^(-2)).

## Diffusion DPO (DiffusionDPOLoss)

```
loss_chosen = mean(MSE(pred_chosen, target), dim=[spatial])     # per sample
loss_rejected = mean(MSE(pred_rejected, target), dim=[spatial]) # per sample

log_ratio = clamp(loss_rejected - loss_chosen, -C, C)
L_DPO = -mean(log(sigmoid(beta * log_ratio)))
L_SFT = mean(MSE(pred_chosen, target))

L = L_DPO + sft_weight * L_SFT
```

**Timestep bias:** Biased to [30%, 80%] — extremes provide weak DPO signal.

**Beta scheduling:** `constant`, `linear` (warmup + tail decay), `cosine`.

## Diffusion SimPO (DiffusionSimPOLoss)

Reference-free preference with a target reward margin. Simplest preference loss.

```
L = -mean(log(sigmoid(beta * (loss_rejected - loss_chosen - gamma))))
```

**gamma:** Target margin (default 0.5). Enforces minimum gap between chosen/rejected.

**No reference model, no SFT term.** Simpler than DPO.

## Diffusion ORPO (DiffusionORPOLoss)

SFT on chosen + odds ratio preference. No reference model.

```
# Use -MSE as implicit log-probability
logp_chosen = -loss_chosen
logp_rejected = -loss_rejected

log_odds = (logp_c - logp_r) - (log(1-exp(logp_c)) - log(1-exp(logp_r)))

L_SFT = mean(MSE(pred_chosen, target))
L_OR  = -beta * mean(log(sigmoid(log_odds)))

L = L_SFT + L_OR
```

## Diffusion CPO (DiffusionCPOLoss)

SFT on chosen + contrastive preference. Reference-free, theoretically cleaner than ORPO.

```
x = beta * (loss_rejected - loss_chosen)
L_pref = -mean((1-eps)*log(sigmoid(x)) + eps*log(sigmoid(-x)))

L = MSE(chosen) + beta * L_pref
```

**label_smoothing (eps):** Set > 0 for conservative regularization on noisy preferences.

## Diffusion KTO (DiffusionKTOLoss)

Unpaired binary feedback. Each image is labeled good/bad independently — **no chosen/rejected pairs needed.**

```
# Per-sample "log ratio" (positive = policy improved over reference)
log_ratio = ref_mse - policy_mse

# KL estimate from batch
KL_ref = clamp(mean(log_ratio), min=0)

# Asymmetric loss
L_desirable   = lambda_d * (1 - sigmoid(beta * (log_ratio - KL_ref)))
L_undesirable = lambda_u * (1 - sigmoid(beta * (KL_ref - log_ratio)))

L = mean(L_desirable) + mean(L_undesirable)
```

**Requires reference model** (frozen copy or base weights via `disable_adapter`).

**lambda_u > lambda_d** implements loss aversion (penalize bad images more than reward good ones).

## Diffusion IPO (DiffusionIPOLoss)

Squared-loss variant of DPO. Prevents overfitting on noisy preferences.

```
# Policy and reference preference margins
pi_margin  = loss_rejected_pi  - loss_chosen_pi
ref_margin = loss_rejected_ref - loss_chosen_ref

L = mean((pi_margin - ref_margin - 1/(2*beta))^2)
```

**Requires reference model.** The `1/(2*beta)` term is a target margin.

## Summary Table

| Loss | Reference Model? | Paired Data? | Best For |
|---|---|---|---|
| **Epsilon/SFT** | No | No (single images) | Standard training |
| **Flow Matching** | No | No (single images) | Flow matching models with latent packing |
| **DPO** | No (simplified) | Yes | General preference alignment |
| **SimPO** | No | Yes | Simple, fast preference tuning |
| **ORPO** | No | Yes | Single-pass SFT + preference |
| **CPO** | No | Yes | Alternative to ORPO |
| **KTO** | Yes | **No** (unpaired) | When you only have quality labels |
| **IPO** | Yes | Yes | Noisy preference labels |
