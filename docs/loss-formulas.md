# Loss Formulas

## Flow Matching (FlowMatchingLoss)

Used with flow matching models: Qwen-Image-Edit, SD3, FLUX.

The model learns the velocity field between noise and data.

```
noisy_input = (1 - sigma) * latents + sigma * noise

target = noise - latents

L = weighted_mean(MSE(model_pred, target))
  = weighted_mean((model_pred - (noise - latents))^2)
```

**Timestep sampling:** Uses density-based sampling (logit-normal distribution) rather than uniform. This concentrates training on the most informative timesteps.

**Sigma computation:** Looked up from the scheduler's sigma table based on sampled timesteps.

**Weighting schemes:**
- `none` (default): All timesteps weighted equally
- `sigma_sqrt`: Weight by `sigma^(-2)`, clamped to prevent instability at low noise levels

## Diffusion DPO (DiffusionDPOLoss)

DPO adapted for diffusion models. The model learns to predict noise more accurately for preferred images.

Given `(prompt, chosen_image, rejected_image)`:

```
# 1. Encode both images to latents
chosen_latents = VAE.encode(chosen_image) * scaling_factor
rejected_latents = VAE.encode(rejected_image) * scaling_factor

# 2. Shared noise and timestep (biased to 30-80% range)
noise ~ N(0, I)
t ~ Uniform(0.3T, 0.8T)

# 3. Add noise to both
noisy_chosen = scheduler.add_noise(chosen_latents, noise, t)
noisy_rejected = scheduler.add_noise(rejected_latents, noise, t)

# 4. Predict noise for both
pred_chosen = model(noisy_chosen, t, prompt)
pred_rejected = model(noisy_rejected, t, prompt)

# 5. Per-sample MSE
loss_chosen = mean(MSE(pred_chosen, noise), dim=[C,H,W])
loss_rejected = mean(MSE(pred_rejected, noise), dim=[C,H,W])

# 6. DPO objective (lower MSE = better prediction = preferred)
log_ratio = clamp(loss_rejected - loss_chosen, -C, C)
L_DPO = -mean(log(sigmoid(beta * log_ratio)))

# 7. SFT regularization
L_SFT = mean(MSE(pred_chosen, noise))

# 8. Total loss
L = L_DPO + sft_weight * L_SFT
```

**Timestep bias:** Sampling is biased to the `[30%, 80%]` range of the scheduler. Very noisy (high t) or nearly clean (low t) timesteps provide weak DPO signal.

**Logit clamping:** `log_ratio` is clamped to `[-C, C]` (default C=5) to prevent extreme gradient spikes when the model strongly prefers one image.

**Beta scheduling:**
- `constant` (default): Fixed beta throughout training
- `linear`: Warmup from 0 to beta, then decay in the final 30% of training
- `cosine`: Cosine annealing from 0 to beta

**SFT weight:** The regularization term prevents the model from forgetting how to denoise entirely. Typical values: 0.1-0.3.
