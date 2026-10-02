# Challenge 1: the five classical detectors (assumptions)

Five classical methods share the same preprocessing and the same interface: a score map in units of the noise σ, then a threshold. They stack increasingly strong priors. This note describes the assumptions. Results are in [`../README.md`](../README.md).

The code is [`baselines.py`](baselines.py). The deep models are in `models.py`, `train.py`, `synth.py` and `data_gen.py`.

## Common preprocessing

An interdot is a thin dip (negative signal) on a background striped horizontally. The stripes follow the fast scan axis.

1. Subtract the median of each row. Sticks cover few pixels, so the median estimates the row background.
2. Flip the sign. The dip becomes a peak.
3. Scale by median / MAD (factor 1.4826). The image is expressed in σ of the background noise. A classical z-score would be inflated by bright sticks and would erase the signal-to-noise ratio. A threshold can then be moved from one scene to another.

## M1. Smoothing

No shape prior. Gaussian (σ ≈ 1 px), then re-whitening of the filtered map by median / MAD.

Assumption: an interdot is a local excess once high-frequency noise is attenuated.

Limit: any bright excess crosses the threshold. A charge line, which is not labelled, responds like a stick.

## M2. Matched filter

Shape prior. A bank of elongated Gaussian templates with long axis around θ ≈ π/4 (typical slope of interdots on this double quantum dot). Each template has zero mean and unit L2 norm. We keep the maximum over the bank then re-whiten, since the max biases the null distribution.

In white noise the matched filter maximizes the signal-to-noise ratio for a known pattern.

Limit: an orientation outside the bank responds weakly. A trace that is not a short stick (long charge line, other angle) is rejected. This is the goal as long as the true slope stays in the bank.

## M3. Hessian ridge

Local geometry prior with no fixed template. After smoothing we take the most negative eigenvalue of the Hessian (strongest curvature across the trace) and keep only ridges.

Assumption: an interdot is a thin valley and not a blob.

Limit: a charge line is also a ridge. The filter does not separate "interdot" from "line".

## M4. Hysteresis

Same map as M1 plus a spatial coherence prior. The high threshold seeds components. A low threshold (a fraction of the high one) extends them. Blobs smaller than a minimum pixel count are discarded.

Assumption: an interdot is a connected object and not an isolated noise pixel.

Limit: a connected charge line is kept and sometimes thickened by the low threshold.

## M5. Logistic regression (retained variant: `M5_min`, matched + s2)

Learned linear combination of the previous maps. Six per-pixel features: image in σ, fine smoothing, wide smoothing, matched filter, ridge, local standard deviation. Fit by iteratively reweighted least squares (IRLS). About seven weights including the intercept.

Positives are rare (about 0.3 % of pixels). The fit takes all positives and a multiple of randomly drawn negatives. The intercept is then miscalibrated on purpose. The threshold, chosen later on validation only, absorbs this bias.

Role: a learned but still classical floor before committing to a model with more capacity.

## Train / val / test separation

- M5 is fitted on the start of the train set only.
- The threshold is chosen on validation (tolerant pixel F1 at 1 px) then frozen.
- The test set is a fresh seed read once.
- The official mask comes from a thresholded blurred rectangle. It fragments. Strict IoU therefore mixes physics and pixelization (how the generator snaps the rectangle onto the pixel grid). Three readings are used: strict pixel, pixel tolerant to 1 px, object (is the stick located).
- Recall by stick amplitude gives the signal-to-noise limit. Below a few σ none of these methods can see the trace.
