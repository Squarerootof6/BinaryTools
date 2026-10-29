# Retraining plan: PyTorch + Planck-normalised targets

Status: **not started**. This note records why the current neural DA model
degrades below ~4300 Å at cool temperatures, and the training-target design
proposed to fix it. The shipped model works inside its validated range
(3800–9200 Å, see README); nothing here is required for that use.

## 1. Root cause of the blue-end artefact

The Koester DA grid used for training is 1066 spectra on
2500–9500 Å (0.5 Å sampling), Teff = 5000–80000 K, logg = 6.5–9.5.

The current model is trained on a **per-pixel min–max normalisation over the
whole training set**:

```
y_λ = (F_λ - min_λ) / (max_λ - min_λ)        min/max taken over all 1066 spectra
```

At 2500 Å the maximum is set by the 80000 K model (2.3e10 in the grid units),
so a 5000 K spectrum sits at `y ≈ 1.5e-6`, i.e. pressed onto zero. The network
residual is ~4e-4 in normalised units, and the absolute error is
`residual × (max_λ - min_λ)`. Because the span at 2500 Å is ~80× the span at
8000 Å, the same residual produces an 80× larger flux error:

| λ | normalisation span | error of a 5000 K model |
|---|---|---|
| 2500 Å | 2.31e10 | 18625 % |
| 3000 Å | 1.17e10 | 446 % |
| 3500 Å | 6.58e9 | 47 % |
| 4000 Å | 4.08e9 | 6.3 % |
| 5500 Å | 1.21e9 | 3.7 % |
| 8000 Å | 2.79e8 | 0.36 % |

This is a **target-design** problem, not a capacity, architecture or weight
problem: the same network reproduces the optical range to 0.11 %.

## 2. Per-spectrum normalisation: tried and rejected

Dividing each spectrum by its own continuum/median equalises the dynamic range,
but it also removes the continuum **shape** — and for a pure-hydrogen atmosphere
the shape is very nearly a blackbody, so this is exactly where Teff lives.
Teff then becomes poorly constrained by the fit. Rejected.

The distinction matters: per-spectrum normalisation deletes temperature
information, whereas dividing by a Planck function at the *known input* Teff
does not.

## 3. Proposed target: Planck normalisation

```
y_λ = log10( F_λ / B_λ(Teff) )
```

with `B_λ` the Planck function evaluated at the label temperature. The bulk of
the temperature dependence moves into an analytic factor that is exact, smooth
and cheap to differentiate; the network only has to learn the
atmosphere-versus-blackbody correction, which is a few per cent.

### Measured on the real grid

Dynamic range across the 1066 spectra, per wavelength:

| λ | raw flux span | after `F/B(Teff)` | log-flux std |
|---|---|---|---|
| 2500 Å | 5.88 dex | **1.27 dex** | 1.18 dex |
| 3000 Å | 4.66 | 0.72 | 0.92 |
| 3500 Å | 3.79 | 0.35 | 0.78 |
| 4000 Å | 3.25 | 0.29 | 0.69 |
| 4500 Å | 2.88 | 0.31 | 0.63 |
| 5500 Å | 2.48 | 0.25 | 0.54 |
| 6500 Å | 2.22 | 0.25 | 0.48 |
| 8000 Å | 1.96 | 0.24 | 0.42 |
| 9200 Å | 1.81 | 0.24 | 0.39 |

Temperature sensitivity that survives in the residual
(`d log10(r) / d log10 Teff`, with `r = F/B`), and the scatter left after
removing a smooth (Teff, logg) trend:

| λ | slope, 5–10 kK | slope, 20–80 kK | scatter |
|---|---|---|---|
| 2500 Å | **+3.07** | −0.03 | 0.17 dex |
| 3000 Å | +0.96 | +0.06 | 0.10 |
| 3500 Å | −0.09 | +0.10 | 0.07 |
| 4000 Å | −0.06 | −0.02 | 0.04 |
| 4500 Å | −0.01 | −0.14 | 0.03 |
| 5500 Å | −0.08 | −0.10 | 0.03 |
| 6500 Å | −0.22 | −0.00 | 0.03 |
| 8000 Å | −0.13 | −0.04 | 0.03 |
| 9200 Å | −0.17 | −0.02 | 0.03 |

Reading: for λ ≳ 3500 Å the residual is almost Teff-independent
(≈0.1 dex per dex) and scatters by only 0.03–0.07 dex, so it really is a small
smooth correction. At 2500 Å and cool Teff it is not — see §5.

## 4. Training recipe

```python
# target
B = planck(wl, Teff)                     # same units as the grid fluxes
y = np.log10(F / B)

# loss: weighted least squares in log space
loss = ((net(x) - y) ** 2 * w).mean()    # w = 1 / sigma_lambda**2 from the grid residuals

# reconstruct
F = B * 10 ** net(x)
```

Fitting derivatives stay well conditioned because part of the temperature
dependence is analytic:

```
dF/dTeff = (dB/dTeff) * r + B * (dr/dTeff)
```

with `r = 10**y` and `dB/dTeff` available in closed form.

## 5. Open issues before writing code

1. **2500–3200 Å, cool stars.** The residual slope is +3.07 per dex, i.e. even
   after Planck normalisation this window stays hard for Teff < 8000 K. Options:
   down-weight λ < 3200 Å, or train separate cool (< 8000 K) and hot networks,
   or simply keep the blue limit in the documentation and stop pretending.
2. **Absolute flux scale.** Verify that every Koester model is computed for the
   same radius/reference before forming `F/B`; if the models are individually
   normalised, restore the physical scale first, otherwise `r` carries a
   per-model constant that is degenerate with radius.
3. **logg degeneracy not yet measured.** Confirm `d log r / d logg` is not
   strongly correlated with `d log r / d Teff` over the grid; if it is, the
   network needs both labels in the target weighting, not just Teff.
4. **Validation set.** Keep the paper's range (SDSS SEGUE LRS, 3800–9200 Å) and
   report the blue-end behaviour separately rather than averaging it away.

## 6. Porting to PyTorch is independent of retraining

The current network is a small MLP, `2 -> 256 -> 256 -> 256 -> 14000` with
LeakyReLU(0.01), saved by Keras 2.4.0. The weights can be copied straight into
`torch.nn.Linear` and validated against the current TF/Keras output (agreement
to <1e-6 in flux); that gives a PyTorch model without waiting for a retrain.
Retraining is only needed for the new target, not for the framework change.

Note the current release loads those weights with `h5py` because
`keras.load_weights()` under Keras 2.14 silently mis-reads a Keras 2.4 file
(correlation −0.83, 46 % pixel error). Any port must be validated against a
known spectrum, not just "it runs".

## 7. Acceptance criteria for the retrained model

- optical range (4000–9000 Å): pixel error ≤ 0.5 % for 5000–9000 K
  (current release: 1.86 % at 5000–5250 K, 0.5 % at 6000–7000 K)
- 4000 Å at 5000 K: ≤ 2 % (current: 6.3 %)
- 3000 Å at 5000 K: ≤ 10 % (current: 446 %) — the main goal
- `check_install.py` exits 0, and the model still returns finite, non-negative
  flux across the grid
