# BinaryTools

Spectral-fitting toolkit for white-dwarf and main-sequence binaries
(DA/DB/MS + subdwarf grids, artificial-neural-network spectrum generators).
Paper: [Liu et al. 2024, A&A 690, A29](https://doi.org/10.1051/0004-6361/202449775).

## Requirements

Validated combination — use exactly this one if you want a guaranteed-working setup:

| package | version | note |
|---|---|---|
| Python | 3.9 – 3.11 | TensorFlow 2.14 has no 3.12 wheels |
| NumPy | `>=1.23.5, <2.0` | TF 2.14 is built against NumPy < 2 |
| TensorFlow | `==2.14.0` | |
| Keras | `==2.14.0` | bundled with TF 2.14 |
| h5py | `>=3.8, <4` | required for the legacy weight loader |

Everything (including `astropy`, `scipy`, `matplotlib`, `pandas`, `lmfit`,
`emcee`, `corner`) is listed in `requirements.txt`.

> **Note on TensorFlow/Keras versions.** The bundled `.h5` weight files were
> written by Keras 2.4. Loading them with `model.load_weights()` in Keras ≥ 2.11
> — and in Keras 3 (TF ≥ 2.16) — silently produces garbage spectra. `gfp.py`
> therefore ships its own loader (`gfp.load_legacy_weights`) that reads the
> arrays with h5py and calls `set_weights()`; that path is version independent,
> so newer TensorFlow works too. TF 2.14 is simply the combination we test.

## Installation

```bash
git clone https://github.com/Squarerootof6/BinaryTools.git
cd BinaryTools

python -m venv .venv && source .venv/bin/activate     # or a conda env
pip install -r requirements.txt
pip install -e .                                      # editable: code changes take effect
```

If the clone was made with Git LFS enabled, fetch the large model files first
(`git lfs install && git lfs pull`); otherwise the `.h5`/`.npy` model files will
be pointers and the package will fail with a "truncated data" error.

## Check the installation

```bash
python check_install.py
```

It prints the versions in use and generates one spectrum per spectral class,
verifying that the fluxes are finite and non-negative. Exit code 0 = usable.
A `skipped: scaling table ...` line means the bundled data for that class is
truncated (currently the subdwarf `sd` grid in some distributions).

## Usage

See `demo.ipynb` for a tutorial. Minimal example:

```python
import numpy as np, astropy.units as u, astropy.constants as c
from binary_tools.gfp import GFP, logteff_logg_to_mass

wl = np.linspace(2500, 9500, 1000)
specclass = 'DA'
gfp = GFP(resolution=(wl.max() - wl.min()) / len(wl), specclass=specclass)
gfp.parallax = (4520 * u.pc).to(u.cm).value
gfp.onlyprofile = True

Teff, logg, rv = 5300.0, 8.0, 0.0
f = logteff_logg_to_mass(specclass)
mass = float(f(np.log10(Teff), logg))
radius = np.sqrt(mass * u.M_sun * c.G / (10**logg * u.cm / u.s**2)).to(u.R_sun).value

flux = gfp.spectrum_sampler(wl, np.array([[Teff, logg, radius, rv]]), specclass=specclass)
```

`logteff_logg_to_mass` accepts three arguments for main-sequence stars
(`teff, logg, feh=0.0`), two for DA/DB.

## Known limitations

**Cool DA models and the blue end.** The spectrum generators are trained on
per-pixel min–max normalised flux over a 5000–80000 K grid. The per-pixel
extrema are set by the hot models, so a spectrum near 5000 K is compressed
towards zero in normalised space; the network's normalised residual
(~4×10⁻⁴) is then multiplied by the normalisation span, which is largest in
the blue (2.3×10¹⁰ at 2500 Å vs 2.8×10⁸ at 8000 Å).

Measured relative error at the same wavelength (3000 Å): **0.2 % at
Teff = 12000 K but 446 % at Teff = 5000 K**.

| Teff | reliable from |
|---|---|
| ≳ 10000 K | ~3000 Å |
| 8000–10000 K | ~3500 Å |
| 6000–8000 K | ~4000 Å |
| 5000–6000 K | ~4300 Å |

Optical-band errors are 1–7 % for the coolest models; blue-end errors reach
10²–10³ %. `GFP` emits a `RuntimeWarning` when you request such a spectrum —
for cool objects blueward of 4000 Å use the raw Koester grids instead.

## Notes for developers

**What the package ships.** Only model-related files: the network weights
(`*_p01.h5`), the wavelength grids (`*_wl.npy`), the per-pixel scaling tables
(`*spec_sc.txt`) and the mass–interpolation tables (`logteff_logg_to_mass_*.npz`).
Training sets and intermediate training artefacts (`*_x.npy`, `*_y.npy`,
`*_all.h5`, `*_normNN.h5`, `MSEVTK_*.npy`, the old `.pkl` interpolators) are
kept in `training_data/` next to the package and are deliberately **not**
packaged or distributed (`MANIFEST.in` prunes them).

* The `.npz` mass tables hold only the interpolation nodes; the triangulation is
  rebuilt on load. The old `.pkl` files were pickled `LinearNDInterpolator`
  objects (244 MB for MS); they remain in `training_data/` for reference.
* The training array `DA_y.npy` (1066 × 14000) is exactly the Koester grid
  resampled onto 2500–9500 Å in 0.5 Å steps, so it can be regenerated from
  `gs:~/Wd/lamostmodel/koesterDA` if needed.
* The subdwarf (`sd`) model is included, but its training array (`sdob_y.npy`,
  ~12 GB) is not; `GFP(specclass='sd')` works for spectrum generation as long as
  `sdob_p01.h5`, `sdob_wl.npy` and `sdobspec_sc.txt` are present.
