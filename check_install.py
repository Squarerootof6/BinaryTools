"""Smoke test: verifies that binary_tools imports and that every available
spectral class generates a finite, physical-looking spectrum.

    python check_install.py

Exit code 0 means the installation is usable.
"""
import os
import sys
import warnings
import traceback

import numpy as np


def main():
    print("=" * 70)
    print("binary_tools self-check")
    print("=" * 70)
    print(f"python     : {sys.version.split()[0]}")
    print(f"numpy      : {np.__version__}")
    try:
        import tensorflow as tf
        print(f"tensorflow : {tf.__version__}")
    except Exception as exc:                      # pragma: no cover
        print(f"tensorflow : IMPORT FAILED ({exc})")
        return 1
    try:
        import h5py
        print(f"h5py       : {h5py.__version__}")
    except Exception as exc:                      # pragma: no cover
        print(f"h5py       : IMPORT FAILED ({exc})")
        return 1

    try:
        import binary_tools
        from binary_tools.gfp import GFP, logteff_logg_to_mass
        print(f"location   : {os.path.dirname(binary_tools.__file__)}")
    except Exception:
        print("\n!! importing binary_tools failed:")
        traceback.print_exc()
        return 1

    ok = True
    print("\n--- mass interpolation ---")
    for t, q in [("DA", (np.log10(5300.0), 8.0)), ("DB", (np.log10(20000.0), 8.0)),
                 ("MS", (np.log10(6000.0), 4.5))]:
        try:
            f = logteff_logg_to_mass(t)
            m = float(np.ravel(f(*q))[0])
            good = np.isfinite(m) and 0.05 < m < 20
            print(f"  {t}: mass = {m:.4f} Msun  {'ok' if good else 'SUSPICIOUS'}")
            ok &= good
        except Exception as exc:
            print(f"  {t}: FAILED ({type(exc).__name__}: {exc})")
            ok = False

    print("\n--- spectrum generation ---")
    # 3rd parameter slot: DA/DB use the radius, MS uses [Fe/H]
    cases = [("DA", 12000.0, 8.0, 8.0, 0.0),
             ("DB", 20000.0, 8.0, 8.0, 0.0),
             ("MS", 6000.0, 4.5, 4.5, 0.0),
             ("sd", 40000.0, 5.5, 5.5, 0.0)]
    for specclass, teff, logg, g, third in cases:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                gfp = GFP(resolution=3, specclass=specclass)
            lam = gfp.lamgrid[specclass]
            try:
                f = logteff_logg_to_mass(specclass)
                mass = float(np.ravel(f(np.log10(teff), logg))[0])
            except FileNotFoundError:
                mass = 0.5                     # nominal, only sets the scale
            radius = float(np.sqrt(mass * 1.989e33 * 6.674e-8 / (10 ** logg)) / 6.957e10)
            third_arg = third if specclass == "MS" else radius
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                spec = gfp.synth_spectrum_sampler(lam, np.array([teff, logg, third_arg, 0.0]))
            spec = np.asarray(spec).ravel()
            finite = np.all(np.isfinite(spec))
            positive = np.all(spec >= 0)
            print(f"  {specclass:3s} {teff:7.0f}K logg={logg}: {len(spec)} px, "
                  f"flux {spec.min():.3g}..{spec.max():.3g}, "
                  f"{'ok' if finite and positive else 'BAD'}")
            ok &= finite and positive
        except RuntimeError as exc:
            # truncated bundled data (currently: subdwarf models)
            print(f"  {specclass:3s} skipped: {str(exc).splitlines()[0][:110]}")
        except Exception as exc:
            print(f"  {specclass:3s} FAILED ({type(exc).__name__}: {exc})")
            ok = False

    print("\n--- demo call (wl 2500-9500, 1000 px, DA 5300 K) ---")
    try:
        import astropy.units as u
        wl = np.linspace(2500, 9500, 1000)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            gfp = GFP(resolution=(wl.max() - wl.min()) / len(wl), specclass="DA")
            gfp.parallax = (4520 * u.pc).to(u.cm).value
            gfp.onlyprofile = True
            f = logteff_logg_to_mass("DA")
            mass = float(np.ravel(f(np.log10(5300.0), 8.0))[0])
            radius = np.sqrt(mass * 1.989e33 * 6.674e-8 / 10 ** 8.0) / 6.957e10
            out = gfp.spectrum_sampler(wl, np.array([[5300.0, 8.0, radius, 0.0]]),
                                       specclass="DA")
        out = np.asarray(out).ravel()
        good = np.all(np.isfinite(out)) and out.max() > 0
        print(f"  output {out.shape}, finite={np.all(np.isfinite(out))}, "
              f"max={out.max():.4g}  {'ok' if good else 'BAD'}")
        for w in caught:
            print(f"  [warning] {w.message}")
        ok &= good
    except Exception:
        print("  FAILED:")
        traceback.print_exc()
        ok = False

    print("\n" + ("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
