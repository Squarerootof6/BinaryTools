try:
    from binary_tools.gfp import GFP
    from binary_tools.spectrum import SpecTools
except Exception as _exc:
    import warnings as _w
    _w.warn('binary_tools.gfp could not be imported (%s: %s). Only the gaia/corr3d '
           'subroutines are available. Run `python -c "import binary_tools.gfp"` '
           'to see the full traceback.' % (type(_exc).__name__, _exc),
           RuntimeWarning, stacklevel=1)
#from wdtools.neural import CNN
from binary_tools.corr3d import *

__bibtex__ = __citation__ = """@ARTICLE{binary_tools,
       author = {Genghao Liu},
        title = "{A new code for low-resolution spectral identificationof white dwarf binary candidates}",
      journal = { A&A, 690, A29 (2024)},
         year = "2024",
}
"""