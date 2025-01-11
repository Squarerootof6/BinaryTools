try:
    from binary_tools.gfp import GFP
    from binary_tools.spectrum import SpecTools
except:
    print('failed to import GFP. only gaia subroutines will work')
#from wdtools.neural import CNN
from binary_tools.corr3d import *

__bibtex__ = __citation__ = """@ARTICLE{binary_tools,
       author = {Genghao Liu},
        title = "{A new code for low-resolution spectral identificationof white dwarf binary candidates}",
      journal = { A&A, 690, A29 (2024)},
         year = "2024",
}
"""