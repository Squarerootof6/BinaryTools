import scipy
import numpy as np
from distutils import filelist
import emcee
import pandas as pd
from glob import glob
import re
from astropy.io import fits
from scipy import interpolate, integrate
from matplotlib import pyplot as plt
import corner
from lmfit.models import LinearModel, VoigtModel, GaussianModel, LorentzianModel
from bisect import bisect_left
from scipy.interpolate import splev, splrep
from astropy.units import Unit
import os

G = 6.67*1e-11*1e3
SPEED_OF_LIGHT = 2.99792e5
RSUN = 6.955*1e10
MSUN = 1.9891*1e33
LOGGSUN = 4.44
METASUN = 4.5
BOLZMANN = 1.3806505e-23*1e7  # erg/K
PLANCKHC = 6.62606896e-34*1e7*2.99792e10  # erg*s *cm/s

def doppler_shift(wl, fl, dv):
    c = 2.99792458e5
    df = np.sqrt((1 - dv/c)/(1 + dv/c))
    new_wl = wl * df
    new_fl = np.interp(new_wl, wl, fl)
    return new_fl
def wav_to_dvel(wav):
    """根据多普勒公式，把相邻两个波长的波长差转化为对应的速度差"""
    """Converts wavelengths to delta velocities using doppler formula"""
    dvel = (wav[1:] - wav[:-1]) / (wav[1:]) * SPEED_OF_LIGHT
    return dvel
def planck(x, T, RD=1, dv=0):
    hc = PLANCKHC  # erg*s *cm/s
    C = 2.99792e5  # km/s
    k = 1.3806505e-23*1e7  # erg/K
    # x in ang lamda in cm dv in km/s
    lamda = x * 1e-8/np.sqrt((1 - dv/C)/(1 + dv/C))
    res = np.pi*np.array(
        [2*hc*C*1e5*lamda**(-5)*(np.exp(hc*(lamda*k*t)**(-1))-1)**(-1) for t in T])
    res *= 1e-8*1e17*RD**2
    # 1e-17 erg/cm^2/s/Ang
    return res
def gaussian(x, amplitude, mean, stddev, b):
    return -amplitude * np.exp(-((x - mean) / np.sqrt(2) / stddev)**2)+b
def Lorentz(x, a, b, c):
    return a/(b+x**2)+c

def calibrates(cal_path, filelist, filters):
    """
    make calibration for a list of lamost spectra
    cal_path: path where fits located
    filelist: Dataframe with lamost_file_name and photometric magnitude
    filters: pyphot.filter. the filters of photometric magnitude
    return: output a figure and the calibrated (wl,flux), return calibration poly y
    """
    # 打开定标光谱
    rows = int((len(filelist)+0.5)/2)
    col = 2
    fig, ax = plt.subplots(rows, col, figsize=(
        8*col, 2*rows), sharex=True, dpi=100)
    plt.tight_layout(pad=2)
    plt.subplots_adjust(left=0.07, bottom=0.6/rows, hspace=0)
    plt.minorticks_on()
    fig.supxlabel(r'Wavelength $[\AA]$', fontsize=15)
    fig.supylabel(
        r'Relative Flux(lamost) / $10^{-17} erg/s/ cm^2/\AA^1$ (calibrated)', fontsize=15)
    # griABzeropoint = [20.769590,21.361337,21.780351]
    for i in range(len(filelist)):
        item = filelist.iloc[i]
        filename = cal_path + \
            item['lamost_file'].split('/')[-1].split('.')[0]+'.fits'
        file = fits.open(filename)
        data = file[0].data
        gri = np.array(item[['gmag', 'rmag', 'imag']])
        wl = data[2]
        flux = data[0]
        ivar = data[1]
        diff = []
        for band in range(len(filters)):
            # zero = 3.630*10**(-9) # erg s-1 cm-2 A-1
            # 计算星等
            f = filters[band].get_flux(wl, flux, axis=-1)
            mag_spec = -2.5 * np.log10(f.value) - filters[band].AB_zero_mag
            diff.append(gri[band]-mag_spec)
        center_gri = [4686, 6166, 7480]
        poly = np.polyfit(center_gri, diff, deg=2)

        y = 10**(np.polyval(poly, wl)/(-2.5))

        file.close()

        ax[i//2][i % 2].plot(wl, flux)
        ax[i//2][i % 2].plot(wl, flux*y*1e17)
        outputname = filename.split('/')[-1].split('.')[0]
        np.save('./calibration/'+outputname,
                np.vstack((wl, flux*y*1e17, ivar/(y*1e17)**2)))
    fig.savefig('./calibration/total.png')
    try:
        return y
    except:
        return 0


def get_local_filter(file):
    from astropy.io import votable
    from pyphot.astropy import UnitFilter
    DETECTOR_TYPE = ['energy', 'photon']
    table = votable.parse_single_table(file)
    params = {p.name: p.value for p in table.params}
    tab = table.to_table()
    return UnitFilter(tab['Wavelength'].to('nm'), tab['Transmission'], name=params['filterID'].replace('/', '_'), dtype=DETECTOR_TYPE[int(params['DetectorType'])])


def calibrate(data, filters, magnitude):
    """
    calibrate one lamost spectra
    data:fits data
    filters: pyphot.filter. the filters of photometric magnitude(gri)
    magnitude: list, gri magnitude
    """
    gri = np.array(magnitude)
    wl = data[2]*Unit("AA")
    flux = data[0]*Unit("flam")
    ivar = data[1]
    diff = []

    for band in range(len(filters)):
        # zero = 3.630*10**(-9) # erg s-1 cm-2 A-1
        # 计算星等
        f = filters[band].get_flux(wl, flux, axis=-1)
        mag_spec = -2.5 * np.log10(f.value) - filters[band].AB_zero_mag
        diff.append(gri[band]-mag_spec)
    wl = wl.value
    flux = flux.value
    center_gri = [4686, 6166, 7480]
    poly = np.polyfit(center_gri, diff, deg=2)

    y = 10**(np.polyval(poly, wl)/(-2.5))
    return np.vstack((wl, flux*y*1e17, ivar/(y*1e17)**2))


def peak_fit(wl, flux, ivar, center, edge=120, types=-1, resolution=3):
    center = np.float(center)
    mask = np.where(np.logical_and(wl > center-edge, wl < center+edge))
    wl = wl[mask]
    flux = flux[mask]
    ivar = ivar[mask]
    model = GaussianModel()
    params = model.make_params()
    params['center'].set(value=center, min=center-edge,
                         max=center+edge, vary=True)
    params['sigma'].set(value=1, vary=True)
    params['amplitude'].set(value=types, vary=True)
    if len(flux) < 3:
        print(mask, center, flux, params, wl)
    res = model.fit(flux, params, x=wl)
    model_flux = model.eval(res.params, x=wl)
    if res.params['fwhm'].value > 3*resolution and np.abs(model_flux).max() > 3*np.sqrt(np.mean(1/ivar)):

        plt.plot(wl, model_flux, c='r')

        return res
    else:
        # print('the fitting result is to narrow or too small!')
        return 0


def normalize(wl, fl, ivar, sfac=1, k=4, niter=0, crop=None, plot=False, n_order=11, consig=10, splr=False):
    exclude_wl_default = np.array([3790, 3810, 3819, 3855, 3863, 3920, 3930,
                                   4020, 4040, 4180, 4215, 4490, 4662.68, 5062.68, 6314.61, 6814.61])
    outwl = (exclude_wl_default < np.min(wl)) & (
        exclude_wl_default > np.max(wl))
    exclude_wl = exclude_wl_default[~outwl]
    if crop is not None:
        c1 = bisect_left(wl, crop[0])
        c2 = bisect_left(wl, crop[1])

        wl = wl[c1:c2]
        fl = fl[c1:c2]
        ivar = ivar[c1:c2]
    fl_norm = fl
    nivar = ivar
    x = wl

    cont_mask = np.ones(len(wl))

    for ii in range(len(exclude_wl) - 1):
        if ii % 2 != 0:
            continue
        c1 = bisect_left(wl, exclude_wl[ii])
        c2 = bisect_left(wl, exclude_wl[ii + 1])

        cont_mask[c1:c2] = 0
    cont_mask = cont_mask.astype(bool)

    # SFAC to scale rule of thumb smoothing
    s = (len(x) - np.sqrt(2 * len(x))) * sfac
    fl_dum = scipy.ndimage.gaussian_filter1d(fl_norm, consig)
    if splr == True:
        spline = splev(x, splrep(
            x[cont_mask], fl_dum[cont_mask], k=k, s=s, w=np.sqrt(nivar[cont_mask])))
    else:
        poly = np.polyfit(x[cont_mask], fl_dum[cont_mask], n_order)
        spline = np.polyval(poly, x)

    # t = [];

    # for ii,wl in enumerate(exclude_wl):
    #     if ii % 2 == 0:
    #         t.append(wl - 5)
    #     else:
    #         t.append(wl  5)

    # spline = LSQUnivariateSpline(x[cont_mask], fl_norm[cont_mask], t = t, k = 3)(x)

    fl_prev = fl_norm
    fl_norm = fl_prev / spline-1
    nivar = nivar * spline**2

    for n in range(niter):  # Repeat spline fit with reduced smoothing. don't use without testing
        fl_prev = fl_norm
        spline = splev(x, splrep(x[cont_mask], fl_norm[cont_mask], k=k, s=s - 0.1 * n * s,
                                 w=np.sqrt(nivar[cont_mask])))
        fl_norm = fl_norm / spline
        nivar = nivar * spline**2
    fl_diff = fl_norm-spline

    if plot:
        fig, ax = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        ax[0].plot(wl, fl_prev, color='k', label='calibrated data')
        ax[0].plot(wl[cont_mask], fl_dum[cont_mask],
                   color='orange', label='convolution')
        ax[0].plot(wl, spline, color='r', label='poly fit')
        ax[0].set_title('Continuum Fit (iteration %i/%i)' %
                        (niter + 1, niter + 1))
        up = np.quantile(fl_prev, 0.7) + 1
        low = np.quantile(fl_prev, 0.2) - 1

        ax[1].plot(wl[~cont_mask], fl_norm[~cont_mask], color='k')
        ax[1].plot(wl[cont_mask], fl_norm[cont_mask], color='r')

        import atom_line
        #He1 = atom_line(wl.min(), wl.max()).minimal_atomlines['Ca 2']
        #He2 = atom_line(wl.min(), wl.max()).minimal_atomlines['Na 1']
        #top = ax[1].get_yticks()[-1]
        #bottom = ax[1].get_yticks()[0]
        # ax[1].vlines(He1, ymin=bottom, ymax=top, color='grey',
        #             linestyle='--', lw=1, zorder=10)
        # ax[1].vlines(He2, ymin=bottom, ymax=top, color='blue',
        #             linestyle='--', lw=1, zorder=10)
        ax[1].set_title('Normalized Spectrum')
        ax[0].legend()
        plt.show()

    return fl_norm, nivar, fl_diff


class atom_line():
    def __init__(self, wl_min, wl_max):
        scipt_path = '/'.join(os.path.realpath(__file__).split('/')[:-1])
        df = pd.read_csv(scipt_path+'/atomic_lines.csv', sep='\t')[['element', 'wave_A']]
        self.atom_lines = df.loc[df['wave_A'] >
                                 wl_min].loc[df['wave_A'] < wl_max]
        self.minimal_atomlines = {
            'H 1': np.array([6562.76701, 4861.296711, 4340.437554, 4101.710277, 3971.20, 3890.12]),
            "He 1": np.array([4026.2, 4387.9, 4471.5, 4713.1, 4921.9, 5015.7, 5047.7, 5875.6, 6678.2, 7065.2, 7281.4]),
            "He 2": np.array([3203.1, 4685.7, 6559.8]),
            "Ca 1": np.array([4227.92, 4435.688, 5588.757, 6102.722, 6462.566, 6493.780, ]),
            "Ca 2": np.array([7601.304, 8498.01, 8542.1, 8662.12, 3933.6614, 3706.026, 3736.901, 3179.332]),
            "CH": np.array([4300, 4340]),
            "C 2": np.array([4267, 4367, 6578, 6582.8, 7236.2, 7231.1]),
            "C 3": np.array([1909, 4647.4, 4650.2, 4651.4, 5696.0]),
            "N 1": np.array([5200.4, 5197.9, 8680.2, 8683.4, 8686.1, 8718.8, 8711.7, 8703.2, 8216.3, 8210.6, 8200.3, 8242.3, 8223.1, 8184.8, 8188.0, 7468.3, 7442.3, 7423.6]),
            "N 2": np.array([6583.4, 6548.1, 5754.6, 4630.5, 4613.9, 4643.1, 4621.4, 4601.5, 4607.2, 6482.1, 3995.0]),
            "N 3": np.array([4097.3, 4103.4, 4640.6, 4634.2, 4641.9, 6549.7689, 6585.1583]),
            "O 1": np.array([6302.5, 6365.54, 5577.4]),
            "O 2": np.array([3728.8, 7319.0, 7330.2, 4641.8, 4676.2, 4661.6, 4650.8, 4349.4, 4336.9, 4325.8, 4367.0, 4345.6, 4319.6, 4317.1, 3749.5, 3727.3, 3712.8]),
            "O 3": np.array([3047.1, 3035.4, 3059.3, 3043.0, 3023.4, 3024.6, 4364.3782, 4364.44, 4932.6, 4960, 5008.24, 5592.4]),
            "O 4": np.array([3063.5, 3071.7, 3411.8, 3403.6, 3413.7, 3811.4, 3834.2]),
            "Ca2 K": np.array([3933.6]),
            "Ca2 H": np.array([3970]),
            "Ti 2": np.array([3759.3, 3761.3, 3685.2, 3685.2, 4395.0, 4443.8, 4468.5, 4501.3, 3900.5, 3913.5]),
            "Ti 1": np.array([5210.4, 5193.0, 5173.7, 5064.7, 5040.0, 5014.2, 4681.9, 4667.6, 4656.5, 3998.6, 3989.8, 3981.8, 3958.2, 3956.3, 3948.7, 3752.9, 3741.1, 3729.8, 4981.7, 4991.1, 4999.5, 5007.2, 5014.3, 4533.2, 4534.8, 4535.6, 4535.9, 4536.1, 4305.9, 4301.1, 4300.6, 4298.7, 4295.8]),
            "TiO ": np.array([4960, 5160, 6150, 7050, 8430, 8860]),
            "Na 1": np.array([3302.37, 5889, 5895.6, 8194.790]),
            "Na 2": np.array([4455.23, 5414.55]),
            "Mg 1": np.array([3838.3, 3832.3, 3829.4, 4571.1, 5183.6, 5172.7, 5167.3, 8806.8]),
            "Mg 2": np.array([4390.572, 4481, 6346.742]),
            "Si 1": np.array([6527.1]),
            "Si 2": np.array([3856, 3862.6, 3853.7, 6347.1, 6371.4, 4130.9, 4128.1]),
            "Si 3": np.array([4552.7, 4567.9, 4574.8]),
            "CN": np.array([4216]),
            "N 2": np.array([6583.4, 6548.1, 5754.6, 4643.5, 4613.9, 4643.1, 4621.4, 4601.5, 4607.2, 6482.1, 3995.0]),
            "telluric": np.array([6873.2745, 7604.3319, 7635.7488, 9351.7642, 9118.1266, 9136.1689, 9106.2780, 9152.5955, 9099.8150, 8228.2974, 8178.0300]),
            "Xe 2": np.array([6277.82, 6051.15, 6097.59]),
            "Fe 1": np.array([3719.93, 3749.4854, 3820.4253, 3859.9114, 3886.2822, 4045.81, 4383.54, 5270.39, 5576.09, 6173.34]),
            "Fe 2": np.array([4233.17, 4923.93, 5018.44]),
            "Mg 1": np.array([3832.300, 5167.322, 5183.604]),
            "Mg 2": np.array([4481.126, 7896.366, 8234.636, 8734.980]),
            "Ne 1": np.array([6598.9529, 6506.5281, 7245.1666, 6266.4950]),
            "Ne 2": np.array([7522.818]),
            "K 1": np.array([7664.8991, 7698.9645, 4044.14, 4047.21])
        }
        self.special_lines = {
            'H 1': np.array([6562.76701, 4861.296711, 4340.437554, 4101.710277, 3971.20, 3890.12]),
            "He 1": np.array([4026.2, 4471.5, 4921.9, 5875.6, 6678.2, 7065.2]),
            "He 2": np.array([4685.4, ]),
            "Ti 1": np.array([5193.0, 5514.3, 5064.6]),
            "O 1": np.array([6302.5, 6365.54, 5577.4, 7773.5]),
            "Mg 1": np.array([5183.6, 5172.7, 5167.3, 8806.8]),
            "Mg 2": np.array([4390.572, 4481, 6346.742]),
            "Fe 1": np.array([3719.93, 3749.4854, 3820.4253, 3859.9114, 3886.2822, 4045.81, 5270.39, 5576.09, 6173.34]),
            "Fe 2": np.array([4233.17, 4923.93, 5018.44]),
            "Na 1": np.array([5889.95, 5895.924, 8183.256, 8194.790]),
            "Si 1": np.array([6527.1, 6237.3]),
            "Ne 1": np.array([6598.9529, 6506.5281, 5764.4188, 7245.1666, 6266.4950, 6929.4673]),
            "Ne 2": np.array([7522.818]),
            "Xe 2": np.array([6277.82, 6051.15, 6097.59]),
            "Ca 2": np.array([7601.304, 8498.01, 8542.1, 8662.12, 3933.6614, 3706.026, 3736.901, 3179.332]),
            "Ca 1": np.array([4227.92, 4435.688, 5588.757, 6102.722, 6462.566, 6493.780, ]),
            "C 2": np.array([4267, 4367, 6578, 6582.8, 7236.2, 7231.1]),
            "Ti 2": np.array([3759.3, 3761.3, 3685.2, 3685.2, 4395.0, 4443.8, 4468.5, 4501.3, 3900.5, 3913.5]),
            "TiO": np.array([4960, 5160, 6150, 7050, 8430, 8860]),
            "K 1": np.array([7664.8991, 7698.9645, 4044.14, 4047.21]),
            "O 3": np.array([3047.1, 3035.4, 3059.3, 3043.0, 3023.4, 3024.6, 4364.3782, 4364.44, 4932.6, 4960, 5008.24, 5592.4]),
        }

    def showlines(self, wl, lines, ax):
        top = ax.get_yticks()[-2]
        bottom = ax.get_yticks()[0]
        tmp = 0
        for i in lines:
            if type(i)==str:
                He1 = self.special_lines[i]
                mask = np.logical_and(He1 > wl.min(), He1 < wl.max())
                ax.vlines(He1[mask], ymin=bottom, ymax=top, linestyle='--', lw=0.5)
                for loc in He1[mask]:
                    if tmp % 2 == 0:
                        tb = top
                    else:
                        tb = bottom
                    tmp += 1
                    ax.text(loc, tb, i)
            else:
                name,ewl = i
                ewl = np.float32(ewl)
                ax.vlines(ewl, ymin=bottom, ymax=top, linestyle='--', lw=0.5)
                if tmp % 2 == 0:
                    tb = top
                else:
                    tb = bottom
                tmp += 1
                ax.text(ewl, tb, name,fontsize=7)
        return ax

    def find_line_mask(self, wl, flux, ivar, elements, resolution=3, types=-1, fit=True, dense=False, minimal=True, mix=True,ax=None):
        if minimal:
            self.atom_lines = self.special_lines
        
        if ax is None:
            flux_norm, ivar_norm, flux_diff = normalize(wl, flux, ivar, plot=False)
            fig,ax = plt.subplots()
            ax.plot(wl, flux_norm)
        # plt.fill_between(wl,flux_norm-np.sqrt(1/ivar_norm),flux_norm+np.sqrt(1/ivar_norm),alpha=0.4,color='k')

        edge = 100
        top = ax.get_yticks()[-1]
        bottom = ax.get_yticks()[0]
        if mix:
            if minimal:
                atom = self.minimal_atomlines
            else:
                atom = self.atom_lines[np.logical_and(self.atom_lines['wave_A']>wl.min(),self.atom_lines['wave_A']<wl.max())]
            found = []
            st = str(wl.min())
            for inde in atom.index:
                item = atom.loc[inde,'element']
                
                i = np.float32(atom.loc[inde,'wave_A'])
                if i < wl.min() or i > wl.max():
                    continue
                else:
                    found.append([item,i])
                    st+=',{}:{}'.format(item,str(i))
                if fit:
                    res = peak_fit(wl, flux_norm, ivar_norm, i,edge=edge, types=types, resolution=resolution)
                    if res != 0:
                        crop = [res.params['center']-edge,res.params['center']+edge]
                        c1 = bisect_left(wl, crop[0])
                        c2 = bisect_left(wl, crop[1])
                        wl_cut = wl[c1:c2]
                        flux_diff_cut = flux_diff[c1:c2]
                        equiv = integrate.simpson(flux_diff_cut, wl_cut)
                        found.append([res, equiv])
                        plt.vlines(i, bottom, top, lw=0.5, linestyle='--')
                        plt.text(i, top, s=item)

            #ax.vlines(wl.min(), bottom, top, lw=0.5, linestyle='--')
            #ax.text(i, top, s=st)
        else:
            elements = elements.split(',')
            for element in elements:
                if minimal:
                    atom = self.atom_lines[element]
                else:
                    # print(self.atom_lines.loc[self.atom_lines['element']==element])
                    atom = self.atom_lines.loc[self.atom_lines['element'] == element]

                    display(atom)
                    atom = np.array(atom['wave_A'])

                found = []

                for item in range(len(atom)):
                    i = atom[item]
                    if item > 0:
                        if np.abs(i-atom[item-1]) < resolution*3 and not dense:
                            continue
                        elif np.abs(i-atom[item-1]) < resolution and dense:
                            continue
                    if fit:
                        res = peak_fit(wl, flux_norm, ivar_norm, i,
                                       edge=edge, types=types, resolution=resolution)
                        if res != 0:
                            crop = [res.params['center']-edge,
                                    res.params['center']+edge]
                            c1 = bisect_left(wl, crop[0])
                            c2 = bisect_left(wl, crop[1])
                            wl_cut = wl[c1:c2]
                            flux_diff_cut = flux_diff[c1:c2]
                            # print(len(wl_cut),c1,c2,res.params['center'],i)
                            equiv = integrate.simpson(flux_diff_cut, wl_cut)
                            found.append([res, equiv])
                            plt.vlines(i, bottom, top, lw=0.5, linestyle='--')
                            plt.text(i, top, s=element)
                    else:
                        plt.vlines(i, bottom, top, lw=0.5, linestyle='--')
                        plt.text(i, top, s=element)
        return found

if __name__ == "__main__":
    dir = 'D:/Wd/calibration/'
    filelist = ['spec-57373-GAC108N13M1_sp09-177.txt',
                'spec-56301-HD081044N520834M01_sp06-213.txt', 'spec-56650-GAC073N23M1_sp04-126.txt', 'spec-57319-HD213312N533525B01_sp01-084.txt', 'spec-57384-HD014236N482459B01_sp14-176.txt', 'spec-56366-GAC117N27M1_sp11-076']
    for file in filelist:
        data = pd.read_csv(
            dir+file, header=0, delimiter='\t')
        flux = np.array(data['flux'])
        wl = np.array(data['waveobs'])*10
        ivar = np.array(data['err'])
        fl_norm, nivar, fl_diff = normalize(
            wl, flux, ivar, n_order=5, k=5, plot=True)

    denosie(wl, flux, ivar)
    plt.show()
