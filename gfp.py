from binary_tools.corr3d import *
from binary_tools.spectrum import SpecTools
import numpy as np
import scipy
import matplotlib.pyplot as plt
import datetime
import glob
import pickle
import sys
import os
import emcee
import corner
from bisect import bisect_left
import warnings
import lmfit
import pandas as pd
import tensorflow.compat.v1 as tf
import multiprocessing
import gc
from tensorflow.keras import backend as K
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *

#tf.disable_v2_behavior()  # 启用TensorFlow 1.x兼容模式

from numpy.polynomial.polynomial import polyfit, polyval
from numpy.polynomial.chebyshev import chebfit, chebval
from scipy.interpolate import splev, splrep, interp1d
import assistlgh as alg
from assistlgh.spectra import calibrate, planck, gaussian, atom_line
from assistlgh.calculate import grad5
path = os.path.abspath(__file__)
dir_path = os.path.dirname(path)
sys.path.append(dir_path)

os.environ["CUDA_VISIBLE_DEVICES"]="-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
print(tf.config.list_logical_devices())



halpha = np.float64(6562.81)
hbeta = 4862.68
hgamma = 4341.68
hdelta = 4102.89
planck_h = 6.62607004e-34
speed_light = 299792458
k_B = 1.38064852e-23
RSUN = 6.955*1e10  # cm
G = 6.67*1e-11*1e3  # cm^3/(g s^2)
MSUN = 1.9891*1e33  # g


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

class GFP:

    """ Generative Fitting Pipeline.

    """
    
    def __init__(self, resolution=3, specclass='DA',allmodel=False):
        '''
        Initializes class.

        Parameters
        ---------
        resolution : float
                Spectral resolution of the observed spectrum, in Angstroms (sigma). The synthetic spectra are convolved with this Gaussian kernel before fitting.
        specclass : str ['DA', 'DB']
                Specifies whether to fit hydrogen-rich (DA) or helium-rich (DB) atmospheric models. DB atmospheric models are not publicly available at this time.
        '''
    
        self.res_ang = resolution
        self.resolution = {}
        self.model = {}
        self.lamgrid = {}
        self.Tgrid = {}
        self.ggrid = {}
        if allmodel:
            for types in ['DA','DB','MS','sd']:
                self.model_region(types, resolution)
        else:
            self.mark_specclass(specclass)
            for types in self.specclass:
                self.model_region(types, resolution)

        self.spec_scaled = {}
        self.spec_scaled['DA'] = np.loadtxt(
            dir_path + '/models/neural_gen/DAspec_sc.txt')
        self.spec_scaled['MS'] = np.loadtxt(
            dir_path + '/models/neural_gen/MSspec_sc.txt')
        self.spec_scaled['DB'] = np.loadtxt(
            dir_path + '/models/neural_gen/DBspec_sc.txt')
        self.spec_scaled['sd'] = np.loadtxt(
            dir_path + '/models/neural_gen/sdobspec_sc.txt')
        self.exclude_wl_default = np.array(
            [3790, 3810, 3819, 3855, 3863, 3920, 3930, 4020, 4040, 4180, 4215, 4490, 4662.68, 5062.68, 6314.61, 6814.61])
        self.exclude_wl = self.exclude_wl_default
        self.norm_kw = {"plot": False}
        self.cont_fixed = False
        self.rv_fixed = False
        self.rv = 0
        self.centroid_dict = dict(alpha=np.float64(6562.8), beta=np.float64(
            4862.68), gamma=4341.68, delta=4102.89, eps=3971.20, h8=3890.12)
        self.distance_dict = dict(
            alpha=125, beta=125, gamma=85, delta=70, eps=45, h8=30)
        self.norm_plot = None
        self.lineprofile_plot = None
        self.sp = SpecTools()
    def mark_specclass(self,specclass):
        classes = specclass.split('+')
        self.specclass = [*classes]
        self.isbinary = len(classes)==2
    def model_region(self, types, resolution):
        # if types == 'DA':
        #     self.Tgrid[types] = [6500, 40000]
        #     self.ggrid[types] = [7, 10]
        #     self.H_DA = 128
        #     self.lamgrid[types] = np.loadtxt(
        #         dir_path + '/models/neural_gen/DA_lamgrid.txt')
        #     self.model[types] = self.generator(
        #         self.H_DA, len(self.lamgrid[types]))
        #     self.model[types].load_weights(
        #         dir_path + '/models/neural_gen/DA_normNN.h5')
        #     self.spec_min, self.spec_max = np.loadtxt(
        #         dir_path + '/models/neural_gen/DA_specsc.txt')
        #     pix_per_a = len(self.lamgrid[types]) / \
        #         (self.lamgrid[types][-1] - self.lamgrid[types][0])
        #     self.resolution[types] = resolution * pix_per_a
        if types == "DA":
            self.Tgrid[types] = [5000, 80000]
            self.ggrid[types] = [6.5, 9.5]
            self.H_MS = 256
            self.lamgrid[types] = np.load(
                dir_path + '/models/neural_gen/DA_wl.npy')
            self.model[types] = self.generator(
                self.H_MS, len(self.lamgrid[types]))
            self.model[types].load_weights(
                dir_path + '/models/neural_gen/DA_p01.h5')
            self.model[types] =  tf.function(lambda x: self.model[types](x)).get_concrete_function(tf.TensorSpec(self.model[types].inputs[0].shape,self.model[types].inputs[0].dtype))
            pix_per_a = len(self.lamgrid[types]) / \
                (self.lamgrid[types][-1] - self.lamgrid[types][0])
            self.resolution[types] = resolution * pix_per_a

        elif types == "MS":
            self.Tgrid[types] = [3000, 26000]
            self.ggrid[types] = [-0.5, 5.5]
            self.H_MS = 256
            self.lamgrid[types] = np.load(
                dir_path + '/models/neural_gen/MS_wl.npy')
            self.model[types] = self.MS_generator(
                self.H_MS, len(self.lamgrid[types]))
            self.model[types].load_weights(
                dir_path + '/models/neural_gen/MS_p01.h5')
            self.model[types] = tf.function(lambda x: self.model[types](x)).get_concrete_function(tf.TensorSpec(self.model[types].inputs[0].shape,self.model[types].inputs[0].dtype))
            pix_per_a = len(self.lamgrid[types]) / \
                (self.lamgrid[types][-1] - self.lamgrid[types][0])
            self.resolution[types] = resolution * pix_per_a
        elif types == "DB":
            self.Tgrid[types] = [8000, 40000]
            self.ggrid[types] = [7.0, 9.5]
            self.H_MS = 256
            self.lamgrid[types] = np.load(
                dir_path + '/models/neural_gen/DB_wl.npy')
            self.model[types] = self.generator(
                self.H_MS, len(self.lamgrid[types]))
            self.model[types].load_weights(
                dir_path + '/models/neural_gen/DB_p01.h5')
            self.model[types] =  tf.function(lambda x: self.model[types](x)).get_concrete_function(tf.TensorSpec(self.model[types].inputs[0].shape,self.model[types].inputs[0].dtype))
            pix_per_a = len(self.lamgrid[types]) / \
                (self.lamgrid[types][-1] - self.lamgrid[types][0])
            self.resolution[types] = resolution * pix_per_a
        elif types == "sd":
            self.Tgrid[types] = [30000, 70000]
            self.ggrid[types] = [4.5, 7.0]
            self.H_MS = 256
            self.lamgrid[types] = np.load(
                dir_path + '/models/neural_gen/sdob_wl.npy')
            self.model[types] = self.generator(
                self.H_MS, len(self.lamgrid[types]))
            self.model[types].load_weights(
                dir_path + '/models/neural_gen/sdob_p01.h5')
            self.model[types] =  tf.function(lambda x: self.model[types](x)).get_concrete_function(tf.TensorSpec(self.model[types].inputs[0].shape,self.model[types].inputs[0].dtype))

            pix_per_a = len(self.lamgrid[types]) / \
                (self.lamgrid[types][-1] - self.lamgrid[types][0])
            self.resolution[types] = resolution * pix_per_a

    def label_sc(self, label_array, types):
        """
        Label scaler to transform Teff and logg to [0,1] interval based on preset bounds.

        Parameters
        ---------
        label_array : array
                Unscaled array with Teff in the first column and logg in the second column
        Returns
        -------
                array
                        Scaled array
        """
        label_array = tf.cast(label_array, tf.float32)

        teffs = label_array[:, 0]
        loggs = label_array[:, 1]
        teffs = (teffs - self.Tgrid[types][0]) / \
            (self.Tgrid[types][1] - self.Tgrid[types][0])
        loggs = (loggs - self.ggrid[types][0]) / \
            (self.ggrid[types][1]-self.ggrid[types][0])
        if types == "MS" and label_array.shape[1] >= 3:
            meta = label_array[:, 2]
            meta = (meta+1.3)/(0.2+1.3)
            if label_array.shape[1] == 4:
                afe = label_array[:, 3]
                afe = afe/0.4
                return np.vstack((teffs, loggs, meta, afe)).T
            return np.vstack((teffs, loggs, meta)).T
        return np.vstack((teffs, loggs)).T

    def inv_label_sc(self, label_array, types):
        """
        Inverse label scaler to transform Teff and logg from [0,1] to original scale based on preset bounds.

        Parameters
        ---------
        label_array : array
                Scaled array with Teff in the first column and logg in the second column
        Returns
        -------
                array
                        Unscaled array
        """
        teffs = label_array[:, 0]
        loggs = label_array[:, 1]
        teffs = (
            teffs * (self.Tgrid[types][1] - self.Tgrid[types][0])) + self.Tgrid[types][0]
        loggs = (
            loggs * (self.ggrid[types][1] - self.ggrid[types][0])) + self.Tgrid[types][0]
        if types == 'MS' and label_array.shape[1] >= 3:
            meta = label_array[:, 2]
            meta = meta*(0.2+1.3)-1.3
            if label_array.shape[1] == 4:
                afe = label_array[:, 3]
                afe = afe*0.4
                return np.vstack((teffs, loggs, meta, afe)).T
            return np.vstack((teffs, loggs, meta)).T
        return np.vstack((teffs, loggs)).T

    def spec_sc(self, spec):
        spec_min, spec_max = self.spec_scaled[self.specclass]
        return (spec - spec_min) / (spec_max - spec_min)

    def inv_spec_sc(self, spec):
        spec_min, spec_max = self.spec_scaled[self.specclass]
        return spec * (spec_max - spec_min) + spec_min

    def MS_generator(self, H, n_pix):
        x = Input(shape=4, name='Input')
        y = Dense(H, activation=LeakyReLU(alpha=0.01))(x)
        y = Dense(H, activation=LeakyReLU(alpha=0.01))(y)
        y = Dense(H, activation=LeakyReLU(alpha=0.01))(y)

        # y = Dropout(self.dropout)(y, training = True)

        out = Dense(n_pix,
                    activation='linear', name='Output')(y)

        model = Model(inputs=x, outputs=out)
        model.compile(optimizer=Adamax(learning_rate=1e-3), loss='mse')
        model.trainable = False
        #run_model = tf.function(lambda x: model(x))
        #func = run_model.get_concrete_function(tf.TensorSpec(model.inputs[0].shape,model.inputs[0].dtype))

        return model
    def generator(self, H, n_pix):
        x = Input(shape=(2,))
        y = Dense(H, activation='relu')(x)
        y = Dense(H, activation='relu')(y)
        y = Dense(H, activation='relu')(y)
        out = Dense(n_pix, activation='linear')(y)

        model = Model(inputs=x, outputs=out)
        model.compile(optimizer=Adamax(learning_rate=1e-3), loss='mse',
                      metrics=['mae'])
        model.trainable = False
        # tf.function 封装
        #run_model = tf.function(lambda x: model(x))
        #func = run_model.get_concrete_function(tf.TensorSpec(model.inputs[0].shape,model.inputs[0].dtype))

        return model
    def synth_spectrum_sampler(self, wl, prmsprv, specclass=None, inv=False):
        """
        Generates synthetic spectra from labels using the neural network, translated by some radial velocity. These are _not_ interpolated onto the requested wavelength grid;
        The interpolation is performed only one time after the Gaussian convolution with the instrument resolution in `GFP.spectrum_sampler`. Use `GFP.spectrum_sampler` in most cases.

        Parameters
        ----------
        wl : array
                Array of spectral wavelengths (included for completeness, not used by this function)
        teff : float
                Effective surface temperature of sampled spectrum
        logg : float
                log surface gravity of sampled spectrum (cgs)
        rv : float
                Radial velocity (redshift) of sampled spectrum in km/s
        specclass : str ['DA', 'DB']
                Whether to use hydrogen-rich (DA) or helium-rich (DB) atmospheric models. If None, uses default.

        Returns
        -------
                array
                        Synthetic spectrum with desired parameters, interpolated onto the supplied wavelength grid.
        """
        prms = prmsprv[:-1]
        rv = prmsprv[-1]
        # radius = prmsprv[-2]
        spec_min, spec_max = self.spec_scaled[specclass]
        if specclass is None:
            specclass = self.specclass
        if specclass =='MS':
            prms=np.append(prms,0)
        label = self.label_sc(np.array(prms.reshape(1, -1)), specclass)
        #synth = self.model[specclass].predict_on_batch(label)[0]
        synth = np.array(self.model[specclass](tf.constant(label))[0])
        synth = synth*(spec_max - spec_min)+spec_min
        synth = self.sp.doppler_shift(self.lamgrid[specclass], synth, rv)
        if specclass == 'DA':
            # 4x erg/cm^2/s/A -> 1e-17 erg/cm^2/s/A
            synth = (np.ravel(synth).astype('float64'))*1e17
        elif specclass == 'DB':
            # 4x erg/cm^2/s/A -> 1e-17 erg/cm^2/s/A
            synth = (np.ravel(synth).astype('float64'))*1e9*np.pi
        elif specclass == 'MS':
            synth = (np.ravel(synth).astype('float64'))*1e17*np.pi
        elif specclass == 'sd':
            synth = (np.ravel(synth).astype('float64')) * 1e9  
            # erg/cm^2/s/cm -> 1e-17 erg/cm^2/s/A
        else:
            print('not single specclass inputed!')
        #K.clear_session()
        #tf.compat.v1.reset_default_graph()
        return synth.copy()
    def spectrum_sampler(self, wl, prms, *polyargs, parallax=None, specclass=None, radius=None,three = False):
        """
        Wrapper function that talks to the generative neural network in scaled units, and also performs the Gaussian convolution to instrument resolution.

        Parameters
        ----------
        wl : array
                Array of spectral wavelengths on which to generate the synthetic spectrum
        teff : float
                Effective surface temperature of sampled spectrum
        logg : float
                log surface gravity of sampled spectrum (cgs)
        polyargs : float, optional
                All subsequent positional arguments are assumed to be coefficients for the additive Chebyshev polynomial. If none are provided,
                no polynomial is added to the model spectrum.
        specclass : str, optional
                Whether to use hydrogen-rich (DA) or helium-rich (DB) atmospheric models. If none, reverts to default.
        Returns
        -------
                array
                        Synthetic spectrum with desired parameters, interpolated onto the supplied wavelength grid and convolved with the instrument resolution.
        """

        if specclass is None:
            specclass = self.specclass
        if parallax is None:
            parallax = self.parallax
        # print('specclass',specclass)
        if type(specclass) != list:
            specclass = [specclass]
        if radius is None:
            radius = []
            for i in range(len(prms)):
                radius.append(prms[i][-2])
        for i in range(len(prms)):
            if self.rv_fixed:
                prms[i][-1] = self.rv[i]
        synth = np.zeros_like(wl)
        synth_list = []
        for i in range(len(specclass)):
            synth_tmp = self.synth_spectrum_sampler(self.lamgrid[specclass[i]], prms[i], specclass[i])
            synth_tmp = scipy.ndimage.gaussian_filter1d(synth_tmp, self.resolution[specclass[i]])
            func_tmp = interp1d(self.lamgrid[specclass[i]], synth_tmp, fill_value=0., bounds_error=False)
            synth_tmp = func_tmp(wl)*(radius[i]*RSUN/parallax)**2
            if three:
                synth_list.append(synth_tmp.copy())
            else:
                synth += synth_tmp

        if self.cont_fixed:
            dummy_ivar = 1 / np.repeat(0.001, len(wl))**2
            nanwhere = np.isnan(synth)
            dummy_ivar[nanwhere] = 0
            synth[nanwhere] = 0
            # Use default KW from function
            wl, synth, _ = self.spline_norm_DA(
                wl, synth, dummy_ivar, kwargs=self.norm_kw)
            
            synth[nanwhere] = np.nan
            
        if len(polyargs) > 0:
            synth = synth * \
                chebval(2 * (wl - wl.min()) /
                        (wl.max() - wl.min()) - 1, polyargs)
        if three:
            return synth_list
        else:
            return synth.copy()
    def spline_norm_DA(self, wl, fl, ivar, kwargs=dict(k=3, sfac=1, niter=3), crop=None):  # SETS DEFAULT KW
        """
        Masks out Balmer lines, fits a smoothing spline to the continuum, and returns a continuum-normalized spectrum

        Parameters
        ----------
        wl : array
                Array of observed spectral wavelengths.
        fl : array
                Array of observed spectral fluxes.
        ivar : array
                Array of observed inverse-variance.
        kwargs : dict, optional
                Keyword arguments that are passed to the spline normalization function
        crop : tuple, optional
                Defines a start and end wavelength to crop the spectrum to before continuum-normalization.

        Returns
        -------
                tuple
                        If crop is None, returns a 2-tuple of (normalized_flux, normalized_ivar). If a crop region is provided,
                        then returns a 3-tuple of (cropped_wavelength, cropped_normalized_flux, cropped_normalized_ivar).
        """

        if crop is not None:
            c1 = bisect_left(wl, crop[0])
            c2 = bisect_left(wl, crop[1])

            wl = wl[c1:c2]
            fl = fl[c1:c2]
            ivar = ivar[c1:c2]
        # linear = np.polyval(np.polyfit(wl, fl, 2), wl)
        # fl = fl / linear
        # ivar = ivar * linear**2 # Initial divide by quadratic continuum

        try:
            if self.norm_plot != None:
                fl_norm, ivar_norm = self.sp.spline_norm(
                    wl, fl, ivar, self.exclude_wl, self.norm_plot[1], **kwargs)
            else:
                fl_norm, ivar_norm = self.sp.spline_norm(
                    wl, fl, ivar, self.exclude_wl,None, **kwargs)
        except:
            print('spline normalization failed... returning zeros')
            fl_norm, ivar_norm = 0*fl, 0*ivar
            raise
        return wl.copy(), fl_norm.copy(), ivar_norm.copy()
    def mass_err(self,evtr,para,cov_matrix,h=1e-6):
        res = np.array(grad5(evtr,para,h))
        cov_matrix = np.array(cov_matrix)
        return np.sqrt(np.dot(res.T,np.dot(cov_matrix,res)))
    def decide_mass(self,TG2M,mle,specclass,cov_matrix=None,err = True):
        res = []
        for i in range(len(specclass)):
            e_mass1 = None
            if specclass[i] == 'MS':
                para = [np.log10(mle[i][0]), mle[i][1], mle[i][2]]
                mass1 = TG2M[i](*para)
                if err:
                    cov_matrix[i][0,:]/=(np.log(10)*mle[i][0])
                    cov_matrix[i][:,0]/=(np.log(10)*mle[i][0])
                    e_mass1 = self.mass_err(TG2M[i],para,cov_matrix[i])
            elif specclass[i] == 'sd':
                mass1 = 10**(mle[i][1])*(mle[i][-2]*RSUN)**2/G/MSUN
                if err:
                    e_mass1 = np.sqrt((mass1*np.log(10)*cov_matrix[i][1][1])**2+(2*mass1/mle[i][-2][-2]*cov_matrix[i][-2][-2])**2)
            else:
                para = [np.log10(mle[i][0]), mle[i][1]]
                mass1 = TG2M[i](*para)
                if err:
                    cov_matrix[i][0,:]/=(np.log(10)*mle[i][0])
                    cov_matrix[i][:,0]/=(np.log(10)*mle[i][0])
                    e_mass1 = self.mass_err(TG2M[i],para,cov_matrix[i])
            if err:
                res.append([np.float64(mass1),e_mass1])
            else:
                res.extend([mass1])
        return res
    def fit_spectrum(self, wl, fl, filters, gri, TG2M, ivar=None, parallax=0, prior_teff=None, mcmc=False, onlyprofile=False, polyorder=0,
                     norm_kw=dict(k=1, sfac=0.5, niter=0),
                     nwalkers=25, burn=25, ndraws=25, sampler_kw=dict(threads=1,moves=None), progress=True,
                     plot_init=False, make_plot=True, plot_corner=False, plot_corner_full=False, plot_trace=False,  savename=None, given_plot=None,crop=None, init_rv=None,
                     verbose=True,lmfit_kw=dict(method='leastsq', epsfcn=0.1, calc_covar=True),
                     rv_kw=dict(plot=False, distance=100, nmodel=2, edge=15),
                     nteff=3,  rv_line='alpha', corr_3d=False, cali=True, addition_elements=None,fig_path = './',output_path='./',init_paras = None):
        """
        Main fitting routine, takes a continuum-normalized spectrum and fits it with MCMC to recover steller labels.

        Parameters
        ----------
        wl : array
                Array of observed spectral wavelengths
        fl : array
                Array of observed spectral fluxes, continuum-normalized. We recommend using the included `normalize_balmer` function from `wdtools.spectrum` to normalize DA spectra,
                and the generic `continuum_normalize` function for DB spectra.
        ivar : array
                Array of observed inverse-variance for uncertainty estimation. If this is not available, use `ivar = None` to infer a constant inverse variance mask using a second-order
                beta-sigma algorithm. In this case, since the errors are approximated, the chi-square likelihood may be inexact - treat returned uncertainties with caution.
        prior_teff : tuple, optional
                Tuple of (mean, sigma) to define a Gaussian prior on the effective temperature parameter. This is especially useful if there is strong prior knowledge of temperature
                from photometry. If not provided, a flat prior is used.
        mcmc : bool, optional
                Whether to run MCMC, or simply return the errors estimated by LMFIT
        onlyprofile : bool, optional
                Whether to fit only the spectral lines with normalized spectrum.
        polyorder : int, optional
                Order of additive Chebyshev polynomial during the fitting process. Can usually leave this to zero unless the normalization is really bad.
        norm_kw : dict, optional
                Dictionary of keyword arguments that are passed to the spline normalization routine.
        nwalkers : int, optional
                Number of independent MCMC 'walkers' that will explore the parameter space
        burn : int, optional
                Number of steps to run and discard at the start of sampling to 'burn-in' the posterior parameter distribution. If intitializing from
                a high-probability point, keep this value high to avoid under-estimating uncertainties.
        ndraws : int, optional
                Number of 'production' steps after the burn-in. The final number of posterior samples will be nwalkers * ndraws.
        threads : int, optional
                Number of threads for distributed sampling.
        progress : bool, optional
                Whether to show a progress bar during the MCMC sampling.
        plot_init : bool, optional
                Whether to plot the continuum-normalization routine
        make_plot: bool, optional
                If True, produces a plot of the best-fit synthetic spectrum over the observed spectrum.
        plot_corner : bool, optional
                Makes a corner plot of the fitted stellar labels
        plot_corner_full : bool, optional
                Makes a corner plot of all sampled parameters, the stellar labels plus any Chebyshev coefficients if polyorder > 0
        plot_trace: bool, optiomal
                If True, plots the trace of posterior samples of each parameter for the production steps. Can be used to visually determine the quality of mixing of
                the chains, and ascertain if a longer burn-in is required.
        savename : str, optional
                If provided, the corner plot and best-fit plot will be saved as PDFs in the working folder.
        DA : bool, optional
                Whether the star is a DA white dwarf or not. As of now, this must be set to True.
        crop : tuple, optional
                The region to crop the supplied spectrum before proceeding with the fit. Can be used to exclude low-SN regions at the edge of the spectrum.
        verbose : bool, optional
                If True, the routine prints several progress statements to the terminal.
        lines : array, optional
                List of Balmer lines to utilize in the fit. Defaults to all from H-alpha to H8.
        lmfit_kw : dict, optional
                Dictionary of keyword arguments to the LMFIT solver
        rv_kw : dict, optional
                Dictionary of keyword arguments to the RV fitting routine
        nteff : int, optional
                Number of equidistant temperatures to try as initialization points for the minimization routine.
        rv_line : str, optional
                Which Balmer line to use for the radial velocity fit. We recommend 'alpha'.
        corr_3d : bool, optional
                If True, applies 3D corrections from Tremblay et al. (2013) to stellar parameters before returning them.

        Returns
        -------
                array
                        Returns the fitted stellar labels along with a reduced chi-square statistic with the format: [[labels], [e_labels], redchi]. If polyorder > 0,
                        then the returned arrays include the Chebyshev coefficients. The radial velocity (and RV error) are always the last elements in the array, so if
                        polyorder > 0, the label array will have temperature, surface gravity, the Chebyshev coefficients, and then RV.
        """
        self.cont_fixed = False
        self.rv_fixed = False
        self.parallax = parallax
        nans = np.logical_and(np.isnan(fl), np.isnan(wl), np.isnan(ivar))

        if np.sum(nans) > 0:

            print('NaN detected in input... removing them...')

            wl = wl[~nans]
            fl = fl[~nans]
            ivar = ivar[~nans]

        if ivar is None:  # REPLACE THIS WITH YOUR OWN FUNCTION TO ESTIMATE VARIANCE
            print('Please provide an IVAR array')
            # raise
        if cali:
            wl0, fl0, ivar0 = calibrate(
                np.vstack((fl, ivar, wl)), filters, gri)
            kick = fl0 == 0
            if verbose:
                print(
                    'drop '+str(len(np.where(fl0 == 0)[0]))+" due to calibration")
            wl0 = wl0[~kick]
            fl0 = fl0[~kick]
            ivar0 = ivar0[~kick]
        else:
            wl0 = wl
            fl0 = fl
            ivar0 = ivar
            kick = np.logical_or(fl0 == 0,ivar0==0)
            if verbose:
                print(
                    'drop '+str(len(np.where(fl0 == 0)[0])))
            wl0 = wl0[~kick]
            fl0 = fl0[~kick]
            ivar0 = ivar0[~kick]
        tmp = []
        for i in list(self.lamgrid.values()):
            tmp.append([np.min(list(i)), np.max(list(i))])
        tmp = np.array(tmp).T
        self.whole_mask = np.logical_and(
            wl0 > tmp[0].max(), wl0 < tmp[1].min())

        wl0 = wl0[self.whole_mask]
        fl0 = fl0[self.whole_mask]
        ivar0 = ivar0[self.whole_mask]
        centroid = []
        for i in addition_elements:
            lst = atom_line(wl0.min(), wl0.max()).special_lines[i]
            lst = lst[np.logical_and(lst > wl0.min(), lst < wl0.max())]
            centroid.extend(lst)
        addition_lines = np.vstack((centroid, np.ones(len(centroid))*50)).T
        if make_plot:
            if given_plot is None:
                if onlyprofile:
                    self.lineprofile_plot = plt.subplots(1, len(addition_elements), figsize=(4*len(addition_elements), 10))
                else:
                    self.norm_plot = plt.subplots(
                        3, 1, figsize=(20, 10), sharex=True)
                    self.norm_plot[0].subplots_adjust(hspace=0)
                    if len(addition_elements)>0:
                        self.lineprofile_plot = plt.subplots(1, len(addition_elements), figsize=(4*len(addition_elements), 10))
                    else:
                        self.lineprofile_plot=None
            else:
                self.norm_plot = given_plot[0]
                self.lineprofile_plot = given_plot[1]


        def lnlike(prms):
            if self.isbinary == True:
                para1 = np.array(prms)[self.para1mask]
                para2 = np.array(prms)[self.para2mask]
                paras = [para1,para2]
            else:
                paras = [np.array(prms)]
            self.cont_fixed = True
            model = self.spectrum_sampler(wl, paras)
            self.cont_fixed = False
            #resid = np.abs(fl[self.mask] - model[self.mask]) * (len(self.cont_mask) - len(np.where(self.cont_mask == False)[0]))/len(self.mask) * np.sqrt(ivar[self.mask])
            #resid = np.abs(np.random.normal(loc=fl[self.mask],scale=np.sqrt(1/ivar[self.mask])) - model[self.mask]) * np.sqrt(ivar[self.mask])
            resid = np.abs(fl[self.mask]- model[self.mask]) * np.sqrt(ivar[self.mask])
            if not onlyprofile:
                c_diff = cont_diff(paras)
                chisq = np.sum(resid)+ np.sum(c_diff)
            else:
                chisq = np.sum(resid)
            if np.isnan(chisq):
                return -np.Inf
            lnlike = -0.5 * chisq
            # print(chisq / (np.sum(self.mask) - len(prms)))
            # check planck
            return lnlike

        def cont_diff(prms):
            # 3647 for maxima wavelength of balmer Discontinuities
            wl_cont = wl0[self.cont_mask]
            fl_cont = fl0[self.cont_mask]
            mass = self.decide_mass(TG2M,prms,self.specclass,err=False)
            radius= []
            for i in range(len(prms)):
                mass_tmp = mass[i]
                if np.isnan(mass_tmp):
                    return fl_cont
                radius.append(np.sqrt(mass_tmp*MSUN*G / 10**(prms[i][1]))/RSUN)
            #radius = [np.sqrt(mass1*MSUN*G / 10**(g1))/RSUN,np.sqrt(mass2*MSUN*G / 10**(g2))/RSUN] 
            # RD2 = radius2/parallax
            predict = self.spectrum_sampler(wl_cont, prms, specclass=self.specclass, radius=radius)
            #diff = np.abs(fl_cont-predict)
            #diff = np.abs(np.random.normal(loc =fl_cont,scale= np.sqrt(1/ivar0[self.cont_mask])) -predict)/2*np.sqrt(ivar0[self.cont_mask])
            diff = np.abs(fl_cont-predict)*np.sqrt(ivar0[self.cont_mask])
            #diff = np.abs(fl_cont -predict)/fl_cont
            return diff

        def lnprior(prms):
            if self.isbinary == True:
                para1 = np.array(prms)[self.para1mask]
                para2 = np.array(prms)[self.para2mask]
                paras = [para1,para2]
            else:
                paras = [np.array(prms)]
            for jj in range(nstarparams):
                if prms[jj] < prior_lows[jj] or prms[jj] > prior_highs[jj]:
                    return -np.Inf
            mass = self.decide_mass(TG2M,paras,self.specclass,err=False)
            for i in range(len(mass)):
                if np.isnan(mass[i]):
                    #print('nan reported', mass1, mass2)
                    return -np.Inf
            return 0

        def lnprob(prms):
            lp = lnprior(prms)
            if not np.isfinite(lp):
                return -np.Inf
            return lp + lnlike(prms)


        if init_rv is None:
            if verbose:
                print('fitting radial velocity...')
            self.rv, e_rv = self.sp.get_line_rv(
                wl0, fl0, ivar0, self.centroid_dict[rv_line], **rv_kw)
        else:
            self.rv,e_rv  = init_rv



        norm_kw['plot'] = plot_init

        outwl = (self.exclude_wl_default < np.min(wl)) & (
            self.exclude_wl_default > np.max(wl))
        self.exclude_wl = self.exclude_wl_default[~outwl]
        
        if len(self.exclude_wl) % 2 != 0:
            print('self.exclude_wl should have an even number of elements!')

        self.norm_kw = norm_kw

        if verbose:
            print('fitting continuum...')
        wl, fl, ivar = self.spline_norm_DA(wl0, fl0, ivar0, kwargs=norm_kw, crop=crop)
            # self.cont_fixed = True
        # Set to True to see how the models are normalized
        self.norm_kw['plot'] = False

        edges = []
        mask = np.zeros(len(wl0))
        if addition_lines is not None:
            for line in addition_lines:
                wl1 = line[0] - line[1]
                wl2 = line[0] + line[1]
                c1 = bisect_left(wl, wl1)
                c2 = bisect_left(wl, wl2)

                edges.extend([wl2, wl1])

                mask[c1:c2] = 1
        edges = np.flip(edges)
        self.mask = mask.astype(bool)
        self.edges = edges
        self.cont_mask = ~self.mask
        # if onlyprofile:
        #    self.mask = np.ones(len(fl0)).astype(bool)

        self.tscale = 10000
        self.lscale = 6
        def make_params(specclass, order, params, param_names,init_para):
            param_names.extend(
                [r'$T_{eff}$'+str(order+1), r'$\log{g}$'+str(order+1)])
            params.add('teff'+str(order+1), value=init_para[0][0]/ self.tscale,
                        min=self.Tgrid[specclass][0] / self.tscale, max=self.Tgrid[specclass][1] / self.tscale)
            params['teff'+str(order+1)].stderr = init_para[1][0]/self.tscale
            params.add('logg'+str(order+1), value=init_para[0][1]/
                        self.lscale, min=self.ggrid[specclass][0]/self.lscale, max=self.ggrid[specclass][1]/self.lscale)
            params['logg'+str(order+1)].stderr = init_para[1][1]/self.lscale

            prior_highs.extend(
                [params['teff'+str(order+1)].max*self.tscale, params['logg'+str(order+1)].max*self.lscale])
            prior_lows.extend(
                [params['teff'+str(order+1)].min*self.tscale, params['logg'+str(order+1)].min*self.lscale])
            if specclass == 'MS':
                #param_names.extend([r'[Fe/H]'+str(order+1), r'$[\alpha /Fe]$'+str(order+1)])
                param_names.extend([r'[Fe/H]'+str(order+1)])
                params.add('meta'+str(order+1), value=init_para[0][2], min=-1.3, max=0.2)
                #params.add('afe'+str(order+1), value=init_para[0][3], min=0, max=0.4, vary=False)
                params['meta'+str(order+1)].stderr = init_para[1][2]
                #params['afe'+str(order+1)].stderr = init_para[1][3]

                #prior_highs.extend([params['meta'+str(order+1)].max, params['afe'+str(order+1)].max])
                prior_highs.extend([params['meta'+str(order+1)].max])
                #prior_lows.extend([params['meta'+str(order+1)].min, params['afe'+str(order+1)].min])
                prior_lows.extend([params['meta'+str(order+1)].min])
            elif specclass == 'sd':
                params.add('R'+str(order+1), value=init_para[0][2],
                            min=0, max=2, vary=True)
                params['R'+str(order+1)].stderr = init_para[1][2]
                prior_highs.extend([10])
                prior_lows.extend([0])
                param_names.extend([r'$R_{%d}$' % order])
            # params.add('R'+str(order+1), value=1, min=0, max=5, vary=True)
            # prior_highs.extend([5])
            # prior_lows.extend([0])
            # param_names.extend([r'$R_{%d}$' % order])
            params.add('rv'+str(order+1),
                        value=self.rv[order], min=-1100, max=1100, vary=True)
            params['rv'+str(order+1)].stderr = e_rv[order]
            prior_highs.extend([1100])
            prior_lows.extend([-1100])
            param_names.extend([r'$rv_{%d}$' % (order+1)])
            return param_names, params
        
        prior_lows = []
        prior_highs = []
        param_names = []
        params = lmfit.Parameters()
        if self.isbinary == False:
            if init_paras is None:
                init_paras=[[np.mean(self.Tgrid[self.specclass[0]]),np.mean(self.ggrid[self.specclass[0]]),0,0],[0,0,0,0]]
            param_names, params = make_params(self.specclass[0], 0, params, param_names,init_paras)
            self.para1mask = np.where(['1' in i for i in list(params.keys())])[0]
            nstarparams = len(param_names)
            param_names.extend(['$c_%i$' % ii for ii in range(polyorder + 1)])
            for ii in range(polyorder):
                params.add('c_' + str(ii), value=0, min=-1, max=1)
                if ii == 0:
                    params['c_0'].set(value=1)

            def residual(params,wl,fl,fl0,ivar):
                para1 = np.array(params)
                para1[0] = para1[0] * self.tscale
                para1[1] = para1[1] * self.lscale
                mass=self.decide_mass(TG2M,[para1],self.specclass,err=False)
                radius1 = np.sqrt(mass[0]*MSUN*G/10**(para1[1]))/RSUN
                self.cont_fixed = True
                model = self.spectrum_sampler(wl, [para1],radius=[radius1])
                self.cont_fixed = False
                resid = np.abs(fl[self.mask] - model[self.mask])* np.sqrt(ivar[self.mask])
                
                res = np.zeros_like(wl)
                if np.isnan(mass[0]):
                    return res+100000
                if onlyprofile == False:
                    c_diff = cont_diff([para1])
                    res[self.mask] += resid
                    res[self.cont_mask] += c_diff
                else:
                    res[self.mask] += resid
                # print(np.sum(chi**2) / (np.sum(self.mask) - len(params)))
                return res
                
            star_rv = self.rv

            res = lmfit.minimizer.MinimizerResult()
            res.params = params
            res.residual = residual(params,wl,fl,fl0,ivar)
            chimin  = np.sum(res.residual**2)
            cov_matrix = np.zeros((len(params),len(params)))
            if verbose:
                print('test bin...')
                print('init_para:',init_paras)
                print('init_chi:',chimin)
            if verbose:
                print('final optimization...')

            teff1,logg1 = np.meshgrid(np.linspace(*self.Tgrid[self.specclass[0]], nteff),np.linspace(*self.ggrid[self.specclass[0]], nteff))

            combinations = np.column_stack((teff1.ravel(), logg1.ravel()))
            for n in range(len(combinations)):
                combination = combinations[n]
                teff1,logg1 = combination
                params['teff1'].set(value=teff1 / self.tscale)
                params['logg1'].set(value=logg1 / self.lscale)
                if progress:
                    alg.visualize.progress_bar(n,len(combinations),notes = str(datetime.datetime.now())+'-'+str(os.getpid()),frequency=10)
                    sys.stdout.flush()
                res_i = lmfit.minimize(residual, params,args=(wl,fl,fl0,ivar), **lmfit_kw)
                chi = np.sum(res_i.residual**2)
                if chi < chimin:
                    res = res_i
                    chimin = chi
                    cov_matrix = res.covar
        elif self.isbinary == True:
            if init_paras is None:
                init_paras=[[[np.mean(self.Tgrid[self.specclass[0]]),np.mean(self.ggrid[self.specclass[0]]),0,0],[0,0,0,0]],[[np.mean(self.Tgrid[self.specclass[1]]),np.mean(self.ggrid[self.specclass[1]]),0,0],[0,0,0,0]]]
            for i in range(len(self.specclass)):
                param_names, params = make_params(
                    self.specclass[i], i, params, param_names,init_paras[i])
            self.para1mask = np.where(
                ['1' in i for i in list(params.keys())])[0]
            self.para2mask = np.where(
                ['2' in i for i in list(params.keys())])[0]
            nstarparams = len(param_names)
            param_names.extend(['$c_%i$' % ii for ii in range(polyorder + 1)])
            for ii in range(polyorder):
                params.add('c_' + str(ii), value=0, min=-1, max=1)
                if ii == 0:
                    params['c_0'].set(value=1)

            def residual_bin(params,wl,fl,fl0,ivar):
                para1 = np.array(params)[self.para1mask]
                para2 = np.array(params)[self.para2mask]
                # params = np.array(params).reshape(-1, 2)
                res = np.zeros_like(wl)
                # params[:, 0] = params[:, 0] * self.tscale
                # params[:, 1] = params[:, 1] * self.lscale
                para1[0] = para1[0] * self.tscale
                para1[1] = para1[1] * self.lscale
                para2[0] = para2[0] * self.tscale
                para2[1] = para2[1] * self.lscale
                # '''
                mass=self.decide_mass(TG2M,[para1,para2],self.specclass,err=False)
                mass1,mass2=mass
                if np.isnan(mass1) or np.isnan(mass2):
                    return res+100000
                # '''
                radius1 = np.sqrt(mass1*MSUN*G/10**(para1[1]))/RSUN
                radius2 = np.sqrt(mass2*MSUN*G/10**(para2[1]))/RSUN
                self.cont_fixed = True
                model = self.spectrum_sampler(wl, [para1, para2],radius=[radius1,radius2])
                self.cont_fixed = False
                resid = np.abs(fl[self.mask] - model[self.mask])* np.sqrt(ivar[self.mask])
                if onlyprofile == False:
                    c_diff = cont_diff([para1, para2])
                    res[self.mask] += resid
                    res[self.cont_mask] += c_diff
                else:
                    res += resid*np.sqrt(ivar)
                return res
                
            star_rv = self.rv
            #if verbose:
            #    for i in range(len(self.rv)):
            #        print('Radial Velocity #%i = %i ± %i km/s' %
            #              (i, self.rv[i], e_rv[i]))
            # self.rv_fixed = True
            #res = lmfit.minimize(residual_bin, params,args=(wl,fl,fl0,ivar), **lmfit_kw)
            res = lmfit.minimizer.MinimizerResult()
            res.params = params
            res.residual = residual_bin(params,wl,fl,fl0,ivar)
            res.redchi = np.sum(res.residual**2)
            chimin = res.redchi
            cov_matrix = np.zeros((len(params),len(params)))

            #init_chi = np.sum(residual_bin(params,wl,fl,fl0,ivar)**2)/(np.sum(self.mask)+np.sum(self.cont_mask) -(len(param_names)+ polyorder))
            #res.redchi = np.sum(res.residual**2)/(len(self.whole_mask) -(len(param_names)+ polyorder))
            
            if verbose:
                print('test bin...')
                print('init_para:',init_paras)
                print('init_chi:',chimin)
            if verbose:
                print('final optimization...')
            teff1, teff2, logg1, logg2 = np.meshgrid(np.linspace(*self.Tgrid[self.specclass[0]], nteff), np.linspace(
                *self.Tgrid[self.specclass[1]], nteff), np.linspace(*self.ggrid[self.specclass[0]], nteff), np.linspace(*self.ggrid[self.specclass[1]], nteff))

            combinations = np.column_stack(
                (teff1.ravel(), teff2.ravel(), logg1.ravel(), logg2.ravel()))
            for n in range(len(combinations)):
                combination = combinations[n]
                teff1, teff2, logg1, logg2 = combination
                params['teff1'].set(value=teff1 / self.tscale)
                params['teff2'].set(value=teff2 / self.tscale)
                params['logg1'].set(value=logg1 / self.lscale)
                params['logg2'].set(value=logg2 / self.lscale)
                if progress:
                    #print('initializing at teff1 = %i K,teff2 = %i K,logg1 = %i ,logg2 = %i \r' %(teff1, teff2, logg1, logg2), end='')
                    alg.visualize.progress_bar(n,len(combinations),notes = str(datetime.datetime.now())+'-'+str(os.getpid()),frequency=10)
                    sys.stdout.flush()
                res_i = lmfit.minimize(
                    residual_bin, params,args=(wl,fl,fl0,ivar), **lmfit_kw)
                chi = np.sum(res_i.residual**2)
                if chi < chimin:
                    res = res_i
                    chimin = chi
                    cov_matrix = res.covar
        if verbose:
            print('done fitting')
            sys.stdout.flush()

        have_stderr = False

        stds = []
        mle = []

        for item in list(res.params.keys()):
            if res.params[item].stderr != None:
                if 'teff' in item:
                    stds.append(res.params[item].stderr*self.tscale)
                    mle.append(res.params[item].value*self.tscale)
                elif 'logg' in item:
                    stds.append(res.params[item].stderr*self.lscale)
                    mle.append(res.params[item].value*self.lscale)
                else:
                    stds.append(res.params[item].stderr)
                    mle.append(res.params[item].value)
                have_stderr = True
            else:
                stds.append(np.nan)
                if 'teff' in item:
                    mle.append(res.params[item].value*self.tscale)
                elif 'logg' in item:
                    mle.append(res.params[item].value*self.lscale)
                else:
                    mle.append(res.params[item].value)
                if verbose:
                    print('no errors from lmfit...')
        if polyorder > 0:
            cheb_coef = np.array(res.params)[4:]
        redchi = chimin / (len(wl) -(len(mle) + polyorder))
        try:
            e_coefs = [
                res.params['c_' + str(ii)].stderr for ii in range(polyorder)]
        except:
            e_coefs = 1e-2 * np.array(cheb_coef)
        if verbose:
            print('res of lmfit:', *list(res.params.keys()))
            print('mle:',mle)
            print('stds:',stds)
            print('redchi:',redchi,'chimin:',chimin)
        sys.stdout.flush()
        if polyorder > 0:
            mle.extend(cheb_coef)
            stds.extend(e_coefs)
        mle = np.array(mle)
        stds = np.array(stds)

        if mcmc:
            ndim = len(mle)

            sampler = emcee.EnsembleSampler(
                nwalkers, ndim, lnprob, **sampler_kw)

            pos0 = np.zeros((nwalkers, ndim))

            if have_stderr and polyorder == 0:  # do not trust covariances when polyorder > 0
                sigmas = stds  # USE ERR FROM LMFIT
            else:
                if verbose:
                    print('using 0.01 err')
                sigmas = np.abs(1e-2*np.array(mle))  # USE 1% ERROR

            init = mle
            for jj in range(ndim):
                pos0[:, jj] = np.random.normal(loc=init[jj],scale=3*np.abs(sigmas[jj]),size=nwalkers)
            '''
            for jj in range(nwalkers):
                pos0[jj] = (np.random.uniform(prior_lows, prior_highs))
            # print(pos0)
            '''
            if verbose:
                print('burning in chains...')
            b = sampler.run_mcmc(pos0, burn, progress=progress)

            sampler.reset()

            if verbose:
                print('sampling posterior...')
            b = sampler.run_mcmc(b.coords, ndraws, progress=progress)
            print('acceptance fraction: {:8.3f}'.format(
                sampler.acceptance_fraction[0]))
            #print('mean autocorrelation time: {:8.3f}'.format(np.mean(sampler.get_autocorr_time())))
            if plot_trace:
                f, axs = plt.subplots(ndim, 1, figsize=(10, 6))
                for jj in range(ndim):
                    axs[jj].plot(sampler.chain[:, :, jj].T,
                                 alpha=0.3, color='k')
                    plt.ylabel(param_names[jj])
                plt.xlabel('steps')
                if savename is not None:
                    plt.savefig(fig_path+savename + '+'.join(self.specclass)+'_trace.pdf',bbox_inches='tight',pad_inches=0.01, dpi=100)
                # plt.show()

            lnprobs = sampler.get_log_prob(flat=True)
            cov_matrix = np.cov(sampler.flatchain, rowvar=False)
            mle = sampler.flatchain[np.argmax(lnprobs)]
            redchi = -2 * np.max(lnprobs) / (len(wl) - ndim)
            # redchi = -2 * np.max(lnprobs)
            stds = np.std(sampler.flatchain, 0)
            m_tmp = []
            r_tmp = []
            if self.isbinary:
                massseries = self.decide_mass(TG2M,np.array([sampler.flatchain.T[self.para1mask],sampler.flatchain.T[self.para2mask]]),specclass=self.specclass,err=False)
                radiusseries = np.sqrt(massseries[0]*MSUN*G/10**(np.array(sampler.flatchain.T[self.para1mask])[1]))/RSUN
                radiusseries = np.vstack((radiusseries,np.sqrt(massseries[1]*MSUN*G/10**(np.array(sampler.flatchain.T[self.para2mask][1])))/RSUN))
                for k in range(len(massseries)):
                    tmp_mask = np.logical_and(np.isnan(massseries[k]),np.isnan(radiusseries[k]))
                    m_tmp.append(np.array(massseries[k])[~tmp_mask])
                    r_tmp.append(np.array(radiusseries[k])[~tmp_mask])
                massseries = m_tmp
                radiusseries = r_tmp
            else:
                massseries = self.decide_mass(TG2M,np.array([sampler.flatchain.T[self.para1mask]]),specclass=self.specclass,err=False)
                radiusseries = np.sqrt(massseries[0]*MSUN*G/10**(np.array(sampler.flatchain.T[self.para1mask])[1]))/RSUN
                tmp_mask = np.logical_and(np.isnan(massseries[0]),np.isnan(radiusseries))
                m_tmp.append(np.array(massseries[0])[~tmp_mask])
                r_tmp.append(np.array(radiusseries)[~tmp_mask])
                massseries = m_tmp
                radiusseries = r_tmp
            if len(massseries) == 0:
                print('empty mass!')
            else:
                print('mass:',len(massseries))
            self.flatchain = sampler.flatchain
            upstd = []
            downstd = []
            for i in range(sampler.flatchain.shape[1]):
                upstd.append(np.percentile(sampler.flatchain[:,i],q=84)-np.percentile(sampler.flatchain[:,i],q=50))
                downstd.append(np.percentile(sampler.flatchain[:,i],q=50)-np.percentile(sampler.flatchain[:,i],q=16))
            upstd = np.array(upstd)
            downstd=np.array(downstd)
            '''if self.isbinary == True and np.shape(np.shape(mle))[0] == 1:
                teff = np.where(['teff' in i for i in list(res.params.keys())])
                logg = np.where(['logg' in i for i in list(res.params.keys())])

            else:
                if (np.min(mle[0]) < 7000 or np.max(mle[0]) > 38000) and progress:
                    print(
                        'temperature is near bound of the model grid! exercise caution with this result')
                if (np.min(mle[1]) < 6.7 or np.max(mle[1]) > 9.3) and progress:
                    print(
                        'logg is near bound of the model grid! exercise caution with this result')'''

            if plot_corner:
                f = corner.corner(sampler.flatchain, labels=param_names[:ndim],
                                  label_kwargs=dict(fontsize=12), quantiles=(0.16, 0.5, 0.84),
                                  show_titles=True, title_kwargs=dict(fontsize=12))
                for ax in f.get_axes():
                    ax.tick_params(axis='both', labelsize=12)
                if savename is not None:
                    plt.savefig(fig_path+savename + '+'.join(self.specclass)+'_corner.pdf',bbox_inches='tight',pad_inches=0.01, dpi=100)
                # plt.show()

            if plot_corner_full:

                f = corner.corner(sampler.flatchain, labels=param_names,
                                  label_kwargs=dict(fontsize=12), quantiles=(0.16, 0.5, 0.84),
                                  show_titles=False)

                for ax in f.get_axes():
                    ax.tick_params(axis='both', labelsize=12)

        if self.isbinary == True:
            if np.shape(np.shape(mle))[0] == 1:
                para1name = np.array(list(res.params.keys()))[self.para1mask]
                para2name = np.array(list(res.params.keys()))[self.para2mask]

                para1 = mle[self.para1mask]
                para2 = mle[self.para2mask]
                std1 = stds[self.para1mask]
                std2 = stds[self.para2mask]
                if mcmc:
                    upstd1   = upstd[self.para1mask]
                    downstd1 = downstd[self.para1mask]
                    upstd2   = upstd[self.para2mask]
                    downstd2 = downstd[self.para2mask]
                    upstd = [upstd1,upstd2]
                    downstd = [downstd1,downstd2]
                print(self.para1mask,cov_matrix)
                para_cov1 = cov_matrix[self.para1mask[:-1]][:,self.para1mask[:-1]]
                para_cov2 = cov_matrix[self.para2mask[:-1]][:,self.para2mask[:-1]]
                mle = [para1, para2]
                stds = [std1, std2]
                para_cov = [para_cov1,para_cov2]
        else:
            para1name = np.array(list(res.params.keys()))[self.para1mask]
            para_cov1 = cov_matrix[self.para1mask[:-1]][:,self.para1mask[:-1]]
            mle = [mle]
            stds = [stds]
            if mcmc:
                upstd1   = upstd[self.para1mask]
                downstd1 = downstd[self.para1mask]
                upstd = [upstd1]
                downstd = [downstd1]
            para_cov = [para_cov1]
        wl_cont = wl0[self.cont_mask]
        fl_cont = fl0[self.cont_mask]

        mass = self.decide_mass(TG2M,mle,self.specclass,para_cov,err=True)
        radius = []
        e_radius=[]
        for i in range(len(mass)):
            r_tmp = np.sqrt(mass[i][0]*MSUN*G/10**(mle[i][1]))/RSUN
            radius.append(r_tmp)
            e_radius.append(np.sqrt((r_tmp/2/mass[i][0])**2*mass[i][1]**2+(r_tmp*np.log(10)/2)**2*stds[i][1]**2))
        if mcmc:
            massup = []
            massdown=[]
            radiusup=[]
            radiusdown=[]
            for i in range(len(self.specclass)):
                massup.append(np.percentile(massseries[i],q=84)-np.percentile(massseries[i],q=50))
                massdown.append(np.percentile(massseries[i],q=50)-np.percentile(massseries[i],q=16))
                radiusup.append(np.percentile(radiusseries[i],q=84)-np.percentile(radiusseries[i],q=50))
                radiusdown.append(np.percentile(radiusseries[i],q=50)-np.percentile(radiusseries[i],q=16))
            
        self.cont_fixed = True
        fit_fl = self.spectrum_sampler(wl, mle,specclass=self.specclass, radius=radius)
        self.cont_fixed = False
        if corr_3d and mle[0] < 15000:
            if verbose:
                print('applying 3D corrections...')
            corr = corr3d(mle[0], mle[1])
            mle[0] = corr[0]
            mle[1] = corr[1]


        if make_plot:
            print('make plot')
            if onlyprofile:
                self.makeplot(wl,fl,fit_fl,ivar,'profile',savename,mle,[*addition_elements],parallax,given_plot=self.lineprofile_plot)
            else:
                norm_ax = self.norm_plot[1]
                norm_ax[1].plot(wl, fit_fl, c='r',
                                label='+'.join(self.specclass))
                norm_ax[2].plot(wl, fl-fit_fl)
                norm_ax[2].hlines(3*np.mean(np.sqrt(1/ivar[np.where(ivar != 0)[0]])),
                                  wl.min(), wl.max(), color='k', lw=1, ls='--')
                norm_ax[2].hlines(-3*np.mean(np.sqrt(1/ivar[np.where(ivar != 0)[0]])), wl.min(),wl.max(), color='k', lw=1, ls='--', label=r'$\pm 3\sigma$')
                norm_ax[2].set_title('residual', y=0.9)
                ax = self.lineprofile_plot[1]
                if len(addition_elements)==1:
                    ax = np.array([ax])
                # fig, ax = plt.subplots(1, 3, figsize=(25, 9), gridspec_kw={'width_ratios': [1, 3, 1]})
                breakpoints = []
                # ax[1].plot(wl0, fl0, c='k')
                # norm_ax[0].plot(wl_cont, fl_cont, c='r')
                ax[0].set_xlabel(r'$\mathrm{\Delta \lambda}\ (\mathrm{\AA})$')
                ax[0].set_ylabel('Normalized Flux')
                for i in range(len(addition_elements)):
                    element = np.flip(addition_elements)[i]

                    lst = atom_line(wl0.min(), wl0.max()).special_lines[element]
                    lst = lst[np.logical_and(lst > wl0.min(), lst < wl0.max())]
                    addition_lines = np.vstack(
                        (lst, np.ones(len(lst))*30)).T
                    breakpoints = []
                    ax[i].set_title(element)
                    ax[i].set_xlim(-50, 50)
                    for kk in range(len(self.edges)):

                        if (kk + 1) % 2 == 0:
                            continue
                        breakpoints.append(bisect_left(wl, self.edges[kk]))
                        breakpoints.append(bisect_left(wl, self.edges[kk+1]))
                    for jj in range(len(breakpoints)):
                        if (jj + 1) % 2 == 0:
                            continue
                        wl_seg = wl[breakpoints[jj]:breakpoints[jj+1]]
                        fl_seg = fl[breakpoints[jj]:breakpoints[jj+1]]
                        fit_fl_seg = fit_fl[breakpoints[jj]:breakpoints[jj+1]]
                        peak = int(len(wl_seg)/2)
                        delta_wl = wl_seg - wl_seg[peak]
                        interval = fl_seg.mean()/2
                        ax[i].text(delta_wl[0], 1 + fl_seg[0] -
                                     interval * jj, '%.0f' % wl_seg[peak], horizontalalignment='right')
                        ax[i].plot(delta_wl, 1 + fl_seg - interval * jj, 'k')
                        ax[i].plot(delta_wl, 1 + fit_fl_seg -
                                     interval * jj, 'r')
                    ax[i].set_xlabel(r'$\mathrm{\Delta \lambda}\ (\mathrm{\AA})$')
                    # ax[i+1].set_ylabel('Normalized Flux')
                '''
                norm_ax[0].plot(wl0, planck(
                    wl0, mle[0][0], RD1, self.rv[0]), label='comp1')
                norm_ax[0].plot(wl0, planck(
                    wl0, mle[1][0], RD2, self.rv[1]), label='comp2')
                norm_ax[0].plot(wl0, planck(wl0, mle[0][0], RD1, self.rv[0]
                                            )+planck(wl0, mle[1][0], RD2, self.rv[1]))
                '''
                # norm_ax[0].plot(wl0, fl0, label='calibrated data', c='k')
                norm_ax[0].set_title(savename)
                if self.isbinary == True:
                    for i in range(len(mle)):
                        print(mle[i])
                        norm_ax[0].plot(wl0, self.spectrum_sampler(
                            wl0, [mle[i]], specclass=self.specclass[i], radius=[radius[i]]), label=self.specclass[i])
                norm_ax[0].plot(wl0, self.spectrum_sampler(
                    wl0, mle, specclass=self.specclass, radius=radius), label='+'.join(self.specclass))
                left = 3600
                bottom = norm_ax[2].get_yticks()[1]
                step = (norm_ax[2].get_xticks()[1] -
                        norm_ax[2].get_xticks()[0])/1.6
                result_str = ''
                for i in range(len(para1name)):
                    result_str += para1name[i] + \
                        r'$= %.2f \pm %.2f$  ' % (mle[0][i], stds[0][i])
                result_str +='\n'
                if self.isbinary == True:
                    for i in range(len(para2name)):
                        result_str += para2name[i] + \
                            r'$= %.2f \pm %.2f$  ' % (mle[1][i], stds[1][i])
                result_str += r'$\chi_r^2$ = %.2f  ' % (redchi)
                for i in range(len(mass)):
                    result_str += r'$R%d /R_{\odot}$: %.2f  ' % (i,radius[i])
                norm_ax[2].text(0.1,0.1, result_str,transform = norm_ax[2].transAxes)
                '''for i in range(len(para1name)):
                    norm_ax[0].text(left, bottom, para1name[i]+r'$= %.2f \pm %.2f$' %
                                    (mle[0][i], stds[0][i]), fontsize=10, ha='right', color='k')
                    left += step
                for i in range(len(para2name)):
                    norm_ax[0].text(left, bottom, para2name[i]+r'$= %.2f \pm %.2f$' %
                                    (mle[1][i], stds[1][i]), fontsize=10, ha='right', color='k')
                    left += step
                norm_ax[0].text(left, bottom, r'$\chi_r^2$ = %.2f' %
                                (redchi), fontsize=10, ha='right')
                norm_ax[0].text(left+step, bottom, s=r'$R1/R_{\odot}$:'+'%.2f' %
                                (radius1/(6.955*1e10)), ha='right')
                norm_ax[0].text(left+2*step, bottom, s=r'$R2/R_{\odot}$:'+'%.2f' %
                                (radius2/(6.955*1e10)), fontsize=10, ha='right')'''
                norm_ax[2].set_xlabel(
                    r'$\mathrm{\lambda}\ (\mathrm{\AA})$')
                norm_ax[0].set_ylabel(r'Flux $(erg/cm^{2}/s/\AA)$')
                for i in range(len(norm_ax)):
                    norm_ax[i].legend()
            
        if savename is not None:
            if onlyprofile == False:
                self.norm_plot[0].savefig(fig_path+savename + '+'.join(self.specclass)+'_fit.pdf',bbox_inches='tight',pad_inches=0.01,dpi=100)
            self.lineprofile_plot[0].savefig(fig_path+savename + '+'.join(self.specclass)+'_profile.pdf', dpi=100,bbox_inches='tight',pad_inches=0.01)
            lock = multiprocessing.Lock()
            with lock:
                if self.isbinary==True:
                    with open(output_path+savename+'.txt','a') as f:
                        f.write('\n'+str(datetime.datetime.now())+'--'+str(os.getpid())+'--'+'+'.join(self.specclass)+'--'+str(redchi)+':\n')
                        f.write('\t'.join([*para1name, *para2name, 'mass1', 'mass2', 'radius1', 'radius2']))
                        f.write('\n')
                        f.write('\t'.join(list(map(str,[*mle[0], *mle[1], *np.array(mass).T[0], *radius]))))
                        f.write('\n')
                        f.write('\t'.join(list(map(str,[*stds[0], *stds[1], *np.array(mass).T[1],*e_radius]))))
                        f.write('\n')
                        
                        if mcmc:
                            f.write('\t'.join(list(map(str,[*upstd[0], *upstd[1],*massup,*radiusup]))))
                            f.write('\n')
                            f.write('\t'.join(list(map(str,[*downstd[0], *downstd[1],*massdown,*radiusdown]))))
                            f.write('\n')
                            f.write('='*100)
                            np.save(output_path+savename+'_chain_'+str(datetime.datetime.now()),sampler.chain)
                            np.save(output_path+savename+'_flatchain_'+str(datetime.datetime.now()),sampler.flatchain)
                            np.save(output_path+savename+'_lnprobs_'+str(datetime.datetime.now()),lnprobs)
                else:
                    #print(mass,radius,e_radius)
                    with open(output_path+savename+'.txt','a') as f:
                        f.write('\n'+str(datetime.datetime.now())+'--'+str(os.getpid())+'--'+'+'.join(self.specclass)+'--'+str(redchi)+':\n')
                        f.write('\t'.join([*para1name,'mass1','radius1']))
                        f.write('\n')
                        f.write('\t'.join(list(map(str,[*mle[0],mass[0][0],radius[0]]))))
                        f.write('\n')
                        f.write('\t'.join(list(map(str,[*stds[0],mass[0][1],e_radius[0]]))))
                        f.write('\n')
                        if mcmc:
                            f.write('\t'.join(list(map(str,[*upstd[0],*massup,*radiusup]))))
                            f.write('\n')
                            f.write('\t'.join(list(map(str,[*downstd[0],*massdown,*radiusdown]))))
                            f.write('\n')
                            f.write('='*100)
                            np.save(output_path+savename+'_chain_'+str(datetime.datetime.now()),sampler.chain)
                            np.save(output_path+savename+'_flatchain_'+str(datetime.datetime.now()),sampler.flatchain)
                            np.save(output_path+savename+'_lnprobs_'+str(datetime.datetime.now()),lnprobs)
        self.exclude_wl = self.exclude_wl_default
        self.cont_fixed = False
        self.rv = 0  # RESET THESE PARAMETERS
        if self.isbinary == True:
            mle = np.array([*mle[0], *mle[1],    *np.array(mass).T[0], *radius]).reshape(1, -1)
            stds = np.array([*stds[0], *stds[1], *np.array(mass).T[1], *e_radius]).reshape(1, -1)
            if mcmc:
                upstd = np.array([*upstd[0], *upstd[1],*massup,*radiusup]).reshape(1, -1)
                downstd = np.array([*downstd[0], *downstd[1],*massdown,*radiusdown]).reshape(1, -1)
                Final_res = pd.DataFrame(np.vstack((np.vstack((mle, stds)),np.vstack((upstd, downstd)))).reshape(
                4, -1), columns=[*para1name, *para2name, 'mass1', 'mass2', 'radius1', 'radius2'])
            else:
                Final_res = pd.DataFrame((np.vstack((mle, stds))).reshape(2, -1), columns=[*para1name, *para2name, 'mass1', 'mass2', 'radius1', 'radius2'])
            if verbose:
                print(mle,stds)
            
        else:
            if verbose:
                print(mle,stds)
            mle = np.array([*mle[0], mass[0][0],*radius]).reshape(1, -1)
            stds = np.array([*stds[0],mass[0][1],*e_radius]).reshape(1, -1)
            if mcmc:
                upstd = np.array([*upstd[0],*massup,*radiusup]).reshape(1, -1)
                downstd = np.array([*downstd[0],*massdown,*radiusdown]).reshape(1, -1)
            if mcmc:
                Final_res = pd.DataFrame(np.vstack((np.vstack((mle, stds)),np.vstack((upstd, downstd)))).reshape(4, -1), columns=[*para1name,'mass','radius'])
            else:
                Final_res = pd.DataFrame(np.vstack((mle, stds)).reshape(2, -1), columns=[*para1name,'mass','radius'])
        return Final_res, redchi, [self.norm_plot, self.lineprofile_plot]
    def makeplot(self,wl,fl,fit_fl,ivar,plottype,title,mle,addition_elements,parallax,width=30,stds=None,paraname=None,radius=None,masses=None,given_plot=None,text=False,showlines=False,kwargs=dict(fontsize=12)):
        '''
        kwargs is varing with different plottype
        profile:addition_element
        norm:title,radius1,radius2,para1name,para2name,mle
        '''
        if given_plot is None:
            if plottype == 'norm':
                given_plot = plt.subplots(
                    2, 1, figsize=(20, 8), gridspec_kw={'height_ratios': [3, 1]},sharex=True)
                given_plot[0].subplots_adjust(hspace=0)
            elif plottype == 'profile': 
                given_plot = plt.subplots(1, len(addition_elements), figsize=(4*len(addition_elements), 10))

        if plottype =='profile':
            ax = given_plot[1]
            if len(addition_elements) == 1:
                ax = np.array([ax])
            ax[0].set_ylabel('Normalized Flux')
            for i in range(len(addition_elements)):
                element = np.flip(addition_elements)[i]
                ax[i].set_title(element)
                lst = atom_line(wl.min(), wl.max()).special_lines[element]
                lst = lst[np.logical_and(lst > wl.min(), lst < wl.max())]
                addition_lines = np.vstack((lst, np.ones(len(lst))*width)).T
                edges = np.vstack((addition_lines.T[0]+addition_lines.T[1],addition_lines.T[0]-addition_lines.T[1])).T.reshape(-1)
                edges = np.flip(edges)
                breakpoints = []
                for kk in range(len(edges)):
                    if (kk + 1) % 2 == 0:
                        continue
                    breakpoints.append(bisect_left(wl, edges[kk]))
                    breakpoints.append(bisect_left(wl, edges[kk+1]))
                for jj in range(len(breakpoints)):
                    if (jj + 1) % 2 == 0:
                        continue
                    wl_seg = wl[breakpoints[jj]:breakpoints[jj+1]]
                    fl_seg = fl[breakpoints[jj]:breakpoints[jj+1]]
                    fit_fl_seg = fit_fl[breakpoints[jj]:breakpoints[jj+1]]
                    peak = int(len(wl_seg)/2)
                    delta_wl = wl_seg - wl_seg[peak]
                    interval = fl_seg.mean()/2
                    ax[i].text(delta_wl[0], 1 + fl_seg[0] -
                                    interval * jj, '%.0f' % wl_seg[peak], horizontalalignment='right')
                    ax[i].plot(delta_wl, 1 + fl_seg - interval * jj, 'k')
                    ax[i].plot(delta_wl, 1 + fit_fl_seg -interval * jj, 'r')
                ax[i].set_xlabel(r'$\mathrm{\Delta \lambda}\ (\mathrm{\AA})$')
        elif plottype == 'norm':
            norm_ax = given_plot[1]
            norm_ax[1].set_ylim(-0.4,0.5)
            norm_ax[0].set_title(title.split('/')[-1],y=0.85,**kwargs)
            if len(mle) ==1:
                y= self.spectrum_sampler(wl, mle, specclass=self.specclass, radius=radius,parallax=parallax,three=False)
            else:
                y1,y2 = self.spectrum_sampler(wl, mle, specclass=self.specclass, radius=radius,parallax=parallax,three=True)
                y=y1+y2
                norm_ax[0].plot(wl,y1, label=self.specclass[0],zorder=10)
                norm_ax[0].plot(wl,y2, label=self.specclass[1],zorder=10)
            norm_ax[0].plot(wl,y , label='+'.join(self.specclass),zorder=10)
            flat_fl,ivar_norm = self.sp.spline_norm(wl, fl, ivar, self.exclude_wl, norm_ax,showline = None,splr=False,plot=True)
            flat_y, _ =self.sp.spline_norm(wl, y, ivar, self.exclude_wl, norm_ax,showline = None,splr=False,plot=False)          
            norm_ax[1].plot(wl, flat_fl-flat_y,c='grey')
            err = 3*np.sqrt(1/ivar_norm[np.where(ivar_norm != 0)[0]])
            norm_ax[1].hlines(3*np.mean(np.sqrt(1/ivar_norm[np.where(ivar_norm != 0)[0]])), wl.min(), wl.max(), color='k', lw=1, ls='--')
            norm_ax[1].hlines(-3*np.mean(np.sqrt(1/ivar_norm[np.where(ivar_norm != 0)[0]])), wl.min(),wl.max(), color='k', lw=1, ls='--', label=r'$\pm 3\sigma$')
            if text:
                result_str = ''
                for j in range(len(mle)):
                    pp = paraname[j]
                    for i in range(len(pp)):
                        if 'afe' in pp[i]:
                            continue
                        result_str += pp[i] + r'$= %.2f \pm %.2f$  ' % (mle[j][i], stds[j][i])
                result_str += r'$R%d/R_{\odot}$: %.2f  ' % (j+1,radius[j]) +r'$M%d/M_{\odot}$: %.2f  ' % (j+1,masses[j])
                result_str += '\n'
                left = 3600
                bottom = norm_ax[1].get_yticks()[1]
                norm_ax[0].text(left, bottom, result_str,fontsize=8,zorder=20)
            norm_ax[1].set_xlabel(
                r'$\mathrm{\lambda}\ (\mathrm{\AA})$',**kwargs)
            norm_ax[0].set_ylabel(r'Flux $(erg/cm^{2}/s/\AA)$',**kwargs)
            if showlines:
                atom_line(wl.min(),wl.max()).showlines(wl,addition_elements,norm_ax[1])

        return given_plot
if __name__ == '__main__':
    print('yes')
    gfp = GFP(resolution=3, specclass='DA')
    wl = np.linspace(3000, 9000, 5000)
    fl = gfp.spectrum_sampler(wl, [6500, 28000], [7.5, 6.58])
    # fl = gfp.spectrum_sampler(wl, 28000, 7.5)
    ivar = np.repeat(0.001, len(wl))
    plt.plot(wl, fl)

    result = gfp.fit_spectrum(wl, fl, ivar, mcmc=True)
