import warnings
import numpy as np
from bisect import bisect_left
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import pickle
import os
import scipy
import lmfit
from lmfit.models import LinearModel, VoigtModel, GaussianModel, LorentzianModel
import numpy.polynomial.polynomial as poly
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
from scipy.signal import correlate
from scipy.interpolate import splev, splrep, LSQUnivariateSpline
import emcee
import corner
import astropy.units as u
import astropy.constants as c

path = os.path.abspath(__file__)
dir_path = os.path.dirname(path)


class SpecTools():

    '''

    Spectrum processing tools and functions. 

    '''

    def __init__(self, plot_continuum=False, smoothing=1e-15, filter_skylines=True, crop=True):
        self.plot_continuum = plot_continuum
        self.smoothing = smoothing
        self.filter_skylines = filter_skylines
        self.crop = crop
        self.halpha = np.float64(6564.61)
        self.hbeta = np.float64(4862.68)
        self.hgamma = np.float64(4341.68)
        self.hdelta = np.float64(4102.89)
        linear_model = LinearModel(prefix='l_')
        self.params = linear_model.make_params()
        voigt_model = VoigtModel(prefix='v_')
        self.params.update(voigt_model.make_params())
        self.cm = linear_model - voigt_model
        self.params['v_amplitude'].set(value=150)
        self.params['v_sigma'].set(value=5)
        self.params['l_intercept'].set(value=25)
        self.params['l_slope'].set(value=0)

    def continuum_normalize(self, wl, fl, ivar=None):
        '''
        Continuum-normalization with smoothing splines that avoid a pre-made list of absorption 
        lines for DA and DB spectra. To normalize spectra that only have Balmer lines (DA),
         we recommend using the `normalize_balmer` function instead. Also crops the spectrum 
         to the 3700 - 7000 Angstrom range. 

        Parameters
        ---------
        wl : array
            Wavelength array of spectrum
        fl : array
            Flux array of spectrum
        ivar : array, optional
            Inverse variance array. If `None`, will return only the normalized wavelength and flux. 

        Returns
        -------
            tuple
                Tuple of cropped wavelength, cropped and normalized flux, and (if ivar is not None)
                 cropped and normalized inverse variance array. 

        '''

        cmask = (
            ((wl > 6900) * (wl < 7500)) +
            ((wl > 6200) * (wl < 6450)) +
            ((wl > 5150) * (wl < 5650)) +
            ((wl > 4600) * (wl < 4670)) +
            ((wl > 4200) * (wl < 4300)) +
            ((wl > 3910) * (wl < 3950)) +
            ((wl > 3750) * (wl < 3775)) + ((wl > np.min(wl))
                                           * (wl < np.min(wl) + 50)) + (wl > 7500)
        )

        spl = scipy.interpolate.splrep(wl[cmask], fl[cmask], k=1, s=1000)
        # plt.figure()
        # plt.plot(wl, fl)
        # plt.plot(wl,scipy.interpolate.splev(wl, spl))
        # plt.show()
        norm = fl/scipy.interpolate.splev(wl, spl)

        if ivar is not None:
            ivar_norm = ivar * scipy.interpolate.splev(wl, spl)**2
            return wl, norm, ivar_norm
        elif ivar is None:
            return wl, norm

    def normalize_line(self, wl, fl, ivar, centroid, distance, make_plot=False, return_centre=False):
        '''
        Continuum-normalization of a single absorption line by fitting a linear model added 
        to a Voigt profile to the spectrum, and dividing out the linear model. 

        Parameters
        ---------
        wl : array
            Wavelength array of spectrum
        fl : array
            Flux array of spectrum
        ivar : array, optional
            Inverse variance array. If `None`, will return only the normalized wavelength and flux. 
        centroid : float
            The theoretical centroid of the absorption line that is being fitted, in wavelength units.
        distance : float
            Distance in Angstroms away from the line centroid to include in the fit. Should include 
            the entire absorption line wings with minimal continum. 
        make_plot : bool, optional
            Whether to plot the linear + Voigt fit. Use for debugging. 

        Returns
        -------
            tuple
                Tuple of cropped wavelength, cropped and normalized flux, and (if ivar is not None) 
                cropped and normalized inverse variance array. 

        '''

        self.params['v_center'].set(value=centroid)

        crop1 = bisect_left(wl, centroid - distance)
        crop2 = bisect_left(wl, centroid + distance)

        cropped_wl = wl[crop1:crop2]
        cropped_fl = fl[crop1:crop2]

        #cropped_fl = cropped_fl / np.nanmax(cropped_fl)

        try:
            res = self.cm.fit(cropped_fl, self.params,
                              x=cropped_wl, nan_policy='omit')
        except TypeError:
            print(
                'profile fit failed. ensure all lines passed to normalize_balmer are present on the spectrum!')
            raise Exception(
                'profile fit failed. ensure all lines passed to normalize_balmer are present on the spectrum!')

        if res.message != 'Fit succeeded.':
            print(
                'the line fit was ill-constrained. visually inspect the fit quality with make_plot = True')
        slope = res.params['l_slope']
        intercept = res.params['l_intercept']

        if make_plot:
            plt.plot(cropped_wl, cropped_fl)
            #plt.plot(cropped_wl, self.cm.eval(params, x=cropped_wl))
            plt.plot(cropped_wl, res.eval(res.params, x=cropped_wl))
            plt.plot(cropped_wl, cropped_wl*slope + intercept)
            plt.show()

        continuum = (slope * cropped_wl + intercept)

        fl_normalized = cropped_fl / continuum

        if ivar is not None:
            cropped_ivar = ivar[crop1:crop2]
            ivar_normalized = cropped_ivar * continuum**2
            return cropped_wl, fl_normalized, ivar_normalized
        elif return_centre:
            return cropped_wl, fl_normalized, res.params['v_center']
        else:
            return cropped_wl, fl_normalized

    def normalize_balmer(self, wl, fl, ivar=None, lines=['alpha', 'beta', 'gamma', 'delta'],
                         skylines=False, make_plot=False, make_subplot=False, make_stackedplot=False,
                         centroid_dict=dict(
                             alpha=6564.61, beta=4862.68, gamma=4341.68, delta=4102.89, eps=3971.20, h8=3890.12),
                         distance_dict=dict(alpha=300, beta=200, gamma=120, delta=75, eps=50, h8=25), sky_fill=np.nan):
        '''
        Continuum-normalization of any spectrum by fitting each line individually. 

        Fits every absorption line by fitting a linear model added to a Voigt profile to 
        the spectrum, and dividing out the linear model. 
        All normalized lines are concatenated and returned. For statistical and plotting 
        purposes, two adjacent lines should not have overlapping regions (governed by the `distance_dict`). 

        Parameters
        ---------
        wl : array
            Wavelength array of spectrum
        fl : array
            Flux array of spectrum
        ivar : array, optional
            Inverse variance array. If `None`, will return only the normalized wavelength and flux. 
        lines : array-like, optional
            Array of which Balmer lines to include in the fit. Can be any combination of 
            ['alpha', 'beta', 'gamma', 'delta', 'eps', 'h8']
        skylines : bool, optional
            If True, masks out pre-selected telluric features and replace them with `np.nan`. 
        make_plot : bool, optional
            Whether to plot the continuum-normalized spectrum.
        make_subplot : bool, optional
            Whether to plot each individual fit of the linear + Voigt profiles. Use for debugging. 
        make_stackedplot : bool, optional
            Plot continuum-normalized lines stacked with a common centroid, vertically displaced for clarity. 
        centroid_dict : dict, optional
            Dictionary of centroid names and theoretical wavelengths. Change this if your wavelength calibration is different from SDSS. 
        distance_dict : dict, optional
            Dictionary of centroid names and distances from the centroid to include in the normalization process. Should include the entire wings of each line and minimal continuum. No 
            two adjacent lines should have overlapping regions. 
        sky_fill : float
            What value to replace the telluric features with on the normalized spectrum. Defaults to np.nan. 

        Returns
        -------
            tuple
                Tuple of cropped wavelength, cropped and normalized flux, and (if ivar is not None) cropped and normalized inverse variance array. 

        '''

        fl_normalized = []
        wl_normalized = []
        ivar_normalized = []
        ct = 0

        for line in lines:
            if ivar is not None:
                wl_segment, fl_segment, ivar_segment = self.normalize_line(
                    wl, fl, ivar, centroid_dict[line], distance_dict[line], make_plot=make_subplot)
                fl_normalized = np.append(fl_segment, fl_normalized)
                wl_normalized = np.append(wl_segment, wl_normalized)
                ivar_normalized = np.append(ivar_segment, ivar_normalized)

            else:
                wl_segment, fl_segment = self.normalize_line(wl, fl, None, centroid_dict[line],
                                                             distance_dict[line], make_plot=make_subplot)
                if make_subplot:
                    plt.show()
                fl_normalized = np.append(fl_segment, fl_normalized)
                wl_normalized = np.append(wl_segment, wl_normalized)

        if skylines:
            skylinemask = (wl_normalized > 5578.5 - 10)*(wl_normalized < 5578.5 + 10) + (wl_normalized > 5894.6 - 10)\
                * (wl_normalized < 5894.6 + 10) + (wl_normalized > 6301.7 - 10)*(wl_normalized < 6301.7 + 10) + \
                (wl_normalized > 7246.0 - 10)*(wl_normalized < 7246.0 + 10)
            fl_normalized[skylinemask] = sky_fill

        if make_plot:
            plt.plot(wl_normalized, fl_normalized, 'k')

        if make_stackedplot:
            breakpoints = np.nonzero(np.diff(wl_normalized) > 5)[0]
            breakpoints = np.concatenate(([0], breakpoints, [None]))
            plt.figure(figsize=(5, 8))
            for kk in range(len(breakpoints) - 1):
                wl_seg = wl_normalized[breakpoints[kk] + 1:breakpoints[kk+1]]
                fl_seg = fl_normalized[breakpoints[kk] + 1:breakpoints[kk+1]]
                peak = int(len(wl_seg)/2)
                delta_wl = wl_seg - wl_seg[peak]
                plt.plot(delta_wl, 1 + fl_seg - 0.35 * kk, 'k')

            plt.xlabel(r'$\mathrm{\Delta \lambda}\ (\mathrm{\AA})$')
            plt.ylabel('Normalized Flux')
            plt.show()

        if ivar is not None:
            return wl_normalized, fl_normalized, ivar_normalized
        else:
            return wl_normalized, fl_normalized

    def find_nearest(self, array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return array[idx]

    def interpolate(self, wl, flux, target_wl=np.arange(4000, 8000)):
        func = interp1d(wl, flux, kind='linear',
                        assume_sorted=True, fill_value='extrapolate')
        interpflux = func(target_wl)[1]
        return target_wl, interpflux

    def linear(self, wl, p1, p2):
        ''' Linear polynomial of degree 1 '''

        return p1 + p2*wl

    def chisquare(self, residual):
        ''' Chi^2 statistics from residual

        Unscaled chi^2 statistic from an array of residuals (does not account for uncertainties).
        '''

        return np.sum(residual**2)

    def find_centroid(self, wl, flux, centroid, half_window=25, window_step=2, n_fit=12, make_plot=False,
                      pltname='', debug=False, normalize=True):
        '''
        Statistical inference of spectral redshift by iteratively fitting Voigt profiles to cropped windows around the line centroid. 

        Parameters
        ---------
        wl : array
            Wavelength array of spectrum
        flux : array
            Flux array of spectrum
        centroid : float
            Theoretical wavelength of line centroid
        half_window : float, optional
            Distance in Angstroms from the theoretical centroid to include in the fit
        window_step : float, optional
            Step size in Angstroms to reduce the half-window size after each fitting iteration
        n_fit : int, optional
            Number of iterated fits to perform
        make_plot : bool, optional
            Whether to plot the absorption line with all fits overlaid.
        pltname : str, optional
            If not '', saves the plot to the supplied path with whatever extension you specify. 

        Returns
        -------
            tuple
                Tuple of 3 values: the mean fitted centroid across iterations, the propagated uncertainty reported by the fitting routine, and the standard deviation
                of the centroid across all iterations. We find the latter is a good estimator of statistical uncertainty in the fitted centroid.

        '''

        window_step = -window_step
        in1 = bisect_left(wl, centroid-100)
        in2 = bisect_left(wl, centroid+100)
        cropped_wl = wl[in1:in2]
        cflux = flux[in1:in2]

        if normalize:

            cmask = (cropped_wl < centroid - 50)+(cropped_wl > centroid + 50)

            p, cov = curve_fit(self.linear, cropped_wl[cmask][~np.isnan(
                cflux[cmask])], cflux[cmask][~np.isnan(cflux[cmask])])

            contcorr = cflux / self.linear(cropped_wl, *p)
        else:
            contcorr = cflux

        linemodel = lmfit.models.GaussianModel()
        params = linemodel.make_params()
        params['amplitude'].set(value=2)
        params['center'].set(value=centroid)
        params['sigma'].set(value=5)
        centres = []
        errors = []

        if make_plot:
            plt.figure(figsize=(10, 5))
            # plt.title(str(centroid)+"$\AA$")
            plt.plot(cropped_wl, contcorr, 'k')
            #plt.plot(cropped_wl, 1-linemodel.eval(params, x = cropped_wl))

        crop1 = bisect_left(cropped_wl, centroid - half_window)
        crop2 = bisect_left(cropped_wl, centroid + half_window)
        init_result = linemodel.fit(1-contcorr[crop1:crop2], params, x=cropped_wl[crop1:crop2],
                                    nan_policy='omit', method='leastsq')

        if debug:
            plt.figure()
            plt.plot(cropped_wl[crop1:crop2], 1-contcorr[crop1:crop2])
            plt.plot(cropped_wl[crop1:crop2], linemodel.eval(
                params, x=cropped_wl[crop1:crop2]))
            plt.plot(cropped_wl[crop1:crop2], linemodel.eval(
                init_result.params, x=cropped_wl[crop1:crop2]))
            plt.show()

        #plt.plot(cropped_wl, init_result.eval(init_result.params, x = cropped_wl))

        adaptive_centroid = init_result.params['center'].value

        for ii in range(n_fit):

            crop1 = bisect_left(
                cropped_wl, adaptive_centroid - half_window - ii*window_step)
            crop2 = bisect_left(
                cropped_wl, adaptive_centroid + half_window + ii*window_step)
            try:

                result = linemodel.fit(1-contcorr[crop1:crop2], params, x=cropped_wl[crop1:crop2],
                                       nan_policy='omit', method='leastsq')
                if np.abs(result.params['center'].value - adaptive_centroid) > 5:
                    continue
            except ValueError:
                print('one fit failed. skipping...')
                continue

            if ii != 0:
                centres.append(result.params['center'].value)
                errors.append(result.params['center'].stderr)

            adaptive_centroid = result.params['center'].value

    #        print(len(cropped_wl[crop1:crop2]))
            if make_plot:
                xgrid = np.linspace(
                    cropped_wl[crop1:crop2][0], cropped_wl[crop1:crop2][-1], 1000)

                plt.plot(xgrid, 1-linemodel.eval(result.params, x=xgrid),
                         'r', linewidth=1, alpha=0.7)
    #            plt.plot(cropped_wl[crop1:crop2],1-linemodel.eval(params,x=cropped_wl[crop1:crop2]),'k--')
        mean_centre = np.mean(centres)
        sigma_sample = np.std(centres)
        if len(centres) == 0:
            centres = [np.nan]
            errors = [np.nan]
            print('caution, none of the iterated fits were succesful')
        final_centre = centres[-1]

        if None in errors or np.nan in errors:
            errors = [np.nan]

        sigma_propagated = np.nanmedian(errors)
        sigma_final_centre = errors[-1]
        total_sigma = np.sqrt(sigma_propagated**2 + sigma_sample**2)

        if make_plot:
            #         gap = (50*1e-5)*centroid
            #         ticks = np.arange(centroid - gap*4, centroid + gap*4, gap)
            #         rvticks = ((ticks - centroid) / centroid)*3e5
            #         plt.xticks(ticks, np.round(rvticks).astype(int))
            plt.xlabel('Wavelength ($\mathrm{\AA}$)')
            plt.ylabel('Flux (Normalized)')
            plt.xlim(centroid - 35, centroid + 35)
            #plt.axvline(centroid, color = 'k', linestyle = '--')
            plt.axvline(mean_centre, color='r', linestyle='--')
            plt.tick_params(bottom=True, top=True, left=True, right=True)
            plt.minorticks_on()
            plt.tick_params(which='major', length=10, width=1,
                            direction='in', top=True, right=True)
            plt.tick_params(which='minor', length=5, width=1,
                            direction='in', top=True, right=True)
            plt.xlabel('Wavelength ($\mathrm{\AA}$)')
            plt.ylabel('Normalized Flux')
            plt.tight_layout()
            # print(np.isnan(np.array(errors)))

        return mean_centre, final_centre, sigma_final_centre, sigma_propagated, sigma_sample

    def doppler_shift(self, wl, fl, dv):
        c = 2.99792458e5
        df = np.sqrt((1 - dv/c)/(1 + dv/c))
        new_wl = wl * df
        new_fl = np.interp(new_wl, wl, fl)
        return new_fl

    def xcorr_rv(self, wl, fl, temp_wl, temp_fl, init_rv=0, rv_range=500, npoint=None):
        if npoint is None:
            npoint = int(2 * rv_range)
        rvgrid = np.linspace(init_rv - rv_range, init_rv + rv_range, npoint)
        cc = np.zeros(npoint)
        for ii, rv in enumerate(rvgrid):
            shift_model = self.doppler_shift(temp_wl, temp_fl, rv)
            corr = np.corrcoef(fl, shift_model)[1, 0]
            # corr = -np.sum((fl - shift_model)**2) # MINIMIZE LSQ DIFF. MAYBE PROPAGATE IVAR HERE?
            cc[ii] = corr
        return rvgrid, cc

    def quad_max(self, rv, cc):
        maxpt = np.argmax(cc)
        max_rv = rv[maxpt]
        # in1 = maxpt - 5
        # in2 = maxpt + 5
        # rv,cc = rv[in1:in2], cc[in1:in2]
        # pfit = np.polynomial.polynomial.polyfit(rv, cc, 2)
        # max_rv = - pfit[1] / (2 * pfit[2])

        # plt.plot(rv, cc)
        # plt.axvline(max_rv)
        # plt.show()
        return max_rv

    # IMPLEMENT UNCERTAINTIES AT SPECTRUM LEVEL
    def get_one_rv(self, wl, fl, temp_wl, temp_fl, r1=1000, p1=100, r2=100, p2=100, plot=False):
        rv, cc = self.xcorr_rv(wl, fl, temp_wl, temp_fl,
                               init_rv=0, rv_range=r1, npoint=p1)

        # if plot:
        #     plt.plot(rv, cc, color = 'k', alpha = 0.1)

        rv_guess = self.quad_max(rv, cc)  # 找到数据最小值作为线心
        rv, cc = self.xcorr_rv(wl, fl, temp_wl, temp_fl,
                               init_rv=rv_guess, rv_range=r2, npoint=p2)
        if plot:
            plt.plot(rv, cc, color='k', alpha=0.1)
        return self.quad_max(rv, cc)

    def get_rv(self, wl, fl, ivar, temp_wl, temp_fl, N=100, kwargs={}):

        nans = np.isnan(fl) + np.isnan(ivar) + np.isnan(temp_fl)

        if np.sum(nans) > 0:
            print("NaNs detected in RV routine... removing them...")
            wl = wl[~nans]
            fl = fl[~nans]
            ivar = ivar[~nans]
            temp_wl = temp_wl[~nans]
            temp_fl = temp_fl[~nans]

        rv = self.get_one_rv(wl, fl, temp_wl, temp_fl, **kwargs)

        rvs = []
        for ii in range(N):
            fl_i = fl + np.sqrt(1/(ivar + 1e-10)) * \
                np.random.normal(size=len(fl))
            rvs.append(self.get_one_rv(wl, fl_i, temp_wl, temp_fl, **kwargs))
        return rv, (np.quantile(rvs, 0.84) - np.quantile(rvs, 0.16)) / 2
        
    def normalize(self,wl,fl,ivar,sfac=1,consig=10,n_order=11, splr=False,k=5,niter=0):
        # SFAC to scale rule of thumb smoothing
        s = (len(wl) - np.sqrt(2 * len(wl))) * sfac
        fl_dum = scipy.ndimage.gaussian_filter1d(fl, consig)
        if splr == True:
            spl_func = splrep(wl, fl_dum, k=k, s=s, w=np.sqrt(nivar))
            spline = splev(wl, spl_func)
            fl_norm = fl / spline
            nivar = ivar * spline**2
            for n in range(niter):
                spl_func = splrep(wl, fl_norm, k=k, s=s - 0.1 * n * s,w=np.sqrt(nivar))
                spline = splev(x, spl_func)
                fl_norm = fl_norm / spline
                nivar = nivar * spline**2
            return lambda x: splev(x,spl_func)
        else:
            spl_func = np.polyfit(wl, fl_dum, n_order)
            spline = np.polyval(spl_func, wl)
            return lambda x: np.polyval(spl_func,x)
    def spline_norm(self, wl, fl, ivar, exclude_wl=np.array([3790, 3810, 3819, 3855, 3863, 3920, 3930, 4020, 4040, 4180, 4215, 4490, 4662.68, 5062.68, 6314.61, 6814.61]), ax=None, sfac=1, k=5, plot=False, niter=0, n_order=3, splr=False, consig=10, showline=None):
        fl_prev = fl
        #fl_norm = fl / np.nanmedian(fl)
        #nivar = ivar * np.nanmedian(fl)**2
        #nivar = ivar
        #x = (wl - np.min(wl)) / (np.max(wl) - np.min(wl))
        #x = wl
        cont_mask = np.ones(len(wl))

        for ii in range(len(exclude_wl) - 1):
            if ii % 2 != 0:
                continue
            c1 = bisect_left(wl, exclude_wl[ii])
            c2 = bisect_left(wl, exclude_wl[ii + 1])

            cont_mask[c1:c2] = 0
        cont_mask = cont_mask.astype(bool)
        spl_func = self.normalize(wl[cont_mask],fl[cont_mask],ivar[cont_mask],sfac=sfac,consig=consig,k=k,niter=niter,n_order=n_order,splr=splr)
        spline = spl_func(wl)
        fl_norm = fl / spline
        nivar = ivar * spline**2
        fl_dum = scipy.ndimage.gaussian_filter1d(fl, consig)
        '''
        # SFAC to scale rule of thumb smoothing
        s = (len(wl) - np.sqrt(2 * len(wl))) * sfac
        
        if splr == True:
            spline = splev(x, splrep(
                wl[cont_mask], fl_dum[cont_mask], k=k, s=s, w=np.sqrt(nivar[cont_mask])))
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

        fl_prev = fl
        fl_norm = fl_norm / spline
        nivar = nivar * spline**2

        # Repeat spline fit with reduced smoothing. don't use without testing
        for n in range(niter):
            fl_prev = fl_norm
            spline = splev(x, splrep(x[cont_mask], fl_norm[cont_mask], k=k, s=s - 0.1 * n * s,
                                     w=np.sqrt(nivar[cont_mask])))
            fl_norm = fl_norm / spline
            nivar = nivar * spline**2
        '''
        if plot:
            from assistlgh.spectra import atom_line
            ax[0].plot(wl, fl_prev, color='k',)
            if ax is None:
                fig, ax = plt.subplots(figsize=(20,6))
                ax.plot(wl, fl_norm, color='pink', alpha=1)
                if showline is not None:
                    atom_line(wl.min(), wl.max()).showlines(wl, showline, ax)
                ax[0].plot(wl[cont_mask], fl_dum[cont_mask],color='orange',ls='--',label='convolution')
                ax[0].plot(wl, spline, color='r', label='poly fit',ls='--')
		        #ax[0].set_title('Continuum Fit (iteration %i/%i)' %(niter + 1, niter + 1), y=0.9)
                up = np.quantile(fl_prev, 0.7) + 1
                low = np.quantile(fl_prev, 0.2)-1
		        # ax[0].set_ylim(low,)
                #ax[1].plot(wl, fl_norm, color='pink', alpha=1)
                #ax[1].plot(wl[cont_mask], fl_norm[cont_mask], color='r')
                #ax[1].minorticks_on()
                #ax[1].grid(ls='--')
                #if showline is not None:
                #    atom_line(wl.min(), wl.max()).showlines(wl, showline, ax[1])
                #ax[1].set_title('Normalized Spectrum', y=0.85)
        return fl_norm.copy(), nivar.copy()
    def vel2wl(self, vel,centroid):
        cwl = centroid*np.sqrt((vel+299792458 * 1e-3)/(299792458 * 1e-3-vel))
        return cwl
    def wl2vel(self, cwl, centroid):
        vel = -299792458 * 1e-3 * (centroid**2-cwl**2)/(centroid**2+cwl**2)
        return vel

    def dwl2dvel(self, wl, centroid, dwl):
        dvel = -dwl*(4*wl*centroid**2)/(centroid**2+wl**2)**2
        return dvel
    def grav_shift(self,wl,M,r0):
        M*=u.M_sun
        r0*=u.R_sun
        return (1+(c.G*M/c.c**2/r0).to(1).value)*wl
        
    def get_line_rv(self, wl, fl, ivar, centroid,norm=True,template=None, return_template=False, distance=50, edge=15, nmodel=2, plot=False, rv_kwargs={}, init_sep=10,init_width=[1,0.3], init_amp=5, model='Voigt1D',mcmc=True):
        # print(centroid,type(centroid))
        c1 = bisect_left(wl, centroid - distance)
        c2 = bisect_left(wl, centroid + distance)
        lower = centroid - distance + edge
        upper = centroid + distance - edge

        cwl, cfl, civar = wl[c1:c2], fl[c1:c2], ivar[c1:c2]

        edgemask = (cwl < lower) + (cwl > upper)
        if norm:
            line = np.polyval(np.polyfit(cwl[edgemask], cfl[edgemask], 1), cwl)
            nfl = cfl / line-1
            nivar = civar * line**2
        else:
            nfl = cfl
            nivar = civar

        #vel = 299792458 * 1e-3 * (cwl - centroid) / centroid
        vel = self.wl2vel(cwl, centroid)
        df = np.sqrt((1-1100/(299792458 * 1e-3)) / (1+1100/(299792458 * 1e-3)))
        if template is None:
            if model == 'Voigt':
                for ii in range(nmodel):
                    if ii == 0:
                        model = VoigtModel(prefix='g' + str(ii) + '_')
                    else:
                        model += VoigtModel(prefix='g' + str(ii) + '_')

                params = model.make_params()
                init_center = [centroid+init_sep, centroid-init_sep]

                # print(init_center)
                edgecenter = 1100/(299792458 * 1e-3)*centroid+centroid
                for ii in range(nmodel):
                    params['g' + str(ii) +
                           '_center'].set(value=init_center[ii], vary=True, min=centroid*df, max=centroid/df)
                    params['g' + str(ii) +
                           '_sigma'].set(value=init_width, vary=True, min=0.1)
                    params['g' + str(ii) +
                           '_amplitude'].set(value=init_amp/nmodel, vary=True)

                #params['g0_center'].set(value = init_center, vary = True, expr = None)

                res = model.fit(nfl, params, x=cwl,
                                method='leastsq', max_nfev=10000)

                # res.params['g0_center'].set(value=centroid)
                # res.params['g1_center'].set(value=centroid)
                rv_kwargs.update({'init_rv': [self.wl2vel(res.params['g0_center'].value, centroid), self.wl2vel(
                    res.params['g1_center'].value, centroid)]})

                template = model.eval(res.params, x=cwl)
                sing_temp = []

                rvs = []
                e_rvs = []
                for ii in range(nmodel):
                    single_model = VoigtModel(prefix='g' + str(ii) + '_')
                    mask = ['g'+str(ii) in i for i in res.params.keys()]
                    key = np.array(list(res.params.keys()))
                    parameters = lmfit.Parameters()
                    for i in range(len(key[mask])):
                        parameters.add(res.params[key[mask][i]])
                    sing_temp.append(single_model.eval(parameters, x=cwl))
                #rvs, e_rvs = self.get_rv_bin(cwl, nfl, nivar, cwl, sing_temp, kwargs=rv_kwargs)
                rvs = [self.wl2vel(res.params['g0_center'].value, centroid), self.wl2vel(
                    res.params['g1_center'].value, centroid)]
                e_rvs = [self.dwl2dvel(res.params['g0_center'].value, centroid, res.params['g0_center'].stderr), self.dwl2dvel(
                    res.params['g1_center'].value, centroid, res.params['g1_center'].stderr)]
            elif model == 'Lorentzian':
                for ii in range(nmodel):
                    if ii == 0:
                        model = LorentzianModel(prefix='g' + str(ii) + '_')
                    else:
                        model += LorentzianModel(prefix='g' + str(ii) + '_')

                params = model.make_params()
                init_center = centroid

                # print(init_center)

                for ii in range(nmodel):
                    params['g' + str(ii) +
                           '_center'].set(value=init_center, vary=True)
                    params['g' + str(ii) +
                           '_sigma'].set(value=init_width, vary=True)
                    params['g' + str(ii) +
                           '_amplitude'].set(value=init_amp/nmodel, vary=True)

                #params['g0_center'].set(value = init_center, vary = True, expr = None)

                res = model.fit(nfl, params, x=cwl, method='nelder')

                #res.params['g0_center'].set(value = centroid)
                
                sing_temp = []

                rvs = []
                e_rvs = []
                for ii in range(nmodel):
                    single_model = LorentzianModel(prefix='g' + str(ii) + '_')
                    mask = ['g'+str(ii) in i for i in res.params.keys()]
                    key = np.array(list(res.params.keys()))
                    parameters = lmfit.Parameters()
                    for i in range(len(key[mask])):
                        parameters.add(res.params[key[mask][i]])
                    sing_temp.append(single_model.eval(parameters, x=cwl))
                    rv_sing, e_rv_sing = self.get_rv(
                        cwl, nfl, nivar, cwl, sing_temp[ii], **rv_kwargs)
                    rvs.append(rv_sing)
                    e_rvs.append(e_rv_sing)
            elif model == 'Voigt1D':

                from astropy.modeling import fitting
                from astropy.modeling.models import Voigt1D
                param_names = ['x_0','amplitude_L', 'fwhm_L', 'fwhm_G']
                bounds = {'x_0': [centroid*df, centroid/df],
                          'amplitude_L': [-20, 20], 'fwhm_L': [0.1, 10], 'fwhm_G': [0.1, 50]}
                if nmodel == 1:
                    init_center = centroid+init_sep
                    v1 = Voigt1D(x_0=init_center, amplitude_L=init_amp /
                                 nmodel, fwhm_L=init_width[0], fwhm_G=init_width[1], bounds=bounds)
                    fit_g = fitting.LevMarLSQFitter(True)
                    g = fit_g(v1, cwl, nfl, weights=np.sqrt( 
                        nivar), maxiter=10000)
                    para = g._parameters
                    print(para)
                    if mcmc:
                        para,redchi,stds,upstd,downstd = mcmc_fit(para,cwl,nfl,nivar,Voigt1D,param_names,nwalkers=100,burn=3000,ndraws=2000)
                    wl_dense=np.linspace(cwl.min(),cwl.max(),1000)
                    sing_temp = [Voigt1D(*para)(wl_dense)]
                    rvs = [self.wl2vel(para[0], centroid)]
                    e_rvs = [0]
                elif nmodel == 2:
                    init_center = [centroid+init_sep, centroid-init_sep]
                    v1 = Voigt1D(x_0=init_center[0], amplitude_L=init_amp/nmodel, fwhm_L=0.5, fwhm_G=init_width, bounds=bounds)+Voigt1D(
                        x_0=init_center[1], amplitude_L=init_amp/nmodel, fwhm_L=0.5, fwhm_G=init_width, bounds=bounds)
                    fit_g = fitting.LevMarLSQFitter(True)
                    g = fit_g(v1, cwl, nfl, weights=np.sqrt(
                        nivar), maxiter=10000)
                    #print(g)
                    para1, para2 = g._parameters.reshape(2, -1)
                    sing_temp = [Voigt1D(*para1)(cwl),
                                 Voigt1D(*para2)(cwl)]
                    rvs = [self.wl2vel(para1[0], centroid),
                           self.wl2vel(para2[0], centroid)]
                    e_rvs = [0, 0]
            elif model == 'Gaussian':
                for ii in range(nmodel):
                    if ii == 0:
                        model = GaussianModel(prefix='g' + str(ii) + '_')
                    else:
                        model += GaussianModel(prefix='g' + str(ii) + '_')

                params = model.make_params()
                init_center = centroid

                # print(init_center)

                for ii in range(nmodel):
                    params['g' + str(ii) +
                           '_center'].set(value=init_center, vary=True)
                    params['g' + str(ii) +
                           '_sigma'].set(value=init_width, vary=True)
                    params['g' + str(ii) +
                           '_amplitude'].set(value=init_amp/nmodel, vary=True)

                #params['g0_center'].set(value = init_center, vary = True, expr = None)

                res = model.fit(nfl, params, x=cwl, method='leastsq')

                #res.params['g0_center'].set(value = centroid)
                sing_temp = []
                rvs = []
                e_rvs = []
                for ii in range(nmodel):
                    single_model = GaussianModel(prefix='g' + str(ii) + '_')
                    mask = ['g'+str(ii) in i for i in res.params.keys()]
                    key = np.array(list(res.params.keys()))
                    parameters = lmfit.Parameters()
                    for i in range(len(key[mask])):
                        parameters.add(res.params[key[mask][i]])
                    sing_temp.append(single_model.eval(parameters, x=cwl))
                    rv_sing, e_rv_sing = self.get_rv(
                        cwl, nfl, nivar, cwl, sing_temp[ii], **rv_kwargs)
                    rvs.append(rv_sing)
                    e_rvs.append(e_rv_sing)
        if plot:
            #plt.plot(vel, self.doppler_shift(cwl, template, rv), 'r')
            #title = 'RV = %.1f ± %.1f km/s' % (rv, e_rv)
            plt.subplots()
            plt.plot(vel, nfl, 'k')
            title = ''
            for ii in range(nmodel):
                if ii == 0:
                    total = sing_temp[ii]
                else:
                    total += sing_temp[ii]
                #plt.plot(vel, self.doppler_shift(cwl,sing_temp[ii], rvs[ii]), ls='--')
                vel_dense = self.wl2vel(wl_dense, centroid)
                plt.plot(vel_dense,sing_temp[ii], ls='--')
                #title += ' RV%.1d = %.1f ± %.1f km/s' % (ii+1, rvs[ii], e_rvs[ii])
                #plt.axvline(rvs[ii], linestyle='--')
                title += r' RV%.1d = %.1f km/s@%.2f $\AA$' % (ii+1, rvs[ii],centroid)                 
            plt.plot(vel_dense,total, c='r')
            plt.xlabel('Relative Velocity')
            plt.ylabel('Normalized Flux')
            plt.title(title) 
            plt.axvline(0, color='k', linestyle='--')

        if return_template:
            return rvs, e_rvs, total
        else:
            return rvs, e_rvs

    def Planck(x, t):
        hc = 6.62607015e-34*2.99792e8
        k = 1.3806505e-23
        lamda = x*1e-10
        res = 8*np.pi*hc*lamda**(-5)*(np.exp(hc*(lamda*k*t)**(-1))-1)**(-1)
        return res

    def xcorr_rv_bin(self, wl, fl, temp_wl, temp_fl, init_rv=0, rv_range=500, npoint=None):
        if npoint is None:
            npoint = int(2 * rv_range)
        rvrange1 = np.linspace(
            init_rv[0] - rv_range, init_rv[0] + rv_range, npoint)
        rvrange2 = np.linspace(
            init_rv[1] - rv_range, init_rv[1] + rv_range, npoint)
        rvgrid = np.array(np.meshgrid(rvrange1, rvrange2)).T.reshape(-1, 2)
        cc = np.zeros(len(rvgrid))
        for ii, rv in enumerate(rvgrid):
            shift_model = self.doppler_shift(
                temp_wl, temp_fl[0], rv[0])+self.doppler_shift(temp_wl, temp_fl[1], rv[1])
            corr = np.corrcoef(fl, shift_model)[1, 0]
            # corr = -np.sum((fl - shift_model)**2) # MINIMIZE LSQ DIFF. MAYBE PROPAGATE IVAR HERE?
            cc[ii] = corr
        return rvgrid, cc

    def quad_max_bin(self, rv, cc):
        maxpt = np.argmax(cc)
        max_rv = rv[maxpt]
        return max_rv

    def get_one_rv_bin(self, wl, fl, temp_wl, temp_fl, r1=100, p1=50, r2=100, p2=50, init_rv=0, plot=False):
        rv, cc = self.xcorr_rv_bin(wl, fl, temp_wl, temp_fl,
                                   init_rv=init_rv, rv_range=r1, npoint=p1)
        rv_guess = self.quad_max_bin(rv, cc)  # 找到数据最小值作为线心
        rv, cc = self.xcorr_rv_bin(wl, fl, temp_wl, temp_fl,
                                   init_rv=rv_guess, rv_range=r2, npoint=p2)
        if plot:
            plt.plot(rv, cc, color='k', alpha=0.1)
        return self.quad_max_bin(rv, cc)

    def get_rv_bin(self, wl, fl, ivar, temp_wl, temp_fl, N=100, kwargs={}):

        nans = np.isnan(fl) + np.isnan(ivar) + np.isnan(temp_fl)

        if np.sum(nans) > 0:
            print("NaNs detected in RV routine... removing them...")
            wl = wl[~nans]
            fl = fl[~nans]
            ivar = ivar[~nans]
            temp_wl = temp_wl[~nans]
            temp_fl = temp_fl[~nans]

        rv = self.get_one_rv_bin(wl, fl, temp_wl, temp_fl, **kwargs)

        rvs = []
        for ii in range(N):
            fl_i = fl + np.sqrt(1/(ivar + 1e-10)) * \
                np.random.normal(size=len(fl))
            rvs.append(self.get_one_rv_bin(
                wl, fl_i, temp_wl, temp_fl, **kwargs))
        return rv, (np.quantile(rvs, 0.84, axis=0) - np.quantile(rvs, 0.16, axis=0)) / 2

def mcmc_fit(mle,wl,fl,ivar,model,param_names,nwalkers,burn,ndraws,progress=True,plot_corner=True,verbose=True,savename=None,**sampler_kw):
    def lnlike(prms):
        resid = 100*(fl- model(*prms)(wl))**2 * np.sqrt(ivar)
        chisq = np.sum(resid)
        lnlike = -0.5 * chisq
        return lnlike
    def lnprior(prms):
        if (prms[0]>10000 or prms[0]<3000) or (prms[1]<-20 or prms[1]>20) or (prms[2]<0.1 or prms[2]>10) or (prms[3]<0.1 or prms[3]>50):
            return -np.inf
        return 0
    def lnprob(prms):
        lp = lnprior(prms)
        res = lp + lnlike(prms)
        return res
    
    ndim = len(mle)
    sampler = emcee.EnsembleSampler(nwalkers,ndim, lnprob, **sampler_kw)
    pos0 = np.zeros((nwalkers, ndim))
    if verbose:
        print('using 0.01 err')
    sigmas = np.abs(1e-2*np.array(mle))  # USE 1% ERROR
    for jj in range(ndim):
        pos0[:, jj] = np.random.normal(loc=mle[jj],scale=3*np.abs(sigmas[jj]),size=nwalkers)
    if verbose:
        print('burning in chains...')
    b = sampler.run_mcmc(pos0, burn, progress=progress)
    sampler.reset()
    if verbose:
        print('sampling posterior...')
    b = sampler.run_mcmc(b.coords, ndraws, progress=progress)
    print('acceptance fraction: {:8.3f}'.format(
        sampler.acceptance_fraction[0]))
    if plot_corner:
        f, axs = plt.subplots(ndim, 1, figsize=(10, 6))
        for jj in range(ndim):
            axs[jj].plot(sampler.chain[:, :, jj].T,
                            alpha=0.3, color='k')
            plt.ylabel(param_names[jj])
        plt.xlabel('steps')
        if savename is not None:
            plt.savefig(fig_path+savename + '+'.join(self.specclass)+'_trace.jpg',bbox_inches='tight',pad_inches=0.0, dpi=100)
    lnprobs = sampler.get_log_prob(flat=True)
    cov_matrix = np.cov(sampler.flatchain, rowvar=False)
    mle = sampler.flatchain[np.argmax(lnprobs)]
    print('lnprobs:',np.max(lnprobs),mle,np.argmax(lnprobs))
    redchi = -2 * np.max(lnprobs) / (len(wl) - ndim)
    stds = np.std(sampler.flatchain, 0)
    #self.flatchain = sampler.flatchain
    upstd = []
    downstd = []
    for i in range(sampler.flatchain.shape[1]):
        upstd.append(np.percentile(sampler.flatchain[:,i],q=84)-np.percentile(sampler.flatchain[:,i],q=50))
        downstd.append(np.percentile(sampler.flatchain[:,i],q=50)-np.percentile(sampler.flatchain[:,i],q=16))
    upstd = np.array(upstd)
    downstd=np.array(downstd)
    if plot_corner:
        f = corner.corner(sampler.flatchain, labels=param_names[:ndim],
                            label_kwargs=dict(fontsize=12), quantiles=(0.16, 0.5, 0.84),
                            show_titles=True, title_kwargs=dict(fontsize=12))
        for ax in f.get_axes():
            ax.tick_params(axis='both', labelsize=12)
        if savename is not None:
            plt.savefig(fig_path+savename + '+'.join(self.specclass)+'_corner.jpg',bbox_inches='tight',pad_inches=0.0, dpi=100)
    return mle,redchi,stds,upstd,downstd

if __name__ == '__main__':
    sp = SpecTools()
    ha = 6562.76701
    sa = sp.vel2wl(-31.7,ha)/ha-1
    hb=4861.296711
    sb = sp.vel2wl(-62.25731201690481,hb)/hb-1
    hc=4340.437554
    sc = sp.vel2wl(-72.6355978111336,hc)/hc-1
    hd=4101.710277
    sd = sp.vel2wl(-65.53811252372914,hd)/hd-1
    he=3971.2
    se = sp.vel2wl(-208.21155063999944,he)/he-1
    hf=3890.12
    sf = sp.vel2wl(-208.24295580229284,hf)/hf-1
    
    print(sa,sb,sc,sd,se)
    print(sp.wl2vel(6000+0.8895053146435609,6000))