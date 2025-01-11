
from assistlgh.spectra import planck
import re
import tensorflow as tf
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
from tensorflow.keras.optimizers import Adamax
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Input, LeakyReLU
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import gc
import inspect
import os
import pickle
import pandas as pd
from lmfit.models import VoigtModel
from scipy.optimize import curve_fit
from bisect import bisect_left
import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# from astroNN.nn import layers as annlayers

# from .spectrum import SpecTools


class NN:
    '''

    Base class for a simple convolutional neural network. Work in progress, do not use.

    '''

    def __init__(self, types,n_input=4, n_output=4000, n_hidden=3, neurons=256,
                 output_activation='linear', regularization=0, loss='mse', dropout=0.1,
                 model='bayesnn', scfile='sdobspec_sc.txt', wl_min=1300, wl_max=12000, n_label=4, learning_rate=1e-3):
        '''

        Args:
                n_input (int): number of inputs
                n_output (int): number of outputs
                n_hidden (int): number of hiddden dense layers
                neurons (int): number of neurons per hidden layers
                activation (str): hidden layer activation function (keras format)
                output_activation (str): output activation function (keras format)
                regularization (float): l2 regularization hyperparameter
                loss (str): loss function (keras format)
                dropout (float): dropout fraction for bayesian inference
                input_bounds (list): list of tuple bounds for each input
                output_bounds (list): list of tuple bounds for each output
                model (str): use a built-in preloaded model

        '''
        self.types = types
        self.Tgrid = {}
        self.ggrid = {}
        self.Tgrid["DA"] = [5000, 80000]
        self.ggrid["DA"] = [6.5, 9.5]
        self.Tgrid[ "MS"] = [3000, 26000]
        self.ggrid[ "MS"] = [-0.5, 5.5]
        self.Tgrid["DB"] = [8000, 40000]
        self.ggrid["DB"] = [7.0, 9.5]
        self.Tgrid["sdob"] = [30000, 70000]
        self.ggrid["sdob"] = [4.5, 7.0]

        self.is_initialized = False
        self.n_input = n_input
        self.n_output = n_output
        self.n_hidden = n_hidden
        self.output_activation = output_activation
        self.reg = regularization
        self.neurons = neurons
        self.loss = loss
        self.dropout = dropout
        self.scaler_isfit = False
        self.lamgrid = np.linspace(wl_min, wl_max, n_input)
        self.n_label = n_label
        self.learning_rate = learning_rate
        self.model = self.gen_nn()
        self.spec_min, self.spec_max = np.loadtxt(model_dir+scfile)

    def gen_nn(self):

        x = Input(shape=self.n_input, name='Input')

        y = Dense(self.neurons, activation=LeakyReLU(alpha=0.01))(x)
        y = Dense(self.neurons, activation=LeakyReLU(alpha=0.01))(y)
        y = Dense(self.neurons, activation=LeakyReLU(alpha=0.01))(y)

        # y = Dropout(self.dropout)(y, training = True)

        out = Dense(self.n_output,
                    activation=self.output_activation, name='Output')(y)

        network = Model(inputs=x, outputs=out)
        network.compile(optimizer=Adamax(
            learning_rate=self.learning_rate), loss=self.loss)
        return network

    def sequential(self):
        model = Sequential()
        model.add()

    def train(self, x_data, y_data, model='default', n_epochs=100, batchsize=64, verbose=0):

        x_data = self.label_sc(x_data.reshape(len(x_data), self.n_input))
        #x_data = x_data.reshape(len(x_data), self.n_input)
        #y_data = y_data/np.mean(y_data)
        # 将数据集分为， 训练集与测试
        X_train_full, X_test, y_train_full, y_test = train_test_split(
            x_data, y_data, test_size=0.1)
        #X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2)
        # 将总训练集分为，训练集与验证集
        X_train, X_valid, y_train, y_valid = train_test_split(
            X_train_full, y_train_full, test_size=0.25)
        del x_data, y_data
        gc.collect()

        # 模型参数保存路径
        checkpoint_save_path = dir_path+"/checkpoint.ckpt"
        if os.path.exists(checkpoint_save_path + ".index"):
            self.model.load_weights(checkpoint_save_path)
            print('model loaded!')
        cp_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_save_path, save_weights_only=True, save_best_only=True)
        h = self.model.fit(X_train, y_train, epochs=n_epochs,
                           verbose=verbose, validation_data=(X_valid, y_valid), batch_size=batchsize,callbacks=[cp_callback])
        mse_test = self.model.evaluate(X_test, y_test)
        print(mse_test)
        return h

    def eval_data(self, x_data, model='default', n_bootstrap=100):

        x_data = x_data.reshape(len(x_data), self.n_input, 1)

        if self.bayesian:
            predictions = np.asarray(
                [self.inv_label_sc(self.model.predict(x_data)) for i in range(n_bootstrap)])
            means = np.nanmean(predictions, 0)
            stds = np.nanstd(predictions, 0)
            results = np.empty(
                (means.shape[0], means.shape[1] + stds.shape[1]), dtype=means.dtype)
            results[:, 0::2] = means
            results[:, 1::2] = stds
            return results

        elif not self.bayesian:
            return self.inv_label_sc(self.model.predict(x_data))

    def labels_from_spectrum(self, wl, flux):
        func = interp1d(wl, flux)
        interp_flux = func(self.lamgrid)
        return self.eval_data(interp_flux.reshape(1, -1))[0]

    def save(self, modelname):
        self.model.save_weights(dir_path + modelname + '.h5')
        print('model saved!')

    def load(self, modelname):
        self.model.load_weights(dir_path + modelname + '.h5')
        print('model loaded!')

    def args(self):
        print(self.__init__.__doc__)
        return None

    def label_sc(self, label_array):
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
        types = self.types
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

    def inv_label_sc(self, label_array,):
        """
        Inverse label scaler to transform Teff and logg from [0,1] to original scale based on preset bounds.

        Parameters
        ---------
        label_array : array
                Scaled array with Teff in the first column and logg in the second column
        Returns
        ------- b   
                array
                        Unscaled array
        """
        types=self.types
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


def makedata():
    model_dir = 'D:/anaconda/Lib/site-packages/wdtools-0.4-py3.7.egg/wdtools/models/coelho_highres/'
    filelist = list(os.walk(model_dir))[0][2]
    x_data, y_data = [], []
    for i in range(len(filelist)):
        with open(model_dir+filelist[i]) as f:
            table = f.readlines()[1:5]
            label = []
            for item in table:
                ii = item.split('(')[0].split('=')
                value = np.float(re.findall('-*\d*.*\d', ii[-1])[0])
                label.append(value)
        f = pd.read_csv(model_dir+filelist[i],
                        skiprows=8, delimiter=' +', header=None, engine='python')
        f.columns = ['wavelength', 'flux']
        #y = (f['flux']-f['flux'].min())
        # /(f['flux'].max()-f['flux'].min())
        # y_data.append(f['flux'])
        # y_data.append(y)
        x_data.append(label)
    np.save('D:/anaconda/Lib/site-packages/wdtools-0.4-py3.7.egg/wdtools/models/MS_wl',
            f['wavelength'][::10])
    np.save(
        'D:/anaconda/Lib/site-packages/wdtools-0.4-py3.7.egg/wdtools/models/MS_x', x_data[::10])
    #np.save('D:/anaconda/Lib/site-packages/wdtools-0.4-py3.7.egg/wdtools/models/MSm_scaled_y_data', y_data[::10])

    return


def test():
    model_dir = '/home/my/squareroot/Wd/wdtools/models/neural_gen/'
    wl = np.load(model_dir+'sdobp252wl.npy')
    x_data = np.load(model_dir+'sdobp252label.npy')
    y_data = np.load(model_dir+'sdobp252y.npy')
    print(y_data.shape, x_data.shape, wl.shape)
    nn = NN(n_input=x_data.shape[1], n_output=y_data.shape[1], n_hidden=3,
            output_activation='linear', regularization=0, loss='mse', dropout=0.1, n_label=x_data.shape[1], learning_rate=1e-3, scfile='sdobp252spec_sc.txt')
    nn.model.summary()
    y_data = (y_data-nn.spec_min)/(nn.spec_max-nn.spec_min)
    h = nn.train(x_data, y_data, n_epochs=1000,
                 batchsize=32, verbose=1)
    nn.save('sdobp252')
    l = h.history['loss']
    plt.plot(l)
    plt.savefig(model_dir+'sdobp252_loss.png')
    return h


def test_result():
    model_dir = '/home/my/squareroot/Wd/wdtools/models/neural_gen/'
    l = np.load(model_dir+'sdobp252label.npy')
    wl = np.load(model_dir+'sdobp252wl.npy')
    f = np.load(model_dir+'sdobp252y.npy')
    nn = NN(n_input=l.shape[1], n_output=f.shape[1], n_hidden=3,
            output_activation='linear', regularization=0, loss='mse', dropout=0.1, n_label=l.shape[1], scfile='sdobp252spec_sc.txt')
    nn.model.load_weights(model_dir+'/'+'sdobp252.h5')
    index = np.random.randint(len(l))
    f = f[index]
    #T, g = l[index]
    print('para:', l[index])
    y = nn.model.predict(nn.label_sc(np.array([l[index]])))
    #scaled = f/(planck(wl, [T])[0]*1e-9)-1
    #scaled = (f-nn.spec_min)/(nn.spec_max-nn.spec_min)
    scaled = f
    y = y*(nn.spec_max-nn.spec_min)+nn.spec_min
    fig, ax = plt.subplots(2, 1, figsize=(10, 6), gridspec_kw={
                           'height_ratios': [3, 1]}, sharex=True)
    ax[0].plot(wl, y[0], label='NN')
    #f['flux'] /= np.mean(f['flux'])
    ax[0].plot(wl, scaled, label='TMAP')
    mask = np.logical_and(wl > 3600, wl < 9500)
    residual = (y[0]-scaled)[mask]
    wl = wl[mask]
    ax[1].plot(wl, residual)
    Rs = 1-np.sum(residual**2)/np.sum((scaled-np.mean(scaled))**2)
    print(Rs)

    sigma = np.sqrt(np.sum((residual-np.mean(residual))**2)/len(residual))
    print(3*sigma)
    ax[1].set_xlabel(r"$\lambda(\AA)$")
    ax[1].hlines(3*sigma, wl.min(), wl.max(), ls='--', color='k', lw=1)
    ax[1].hlines(0, wl.min(), wl.max(), color='k', lw=1)
    ax[1].hlines(-3*sigma, wl.min(), wl.max(), ls='--', color='k', lw=1)
    ax[0].set_ylabel(r'Flux($erg/cm^2/s/cm$)')
    ax[1].set_ylabel(r'residual')
    ax[0].set_xlim(3600, 9500)
    ax[0].legend()
    #ax[0].text(ax[0].get_xticks()[-3], ax[0].get_yticks()[2], r'$T_{eff} = %d  K$' % T)
    #ax[0].text(ax[0].get_xticks()[-3], ax[0].get_yticks()[3], r'log g = %.1f log(cm/$s^{2}$)' % g)
    plt.tight_layout(h_pad=0)
    plt.savefig(model_dir+'/sdobp252fig.png')

    labels = np.load(model_dir+'sdobp252label.npy')
    wl = np.load(model_dir+'sdobp252wl.npy')
    f = np.load(model_dir+'sdobp252y.npy')
    #f = (f-np.min(f,axis=1)[:,None])/((np.max(f,axis=1)-np.min(f,axis=1))[:,None])
    #f = (f - planck(wl, labels[:,0])* 1e-9)/(planck(wl, labels[:,0])*1e-9)
    f = (f-nn.spec_min)/(nn.spec_max-nn.spec_min)
    index = np.random.randint(len(wl))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(r'high resolution scaled predict @$\lambda$ = ' +
                 str(wl[index])+r"$\AA$")

    index = np.random.randint(len(wl))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(r'high resolution scaled predict @$\lambda$ = ' +
                 str(wl[index])+r"$\AA$")
    # g
    T = 50000.
    mask = l[:, 0] == T
    grid = l[mask][:, 1]
    print(grid)
    grid_dense = np.linspace(grid.min(), grid.max(), 1000).reshape(-1, 1)
    tmp2 = np.array([[T]]*1000)
    params = np.hstack((tmp2, grid_dense))
    string = r'$log g$'
    y = nn.model.predict(nn.label_sc(params))[:, index]
    ax = axes[1]
    ax.plot(grid_dense.reshape(-1), y, c='r', lw=1, label='NN')
    ax.scatter(grid, f[mask].T[index], label='Model', c='k', s=5)
    ax.set_xlabel(string)
    ax.set_ylabel(r' Relative Flux')
    ax.legend()
    #ax.text(ax.get_xticks()[4],ax.get_yticks()[4],r'$\lambda$ = '+str(wl[index])+r"$\AA$")
    # T
    logg = 6
    mask = l[:, 1] == logg
    grid = l[mask][:, 0]
    print(grid)
    grid_dense = np.linspace(grid.min(), grid.max(), 1000).reshape(-1, 1)
    tmp1 = np.array([[logg]]*1000)
    params = np.hstack((grid_dense, tmp1))
    y = nn.model.predict(nn.label_sc(params))[:, index]
    string = r"$T_{eff}$"
    ax = axes[0]
    ax.plot(grid_dense.reshape(-1), y, c='r', lw=1, label='NN')
    ax.scatter(grid, f[mask].T[index], label='Model', c='k', s=5)
    ax.set_xlabel(string)
    ax.set_ylabel(r'Relative Flux')
    ax.legend()
    #ax.text(ax.get_xticks()[4],ax.get_yticks()[4],r'$\lambda$ = '+str(wl[index])+r"$\AA$")
    plt.savefig(model_dir+'sdobp252para.png')


#path = os.path.abspath(__file__)
path = '/home/my/squareroot/Wd/wdtools/models/neural_gen/'
dir_path = os.path.dirname(path)
model_dir = '/home/my/squareroot/Wd/wdtools/models/neural_gen/'
if __name__ == '__main__':
    #model = NN().gen_nn()
    # model.summary()
    # makedata()
    #test()
    test_result()
    # check_data()
    #path = os.path.abspath(__file__)
    #dir_path = os.path.dirname(path)
    # print(dir_path)
