from setuptools import setup, find_packages

# read the README
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="binary_tools",  # package name
    version="1.0.0",  # version
    author="Genghao Liu",
    author_email="Squarerootof6@outlook.com",
    description="DA,DB,MS binary spectra fitting and mass measurement tools",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Squarerootof6/BinaryTools.git",  # project page
    packages=find_packages(),  # locate packages automatically
    include_package_data=True,  # include files listed in MANIFEST.in
    package_data={
        # release ships model files only (weights, wavelength grids, scaling tables);
        # training arrays live in ../training_data and are deliberately not packaged
        'binary_tools': [
            'atomic_lines.csv',
            'models/*.npz',
            'models/neural_gen/*.h5',
            'models/neural_gen/*.npy',
            'models/neural_gen/*.txt',
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.9,<3.12',  # TensorFlow 2.14 ships no Python 3.12 wheels
    install_requires=[
        # validated: Python 3.11.5 / NumPy 1.26.4 / TensorFlow 2.14.0 / Keras 2.14
        "numpy>=1.23.5,<2.0",      # TF 2.14 is built against NumPy < 2
        "tensorflow==2.14.0",
        "keras==2.14.0",
        "h5py>=3.8,<4",            # legacy .h5 weight loading
        "scipy>=1.9",
        "astropy>=5.0",
        "matplotlib>=3.5",
        "pandas>=1.5",
        "lmfit>=1.2",
        "emcee>=3.0",
        "corner>=2.2",
    ],
)
