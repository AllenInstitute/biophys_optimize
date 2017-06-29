from setuptools import setup, find_packages
import biophys_optimize

setup(
    version = biophys_optimize.__version__,
    name = 'biophys_optimize',
    author = 'Nathan Gouwens',
    author_email = 'nathang@alleninstitute.org',
    packages = find_packages(),
    package_data = {
        'biophys_optimize': [ 'passive/*', 
                              'fit_styles/*.json',
                              'modfiles/*.mod',
                              '*.hoc' ],
        
        },
    include_package_data = True,
    install_requires=['deap', 'neuron', 'numpy', 'allensdk' ]
)
