from setuptools import setup, find_packages
from codecs import open
from os import path

__version__ = '0.0.1'


here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='ivis_explanations',
    version=__version__,
    description='collection of routines to explain ivis embeddings',
    long_description=long_description,
    url='https://github.com/beringresearch/ivis-explain',
    license='Apache 2.0',
    classifiers=[
      'Development Status :: 3 - Alpha',
      'Intended Audience :: Developers',
      'Programming Language :: Python :: 3',
    ],
    keywords='python data science machine learning pandas sklearn ivis',
    packages=find_packages(exclude=['tests*']),
    include_package_data=True,
    author='Bering Limited',
    install_requires=[
        'scikit-learn>=0.20.0',
        'ivis>=1.1.4'
    ],
    author_email='info@beringresearch.com'
)
