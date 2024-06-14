from setuptools import setup, find_packages

setup(
    name='moudila_regpoisson',
    version='0.1',
    description='A Python module for Poisson regression',
    author='Marcel Moudila',
    author_email='moudilamarcel@gmail.com',
    url='https://github.com/marcel-moudila/moudila_regpoisson',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'scipy',
        'scikit-learn'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
