from setuptools import find_packages, setup

with open("README.md", 'r') as fh:
    long_description = fh.read()

setup(
    name='cycle-wgan',
    version='0.9.0',
    author='Ben Talbot',
    author_email='b.talbot@qut.edu.au',
    url='https://github.com/best-of-acrv/cycle-wgan',
    description=
    'Generalised zero-shot learning (GZSL) semantic classification using '
    'multi-modal cycle-consistent feature generation',
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=find_packages(),
    package_data={'cycle_wgan': ['*.json']},
    install_requires=['acrv_datasets'],
    entry_points={'console_scripts': ['cycle-wgan=cycle_wgan.__main__:main']},
    classifiers=(
        "Development Status :: 4 - Beta",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ))
