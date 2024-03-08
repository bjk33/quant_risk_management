from setuptools import find_packages, setup

setup(
    name='quant_risk_lib',
    packages=find_packages(include=['quant_risk_lib']),
    version='0.1.11',
    description='My FINTECH545 Python library',
    author='Brandon Kaplan',
    install_requires=['numpy', 'pandas', 'scipy', 'matplotlib', 'statsmodels']
)