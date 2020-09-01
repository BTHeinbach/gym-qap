from setuptools import find_packages, setup

setup(name='gym_qap',
      version='0.0.1',
      install_requires=['gym', 'numpy'],
      packages=find_packages(),
      include_package_data=True
)
