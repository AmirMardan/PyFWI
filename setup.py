# from distutils.core import setup 
from setuptools import setup
from setuptools.extension import Extension

with open("README.md", "r") as fh:
    long_description = fh.read()
              
setup(name='PyFWI',
      version='0.1.5',
      packages=['PyFWI'],
      description='The initial package for seismic FWI', 
      long_description=long_description,
      long_description_content_type="text/markdown",
      author='Amir Mardan',
      author_email='mardan.ah69@gmail.com',
      url = 'https://github.com/AmirMardan/PyFWI',
      project_urls={
        "Bug Tracker": "https://github.com/AmirMardan/PyFWI/issues",
        },
      classifiers=[
          'Intended Audience :: Education',
          'Programming Language :: C',
          'Programming Language :: Python :: 3.7',
          'Programming Language :: Python :: 3.10',
          'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',
          'Operating System :: OS Independent',
          'Operating System :: Microsoft :: Windows',
          'Intended Audience :: Education',
          'Intended Audience :: Science/Research',
                   ],
      package_dir={"": "src"},
      install_requires=[
          "setuptools>=42",
          "wheel",
          "numpy",
          "pyopencl==2021.2.10",
          "matplotlib",
          "scipy",
          "hdf5storage",
          "requests",
          "segyio",
          "datetime"
      ],
      package_data={
        "PyFWI": ["elastic.cl",
                  "elastic_crosswell.cl",
                  "elastic_surface.cl"
                  ],
        },
      zip_safe=False,
      python_requires=">=3.7"
      
      )
     

