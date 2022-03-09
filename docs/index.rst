.. PyFWI documentation master file, created by
   sphinx-quickstart on Sat Jan 29 00:58:32 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to PyFWI's documentation!
=================================

**PyFWI** is an open source Python package to perform seismic modeling and full-waveform inversion (FWI) in elastic media. 
This packages is implemented in time domain and coded using GPU programming (PyOpenCL) to accelerate the computation.

.. toctree::
   :maxdepth: 1
   :caption: Getting started

   sub_doc/example

.. toctree::
   :maxdepth: 1
   :caption: Forward modeling

   sub_doc/forward_modeling


.. toctree::
   :maxdepth: 1
   :caption: Inversion

   sub_doc/inversion


.. toctree::
   :maxdepth: 1
   :caption: Tools

   sub_doc/tools

.. toctree::
   :maxdepth: 1
   :caption: Rock physics

   sub_doc/rock_physics


Citing PyFWI
============

::

   @software{PyFWI,
   author       = {Mardan Amir and
                  Bernard Giroux and
                  Gabriel Fabien-Ouellet},
   title        = {{PyFWI}: {Python} Python package for Full-Waveform Inversion (FWI)},
   month        = Jan,
   year         = 2022,
   publisher    = {Zenodo},
   doi          = {10.5281/zenodo.5813637},
   url          = {https://doi.org/10.5281/zenodo.5813637}
   }
