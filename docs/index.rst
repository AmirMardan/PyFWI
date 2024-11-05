.. PyFWI documentation master file, created by
   sphinx-quickstart on Sat Jan 29 00:58:32 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to PyFWI's documentation!
=================================

**PyFWI** is an open source Python package to perform seismic modeling and full-waveform inversion (FWI) in elastic media. 
This package is implemented in time domain and coded using GPU programming (PyOpenCL) to accelerate the computation.

If you have any questions about PyFWI, please use the following this `link <https://github.com/AmirMardan/PyFWI/discussions>`_.

If you have any technical questions about FWI, TL-FWI, or PyFWI, please visit `my personal website <https://amirmardan.github.io/tlfwi.html/>`_.
All my publications are available there. I will be happy to assist if you contact me via email.

For bugs, developments, and errors, please use issues in the GitHub repository available `here <https://github.com/AmirMardan/PyFWI/issues>`_ to ask your questions.

.. toctree::
   :maxdepth: 1
   :caption: Getting started

   sub_doc/example
   sub_doc/fwi_example

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

.. toctree::
   :maxdepth: 1
   :caption: Deep learning

   sub_doc/grad_pytorch

Citing PyFWI
============

::

   @article{mardan2023pyfwi,
  title = {PyFWI: {A Python} package for full-waveform inversion and reservoir monitoring},
  author = {Mardan, Amir and Giroux, Bernard and Fabien-Ouellet, Gabriel},
  journal = {SoftwareX},
  volume = {22},
  pages = {101384},
  year = {2023},
  publisher = {Elsevier},
  doi = {10.1016/j.softx.2023.101384}
   }
