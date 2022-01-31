Regularization 
==============

.. autoclass:: PyFWI.fwi_tools.regularization
    :exclude-members:

Total variation (TV) 
--------------------

.. automethod:: PyFWI.fwi_tools.regularization.tv


Tikhonov  
---------

.. automethod:: PyFWI.fwi_tools.regularization.tikhonov
    
       
Gradient switching
===================

lmd to vd
---------------------------------

.. automethod:: PyFWI.fwi_tools.grad_lmd_to_vd


vd to lmd
---------------------------------

.. automethod:: PyFWI.fwi_tools.grad_vd_to_lmd

Cost Function
=============
.. autoclass:: PyFWI.fwi_tools.CostFunction
    :members: __call__


Visualization
=============

Earth model
-----------
.. automethod:: PyFWI.seiplot.earth_model


Seismic Section
---------------
.. automethod:: PyFWI.seiplot.seismic_section

