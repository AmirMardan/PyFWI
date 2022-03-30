Regularization 
==============

.. autoclass:: PyFWI.fwi_tools.Regularization
    :exclude-members:

Total variation (TV) 
--------------------

.. automethod:: PyFWI.fwi_tools.Regularization.tv


Tikhonov  
---------

.. automethod:: PyFWI.fwi_tools.Regularization.tikhonov


Parameter relation  
-------------------

.. automethod:: PyFWI.fwi_tools.Regularization.parameter_relation   
       
Gradient switching
===================

lmd to vd
---------------------------------

.. automethod:: PyFWI.grad_swithcher.grad_lmd_to_vd


vd to lmd
---------------------------------

.. automethod:: PyFWI.grad_swithcher.grad_vd_to_lmd

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

