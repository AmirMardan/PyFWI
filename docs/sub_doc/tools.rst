Regularization 
==============

.. autoclass:: PyFWI.fwi_tools.Regularization
    :exclude-members:

Total variation (TV) 
--------------------

.. automethod:: PyFWI.fwi_tools.Regularization.tv
    :noindex:

Tikhonov  
---------

.. automethod:: PyFWI.fwi_tools.Regularization.tikhonov
    :noindex:

Parameter relation  
-------------------

.. automethod:: PyFWI.fwi_tools.Regularization.parameter_relation   
   :noindex:

Prior information  
-------------------

.. automethod:: PyFWI.fwi_tools.Regularization.priori_regularization
    :noindex:

Gradient switching
===================

lmd to vd
---------------------------------

.. automethod:: PyFWI.grad_switcher.grad_lmd_to_vd
    :noindex:

vd to lmd
---------------------------------

.. automethod:: PyFWI.grad_switcher.grad_vd_to_lmd
    :noindex:

Cost Function
=============
.. autoclass:: PyFWI.fwi_tools.CostFunction
    :members: __call__


Visualization
=============

Earth model
-----------
.. automethod:: PyFWI.seiplot.earth_model
    :noindex:

Seismic Section
---------------
.. automethod:: PyFWI.seiplot.seismic_section
    :noindex:
