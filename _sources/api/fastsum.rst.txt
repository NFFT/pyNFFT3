``pyNFFT3.fastsum`` - fastsum Class
===================================================

fastsum Methods
-----------------

.. autoclass:: pyNFFT3.FASTSUM
   :members: 
   :undoc-members:
   :member-order: bysource
   :exclude-members: x, y, alpha, __init__

fastsum Attributes
--------------------

   .. attribute:: plan <fastsum_plan>

      fastsum plan (C pointer)

   .. attribute:: N <int>

      The number of source nodes. Must be a positive integer.

   .. attribute:: M <int>

      The number of target nodes. Must be a positive integer.

   .. attribute:: n <int>

      The expansion degree. Must be a positive integer.

   .. attribute:: p <int>

      The degree of smoothness. Must be a positive integer.

   .. attribute:: kernel <string>

      Name of kernel function *K*.

   .. attribute:: c <numpy.ndarray>

      Kernel parameters; length depends on value of **kernel**.
    
   .. attribute:: eps_I <float>

      The inner boundary.

   .. attribute:: eps_B <float>

      The outer boundary. Value must be ∈ (0.0, 0.5)

   .. attribute:: nn_x <int>

      The oversampled **nn** in **x**. 

   .. attribute:: nn_y <int>

      The oversampled **nn** in **y**. 
    
   .. attribute:: m_x <int>

      The NFFT-cutoff in **x**.

   .. attribute:: m_y <int>

      The NFFT-cutoff in **y**. 

   .. attribute:: x <numpy.ndarray>

      Float array for source nodes.

   .. attribute:: y <numpy.ndarray>

      Float array for target nodes.

   .. attribute:: alpha <numpy.ndarray>

      Complex array for source coefficients.

   .. attribute:: f <numpy.ndarray>

      Complex array for targer evalutations.

   .. attribute:: d <int>

      The number of dimensions.

   .. attribute:: init_done <boolean>

      Boolean to indicate if the fastsum plan is initialized.

   .. attribute:: finalized <boolean>

      Boolean to indicate if the fastsum plan is finalized.