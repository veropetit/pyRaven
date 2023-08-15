.. Raven documentation master file, created by
   sphinx-quickstart on Mon Jun  7 13:58:56 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Documentation for pyRaven
========================================

This is a python implementation of the Bayesian method for determining 
magnetic dipolar field upper limits from a set of Stokes V spectropolarimetric
observations from a given star of Petit & Wade 2012 (MNRAS 420 773). 

.. toctree::
   :hidden:
   :maxdepth: 1

   Installation

.. raw:: html

   <a class="button" href="Installation.html">Installation</a>

|

How to run pyRaven tutorial
----------------------------

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: How to run pyRaven tutorial

   nblink/01-DataSetup
   nblink/02-FitIntensity
   nblink/03-ChiSquareCalcLoop
   nblink/04-CalculateProbabilities

If you are interested in the simple application of pyRaven, please follow the tutorials below. 

.. raw:: html
   
   <a class="button" href="01-DataSetup.html">01 - Data Setup</a>
   <a class="button" href="02-FitIntensity.html">02- Fit of the intensity profiles for single stars</a>
   <a class="button" href="03-ChiSquareCalcLoop.html">03- Calculating the chi2 with the loop code</a>
   <a class="button" href="04-CalculateProbabilities.html">04 - Calculate probabilites from the chi2 files</a>

|

Standalone tutorials
----------------------------

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Standalone tutorials

   nblink/ParamsObject_tutorial
   nblink/BayesObjects_user

.. raw:: html
   
   <a class="button" href="ParamsObject_tutorial.html">Tutorial on how to use the Params Dictionary objects</a>
   <a class="button" href="BayesObjects_user.html">User’s manual for the BayesObjects.py classes</a>

|

.. toctree::
   :maxdepth: 1
   :caption: Underlying physics and statistics

   nblink/diskint2_doc
   nblink/geometry_doc
   nblink/bayes_doc


.. toctree::
   :caption: API documentation
   :maxdepth: 2

   API






Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
