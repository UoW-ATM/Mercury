.. _code_organisation:

Organisation of the code
========================

The code is organised in different folders:

root
^^^^

The root folder contains the three important entry points to the model:

- mercury.py: CLI interface to the model. Useful for running the simulator on a cluster for instance. See :ref:`cli`.
- Mercury.ipynb: Jupyter notebook, useful to play with the Mercury object, importable in other scripts. See :ref:`notebook`.
- mercury_gui.py: Dash interface. Useful to explore the input data and prepare new datasets.

agents
^^^^^^

This folder gathers all the definition of the agents.

TBC
