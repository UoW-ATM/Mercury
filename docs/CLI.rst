.. _cli:

Command Line Interface
======================

The CLI version of Mercury is usable in any shell, e.g. bash in linux or powershell in Windows. Its basic syntax is the
following:

.. code:: bash

    ./mercury.py -id -1

This will for instance run one iteration of the basic scenario "-1", reading the default `config/mercury_config.toml`
config file.

The CLI can be used in a very agile way. Indeed, all parameters from the mercury config are accessible through the
interface (the list of parameter is built at runt-time, so even new custom parameters will be accessible. All parameters
are listed in :ref:`simulation_parameters` but some have shortcuts and the most important ones are explained here:

- "-id": the id of the scenario to run. If "-id X" is run, there must be a "scenario=X" folder in the input folder.

