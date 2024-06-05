.. _how_to_run_mercury:

How to run Mercury
==================

There are three entry points to the model, all in the root folder:

- ``mercury.py``: CLI interface. Useful for running the simulator on a cluster for instance. See :ref:`cli`.
- ``Mercury.ipynb``: Jupyter notebook. Useful to play with the Mercury object, importable in other scripts. See :ref:`notebook`.
- ``mercury_gui.py``: Dash interface. Useful to explore the input data and prepare new datasets. See :ref:`gui`.

To run the simulator all of them use the same underlying engine. The behaviour of the engin can be driven via parameter
files and input data. The main configuration file for Mercury can be found in ``config/mercury_config.toml``. This file
contains parameters related to **how** Mercury runs, for instance the location of the input data, their format,
parallelisation, etc. By default, the three interfaces will read this file for the parameters. You can also use another
parameter file, for instance passing to the CLI interface the option ``-psi path/to/my_custom_profile_file.toml``. More
details can be found in :ref:`cli`. All parameters are listed here: :ref:`config_parameter_file`.

The config parameter includes in particular the path to the data, by default ``../input``. The input folder needs to be
organised as explained in :ref:`data_input` to be readable by Mercury. In particular, it needs to include a file called
"scenario_config.py", which is compiling all the necessary information to run this particular scenario. It is organised
in two parts:

- the path too all tables needed to run the scenario, all of them included in the data folder.
- the parameters needed to set the simulation up. These parameters are linked to the agent behaviours (e.g.price of fuel).

All scenario parameters are described here: :ref:`scenario_parameter_file`.

The mercury config also includes the path to where the results will be saved, but default in ``../results``. The structure
of the results is discussed and explained in :ref:`data_output`.

All parameters both from the mercury_config.toml file and the scenario parameter file can be set at runtime, in the cli
version, with the Mercury object, or in the GUI (e.g. using the argument ``--airlines__fuel_price 1.2`` in the CLI).

Finally, the three entry points (CLI, Mercury object, GUI) are available as Docker images, which requires only the
installation of the docker third-party app. More information can be found here: :ref:`docker`.









