.. _cli:

Command Line Interface
======================

The CLI version of Mercury is usable in any shell, e.g. bash in linux or powershell in Windows. Its basic syntax is the
following:

.. code:: bash

    ./mercury.py -id -1

This will for instance run one iteration of scenario "-1", reading the default ``config/mercury_config.toml``
config file. Another path to the config file can be set using the "-psi" option, see below.

The CLI can be used in a very agile way. Indeed, all parameters from the mercury config are accessible through the
interface. All parameters
are listed in :ref:`simulation_parameters` but some have shortcuts and the most important ones are explained here:

- "-id": the id of the scenario to run. If "-id X" is run, there must be a "scenario=X" folder in the input folder.
  Compulsory.
- "-psi": path to the toml mercury config file. Default is ``config/mercury_config.toml``.
- "-cs": the id of the case study to run. If "-id X -cs Y" is run, there must be a "scenario=X/case_studies/case_study=Y"
  folder in the input directory.
- "-n": number of iterations to run a given set of parameters.
- "-fl": once the input data is stable, this option can be used to speed up the data loading.
- "-pc": number of core to use in parallel for multiple iterations (a single run of Mercury is always single-core and
  single thread)

On top of that, one can use the cli to set and iterate over parameters . Iterable parameters
include scenario ids, case study ids. For instance, this command performs one iteration on scenario -1
and one iteration on scenarios -2:

.. code:: bash

    ./mercury.py -id -1 -2

Any parameter defined in the scenario (or case_study) parameter file can also be fixed or iterated through the CLI
(the list of parameter is built at runtime, so even new custom parameters and parameters coming from modules will be
available). For instance, if the price of fuel is defined as:

.. code:: toml

    [paras]
        [paras.airlines]
        fuel_price = 0.1

in the scenario config file, then one can iterate over it like this:

.. code:: bash

    ./mercury.py -id -1 -airlines__fuel_price 0.3 0.5 0.7

Note that by specifying iterations over several parameters, one ends up always with the full combination of parameter
values (the tensorial product), for instance:

.. code:: bash

    ./mercury.py -id -1 -2 -airlines__fuel_price 0.3 0.5 0.7 -n 2

will perform 2 iterations on scenario -1 with a fuel price of 0.3, 2 iterations of scenario -1 with a fuel price
of 0.5, etc.

By default, the CLI will save two kinds of results:

- some aggregated results, by default saved in ``../results/results.csv``
- some detailed results on each flight, passenger, airport, etc, by default in the
  ``../results/[model version]_[scenario id]_[case study id]_[iteration number]`` folder

More details on the output data can be found in :ref:`data_output`.

Note: iteration over string parameters will not work through the CLI.


