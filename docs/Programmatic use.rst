.. _notebook:

Programmatic use
================


The ``Mercury.ipynb`` notebook included in the repository shows examples of how use Mercury programmatically, using the
Mercury object.

The Mercury object is designed to provide an entry to Mercury to users that would like to include Mercury in their own
scripts. It has almost the same capabilities than the CLI interface, with added flexibility for output management.

By default, the object can initialised without any parameters:

.. code:: python

    mercury = Mercury()

It can then be used to run simulation, by passing various arguments. For instance:

.. code:: python

    scenarios = [-1]
    case_studies = [0, -1]
    results, results_seq = mercury.run(scenarios=scenarios,
                                          case_studies=case_studies)

will run case study 0 and and -1 from scenario -1.

Just like the CLI, one can fix or iterate over any argument included in the Mercury config file or the scenario/case
study config file, with a slightly different interface than the CLI:

.. code:: python

    ps = read_mercury_config(config_file='config/mercury_config.toml')
    ps['computation__num_iter'] = 2 # number of computations per case study

    # Choose scenario and case study to be simulated
    scenarios = [-1]
    case_studies = [-1]

    ## Choose some values to be simulated
    # The first dictionary will set values to a specific value for all iterations. If you want to sweep
    # some parameters, see next dictionary
    paras_sc_fixed = {
                    'modules__modules_to_load':[]
                    }

    # The second dictionary will iterate through different values for each parameters. Each value of each parameter
    # is simulated against all values of the other parameters.
    paras_sc_iterated = {'airports__sig_ct':[15., 17.],
                           }
    results, results_seq = mercury.run(scenarios=scenarios,
                                       case_studies=case_studies,
                                        paras_sc_iterated=paras_sc_iterated,
                                        paras_sc_fixed=paras_sc_fixed,
                                        paras_simulation=ps
                                      )

In this case we need to load the mercury config file independently beforehand, then modify a parameter inside. There is
however no need to load the scenario config file, and parameter can be fixed or iterated through the use of two
dictionaries, respectively ``paras_sc_fixed`` and ``paras_sc_iterated``. Note that the syntax for the parameters are the same than for the CLI, i.e.
``[section of config file]__[name of parameter]``.

The run method returns two dataframes containing the results from the runs. The first one corresponds to the one saved
by the CLI as aggregated results. The second one contains other advanced results. Note that by default, Mercury also saves
the full detailed results (for each iteration), similarly to the CLI, by default in
``../results/[model version]_[scenario id]_[case study id]_[iteration number]``.

Note that contrary to the CLI version, in general string parameters can be iterated with the object interface.

The Mercury object can also use two important features to help the user, the parametriser and the aggregator. Both
are described in detail in the advanced usage section, here: :ref:`parametriser_aggregator`





