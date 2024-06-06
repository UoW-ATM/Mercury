.. _data_input:

Data Input
==========

Note: a sample of data is provided with Mercury and can be downloaded
`here <https://zenodo.org/records/11384379/files/Mercury_data_sample.zip?download=1>`_.

Running Mercury requires to input the data in the right format. The input data is organised in scenarios and case
studies. Scenarios can be seen typically as a set of schedules,
passenger itineraries, as well as the definition of other agents like the AMAN. A case study is usually represented by a
subset of flights and/or different operational configuration. When running Mercury, one has to specify as least the id
of the scenario, and optionally the id of a case study. If no case study is chosen, Mercury will run the case "0",
coinciding exactly with the data and parameters defined by the scenario itself.

The scenarios are read from the input folder, defined in the ``mercury_config.toml`` file, by default ``../input``. The
input folder should follow the following structure:

- ``input``

  - ``scenario=-1``

    - ``scenario_config.toml``
    - ``data`` -> with all base data
    - ``case_studies`` -> with all data replacing the base data.

      - ``case_study=0``

        - ``case_study_config.toml``
        - ``data``
  - ``scenario=0``

    - etc...

The first important file in each scenario is the ``scenario_config.toml`` file. This file is organised in two sections.
The first one, under the header ``data``, is a list of all the tables needed for the simulation, and what are their
specific names in this scenario folder, for instance:

.. code:: toml

    [data.delay]
    input_delay_paras = 'delay_parameters'

    [data.network_manager]
    input_atfm_delay = 'iedf_atfm_static'
    input_atfm_prob = 'prob_atfm_static'

The second part of this file is composed of parameters and their values, for instance:

.. code:: toml

    [paras.modules]
    modules_to_load = ['CM']
    path = 'modules'

    [paras.airlines]
    non_ATFM_delay_loc =  0.0
    compensation_uptake = 0.11
    delay_estimation_lag = 60

Like in the first part, the parameters are organised in different sections and subsections, here for instance "modules"
and "airlines". This is important when using the CLI or the programmatic interface, because the parameters have to be called
based on their section name, for instance ``airlines__non_ATFM_delay_loc``, i.e. the name of the section, two underscores
``__``, then the name of the parameter. A full description of the parameters can be found here:
:ref:`scenario_parameter_file`.

The tables listed in the first part have to be included in the scenario folder, in ``data``, following the same structure
as the toml file. All tables must be in parquet format. A detailed description of the all the input tables can be
found here: :ref:`input_tables`.

When defining a case study for a scenario, one has to follow a similar approach. First, all case studies for a scenario
should be included in the ``case_studies`` folder inside the scenario directory. Then one has to put a case study
configuration file called ``case_study_config.toml``1, similar to the scenario's. However, one only needs to put the modification with respect to the
scenario in this new folder. For instance, this config file:

.. code:: toml

    [info]
        case_study_id = -1
        description = "Only flights departing from ED"

    [data]
        [data.schedules]
            input_subset = "flight_subset"

    [paras]
        [paras.airlines]
        fuel_price = 0.1

        [paras.modules]
        modules_to_load = ["nostromo_EAMAN"]

    [agents_configuration]

All the four main sections (``info``, ``data``, ``paras``, ``agents_configurations``) have to appear in the files but
some can be empty. Only parameters that are different need to appear, in this case ``fuel_price`` and ``modules_to_load``.
Moreover, one can replace or modify some of the data included in the scenario by including new tables in the ``data``
folder inside the case study one, and mention it in the toml file (here the ``flight_subset``).

Finally, note that the GUI version of Mercury provides an easy way of exploring the different types of data, modifying
them, creating scenarios and case studies, etc. More information here: :ref:`gui`.











