.. _data_output:

Data Output
===========

Mercury is an agent-based model with a high number of agents, therefore producing high volumes of data
on each agents.

Detailed output
---------------

The main output of Mercury is a set of tables, stored in the output directory defined by the variable
``write_profile.path`` in the mercury config file (by default ``../results``).
These tables (saved as ``.csv.gz`` files) are stored in a folder whose name is structured as
``[model version]_[scenario id]_[case study id]_[iteration number]``. One can also insert a timestamp in the folder name
by setting the ``outputs_handling.insert_time_stamp`` parameter to ``True`` in the mercury config file.

The two most important tables saved by Mercury are the following:

- output_flights: compiles important information the flights, including fuel consumption, delay, etc.
- output_pax: gathers information related to passengers, in particular their final delay, whether they missed
  connections etc.

Other types of data are also saved at the same time, including:

- output_dci: information related to Dynamic Cost Indexing when used.
- output_eaman: information related to the optimisation process followed by the EAMAN.
- output_events: information related to various events in the simulation, mostly used for benchmarking.
- output_general_simulation: information related to the simulation itself.
- output_hotspot: information related to the resolution of hotspots in the airspace.
- output_messages: information related to internal messages during the simulation, mostly used for benchmarking.
- output_RNG: information related to the Random Number Generator (probably not working at the moment).
- output_wfp: information related to the "Wait For Passenger" process, whereby airlines may wait for late connecting
  passengers

Finally, a copy of all the parameters used for this simulation is saved as a pickled dictionary ``paras.pic``, and
a script ``unzip_results.py`` is included to easily unzip all the files.

The description of all fields appearing in the output tables can be found here: :ref:`output_tables`.

Aggregated output
-----------------

On top of the detailed output, Mercury also produces by default a summary of the results, encompassing all
iterations. This summary is saved to the result folder and is name by the ``outputs_handling.file_aggregated_results``
parameter, by default ``results.csv`` (note: unless the user manually changes this name in the config file, this file is
thus overwritten everytime Mercury is run).

These aggregated results represent a subset of the metrics gathered in the full output, computed in average (+std) for
each iteration over all flights or passengers. The table is structured so that it features the scenario id, the case
study id, the parameters that have been swept by the user. An example can be seen in the figure below:

.. image:: docs/images/example_aggregated_output.png
  :width: 600
  :alt: Example of aggregated output

This aggregated output is the same that is produced by the ``run`` method of the Mercury object (see :ref:`notebook`).
It is designed to be useful for quick and easy tests, but can also be customised, see :ref:`parametriser_aggregator`.




