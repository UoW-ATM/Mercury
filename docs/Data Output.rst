.. _data_output:

Data Output
===========

Mercury is an agent-based model with a high number of agents, therefore producing potentially high volume of data
on each agents.

The main output of Mercury is a set of tables, stored in the output directory defined by the variable
``write_profile.path`` in the mercury config file (by default ``../results``).
These tables (saved as ``.csv.gz`` files) are stored in a folder whose name is structured as
``[model version]_[scenario id]_[case study id]_[iteration number]``. One can also insert a timestamp in folder name
by setting the ``outputs_handling.insert_time_stamp`` parameter to True in the mercury config file.

The two most important tables saved by Mercury are the following:

- output_flights: compiles important information the flights, including fuel consumption, delay, etc.
- output_pax: gathers information related to passengers, in particular their final delay, whether they missed
  connections etc.

Other types of data are also saved at the save time, including:

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


