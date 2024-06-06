.. _overview:

Overview
========

What is Mercury?
----------------

Mercury is an open source mobility simulator designed by and for researchers. It's been built around issues related to
air transportation and it's:

- agent-based driven,
- stochastic,
- event-driven,
- microscopic, down to single flight and passenger,
- encompassing the whole European Airspace and beyond (ECAC - European Civil Aviation Conference),
- modular,
- able to communicate with other models, optimisers, and even HMIs.

Ok, but what does it do? In very short:

- it tracks aircraft across Europe across their trajectory, but also during their time at the airport,
- it tracks passengers for their entire journey, gate-to-gate including connections and possible hurdles (cancelled flights etc),
- it uses a network manager and internal complex airline rules and optimisation process to make millions of decision each day of operation,
- it measures every aspect of the system, from passenger real delay to flight emissions and arrival manager (AMAN) optimisation processes.

Mercury is particularly well suited to address problems related to:

- systemic effects in the air transportation system, for instance how delays propagate,
- impact study on changing policies, prices, and processes, in particular related to pre-tactical and strategic ones
- passenger centric-issues, like modifications to passenger compensation.

If you want to have a quick start, go here: :ref:`quickstart`. If you want more details about how to run Mercury,
like the options and the input/output data specs, go here: :ref:`basic_usage`. If you need more details
about advanced usage of Mercury like Module creation, go here: :ref:`advanced_usage`. If you need a detailed description
of all input and output tables, as well as the list of parameters of the model, go here: :ref:`io_specs`.
If you want to know more about the underlying model, go here: :ref:`model_presentation`.
If you want more details about the code itself, go here: :ref:`api`.


Quick FAQ
---------

**Can I use it for my own research?**

Yes, please do! Just cite it as:
XXXX TODO

**Can I use it for a commercial application?**

No! The GNU licence forbids the commercial application use. So far, the UoW is the main copyright holder and we
could issue a double licence in theory, please contact us if you are interested.

**Is it reliable?**
It has been used quite extensively in different research projects and this is never a guarantee. Please consider it as any other
research tool, prone to errors and requiring frequent external visualisations.

**What's with the different versions, CLI, NB, GUI? Which one should I use?**

For a first contact, we suggest we use the Graphical User Interface (GUI) version, which shows input data in a user friendly way.

The NB (Jupyter NoteBook) version is good for quick and easy use, and to show the programmatic interface, if you need to
wrap Mercury in another piece of code.

The CLI (command line interface) is destined to provide an easy interface for for instance for cluster computation. Most
of the programmatic capabilities are included in the CLI version.

**What are the Docker packages? Should I use that?**

Docker is a widely used software that uses virtualisation to run super light versions of OSs on other OS for specific
application. You do not need to use Docker, you can just install and run the Mercury application natively, but Docker
provides you with a way of doing it in just a few lines of code.

**What is the difference between this model and other models like R-NEST or Blue Sky?**

R-NEST is a model developed by EUROCONTROL, focused on air traffic management and airspace usage, while Mercury
is more focused on systemic interactions beetwen airports, air traffic flow management (ATFM), passengers, and flights. Moreover, R-NEST is
closed and undocumented, whereas Mercury is open.

Blue Sky is another simulator developed by TU Delft, it is open. It is however focused on aircraft performance, ATC
control airspace dynamics, etc. On the contrary, Mercury is relatively crude when it comes to the trajectory management,
even though it is using OpenAP (like Blue Sky) and can be used with BADA3 and BADA4 for accurate fuel burn estimates.




