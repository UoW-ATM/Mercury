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
- microscopic, down to single flights and passengers,
- encompassing the whole European Airspace and beyond (ECAC),
- modular,
- able to communicate with other models, optimisers, and even HMIs.

Ok, but what does it do? In very short:

- It tracks flights across Europe during their trajectory, but also during their time at the airport,
- it tracks passengers for their entire journey, gate-to-gate including connections and possible hurdles (cancelled flights etc),
- it uses a network manager and internal complex airline rules and optimisation process to make millions of decision each deay of operation,
- it measures every aspect of the system, from passenger real delay to flight emissions and AMAN optimisation processes.

Mercury is particularly well suited to answer to problem related to:

- systemic effects in the air transportation system, for instance how delays propagate,
- impact study on changing policies, prices, and processes, in particular related to pre-tactical and strategic ones
- passenger centric-issues, like modifications to passenger compensation.

If you want to have a quick start, go there: :ref:`quickstart`. If you want have more details about how to run Mercury,
like how the options to run it and the input/output data specs, go there: :ref:`basic_usage`. If you need more details
about advanced usage of Mercury like Module creation, go there: :ref:`advanced_usage`. If you want to have a look at the
underlying model, go there: :ref:`model_presentation`. If you want to access the API, go there: :ref:`api`.

Quick FAQ
---------

**Can I use it for my own research?**

Yes, please do! Just cite it as:
XXXX TODO

**Can I use it for a commercial application?**

No! The GNU licence forbids to use it for commercial application. But so fat UoW is the main copyright holder and we
could issue a double licence in theory, please contact us if you are interested.

**Is is reliable?**
It has been used quite extensively in different research projects and this is never a guarantee. Please consider it as any other
research tool, prone to errors and requiring frequent external visualisations.

**What's with the different versions, CLI, NB, GUI? Which one should I use?**

For a first contact, we suggest we use the Graphical User Interface (GUI) version, which shows input data in a user friendly way.

The NB (Jupyter NoteBook) version is good for quick and easy use, and to show the programmatic interface, if you need to
wrap Mercury in another piece of code.

The CLI (command line interface) is destined to provide an easy interface for for instance for cluster computation. Most
of the programmatic capabilities are included in the CLI version.

**What are the Docker packages? Should I use that?**

Docker is a wildly used software that uses virtualisation to run super light versions of OSs on other OS for specific
application. You do not need to use Docker, you can just install and run the Mercury application natively, but Docker
provides you with a way of doing it is just a few lines of code.

**What is the difference between this models and other models like RNEST or Blue Sky?**

RNEST is a model developed by EUROCONTROL mainly focused on air traffic management and airspace usage, while Mercury
is more focused on systemic interactions beetwen airports, ATFM, passenger, and flights. Moreover, RNEST is completely
closed and undocumented, whereas Mercury is open.

Blue Sky is another simulator developed by TU Delft, it is open. It is however for focused on aircraft performance, ATC
control airspace dynamics etc. On the contrary, Mercury is relatively crude when it comes to the trajectory management,
even though it using OpenAP (like Blue Sky) and can be used with BADA3 and BADA4 for accurate fuel burn estimations.




