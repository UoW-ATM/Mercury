.. figure:: mercury_logo_small.png
   :alt: mercury_logo_small.png
\
\
|ImageLinkBadgeDocs|
|ImageLinkBadgeDocker|

.. |ImageLinkBadgeDocs| image:: https://github.com/UoW-ATM/Mercury/actions/workflows/docs.yml/badge.svg
.. _ImageLinkBadgeDocs: https://github.com/UoW-ATM/Mercury/actions/workflows/docs.yml/

.. |ImageLinkBadgeDocker| image:: https://github.com/UoW-ATM/Mercury/actions/workflows/docker.yml/badge.svg
.. _ImageLinkBadgeDocker: https://github.com/UoW-ATM/Mercury/actions/workflows/docker.yml/

Mercury is a research-oriented air transportation mobility simulator
with a strong agent-based paradigm.

.. inclusion-marker-do-not-remove

Table of Contents
=================

- `Quickstart <#quickstart>`_
   - `Docker <#docker>`_
   - `Installation <#installation>`_
   - `Running the CLI version <#cli>`_
   - `Programmatic use of Mercury <#programmatic>`_
   - `Graphical interface <#gui>`_
- `Manual and references <#manual>`_
- `Software Architecture <#soa>`_
- `About <#about>`_
   - `Authorship <#authors>`_
   - `Licence and copyright <#licence>`_


Quickstart
==========
.. _quickstart:

Docker
------
.. _docker:

NEW! Docker versions are now available for:

- the command line interface (CLI) version: `mercury_cli <https://github.com/orgs/UoW-ATM/packages/container/package/mercury%2Fmercury_cli>`_;
- the notebook version: `mercury_nb <https://github.com/orgs/UoW-ATM/packages/container/package/mercury%2Fmercury_nb>`_;
- the GUI version: (coming soon).

Docker allows you to use the model on any OS without installing anything except a docker environment (see https://www.docker.com/get-started/).
With a terminal (e.g. PowerShell in Windows), you can download the docker images like this:

.. code:: bash

    docker pull ghcr.io/uow-atm/mercury/mercury_nb:latest
    docker tag ghcr.io/uow-atm/mercury/mercury_nb:latest mercury_nb
    docker pull ghcr.io/uow-atm/mercury/mercury_cli:latest
    docker tag ghcr.io/uow-atm/mercury/mercury_cli:latest mercury_cli

In a terminal, you can then use the docker image like this:

- for the CLI:

.. code:: bash

    docker run mercury_cli -id -1 -cs -1

- for the notebook:

.. code:: bash

    docker run -p 8888:8888 mercury_nb

You probably need to copy the URL appearing in the terminal after this command and copy/paste it into your browser.

- for the GUI: (coming soon)



Installation
------------
.. _installation:

Mercury has been tested on Ubuntu-like machines and, to a lesser extent, Windows, using minicoonda/anaconda. Python 3.10 is
recommended, Python 3.12 will raise issues.

Quick install
^^^^^^^^^^^^^

A bash script is available for quick installations in Linux. You can download it from
`here https://github.com/UoW-ATM/Mercury/blob/master/mercury_quick_install_stable.sh`_ for the stable version (master
branch) and `here https://github.com/UoW-ATM/Mercury/blob/dev/mercury_quick_install_dev.sh`_ for the dev version. You
need to download them first and then run them in a terminal like this:

.. code:: bash

    ./mercury_quick_install_dev.sh

This might or might not work depending on your specific environment, particularly your virtual environment setting.
If it fails, you can follow the steps below.

Full install
^^^^^^^^^^^^

-  Start by cloning the repository, for instance:

.. code:: bash

    git clone https://github.com/UoW-ATM/Mercury

-  Use this to download the third-party libraries:

.. code:: bash

   cd Mercury
   git submodule update --recursive --remote --init

-  In a fresh Python environment, install all the required packages:

In Linux, use:

.. code:: bash

   sudo apt-get install libproj-dev libgeos-dev build-essential python3-dev proj-data proj-bin
   python -m pip install shapely cartopy --no-binary shapely --no-binary cartopy
   pip install -r requirements.txt

In Windows, you need to install the requirements in the dedicated environment:

.. code-block:: bash

    pip install -r requirements.txt

You may also need to install Visual Studio C++-built tools if that's not the case already.

-  Download the sample data here:
   https://zenodo.org/records/11384379/files/Mercury_data_sample.zip?download=1. Extract the data. Put the results "input" folder outside the main Mercury root folder (side by side). You can also use the following commands from inside the Mercury root folder to achieve the same result:

.. code:: bash

    wget https://zenodo.org/records/11384379/files/Mercury_data_sample.zip?download=1 -O ../mercury_public_dataset.zip
    unzip ../mercury_public_dataset.zip -d ../input/
    rm ../mercury_public_dataset.zip


By default, Mercury uses the `OpenAP <https://github.com/TUDelft-CNS-ATM/openap>`_ model for aircraft performance.
However, Mercury also supports the BADA models developed by EUROCONTROL. If you want to use it, you can request a licence
from EUROCONTROL (here: https://www.eurocontrol.int/model/bada), then use the script ``generate_bada3_input.py`` to
transform the AFP, OFP and PTD files from BADA3 into tables (parquet files) that Mercury will read.
In the following command, replace ``BADA3_FILES_PATH`` with the location of the downloaded bada files:

.. code:: bash

   python generate_bada3_input.py -s BADA3_FILES_PATH -d .

Ensure you copy the generated parquet files into ``Mercury/libs/performance_models/bada3/data/``.

If you want to use BADA4, please contact us directly, and we'll offer general guidance. We are also working on a support
for EUROCONTROL's pyBADA library.

Running the CLI version
-----------------------
.. _cli:

You can test the model by running:

.. code:: bash

   ./mercury.py -id -1 -cs -1

Use ``-h`` to have list of all the possible arguments.

Programmatic use of Mercury
---------------------------
.. _programmatic:

Mercury can be used as an object. An example of its use and some
examples to run can be found in the ``Mercury.ipynb`` Jupyter notebook.
The notebook shows the possible uses of Mercury in terms of parameter
setting, scenarios, case study, etc.

Graphical interface
-------------------
.. _gui:

You can use a GUI to explore the data input and output structure, create
new scenarios, case studies, etc. Use the following command to start it:

.. code:: bash

   python mercury_gui.py

.. inclusion-marker-do-not-remove2

Manual and references
=====================
.. _manual:

A more complete manual is in construction and can be found here_.

.. _here: https://uow-atm.github.io/Mercury

The following articles can also be consulted to learn more about
Mercury:

-  the one included in `the repo <https://github.com/UoW-ATM/Mercury/blob/master/docs/SIDs_2023_OpenMercury.pdf>`_, presenting the general approach to Mercury,
-  the one available there_ showing some details of Mercury and some examples of its uses, from a few years back.

.. _there: https://www.sciencedirect.com/science/article/abs/pii/S0968090X21003600

.. inclusion-marker-do-not-remove3

Software Architecture
=====================
.. _soa:


Mercury is organised in three packages:

1.	agents: This is the main package containing the implementation of different agents in Mercury. The agents are developed following an object-oriented approach. Each agent type is a Class containing its memory (attributes) and Roles. The Roles are independent Classes contained within the Agents. All agent types inherit from a generic Agent class, which provides the shared functionalities of initialisation, mailbox and functionalities required to modify their behaviour through the application of Modules. Two sub-packages are located inside the agents' package:

* Modules: 
   This package stores different modules that can be loaded into Mercury. A Module is composed of three files:

   * the Python code implementing the functionalities that need to be added and/or replaced in the different Roles,
   * a configuration file indicating which functions need to be added/replaced for which roles, and
   * an optional configuration file with any additional parameters needed for the new functionalities implemented in the module.

* Commodities:
   Contains different objects used and manipulated by the agents, such as the definition of aircraft, alliance, slots, etc. Each one of these concepts will be represented by one or several classes.

2.	libs: The libs package contains functionalities required by Mercury, such as the implementation of the Delivery system, World builder (to create the agents at the instantiation of a simulation), Simulation manager (to manage the execution of Mercury), Case study loader, etc. Functionalities to manage the input and output of Mercury are also provided here (Input and Output managers). Finally, external libraries are also included here.

3.	config: The config package contains the configuration files of Mercury and the simulations.


About
=====
.. _about:

Authorship
----------
.. _authors:

Up to the open-source release, all Mercury code has been written by
Gérald Gurtner and Luis Delgado, to the exception of:

-  The Dynamic Cost Indexing module, written by Damir Valput
-  The GUI, written by Michal Weiszer

We also thank Tanja Bolic for many waves of testing.

Licence and copyright
---------------------
.. _licence:

Mercury is released under the GPL v3 licence. The licence can be found
in LICENCE.TXT

Mercury uses the Hotspot library
(https://github.com/andygaspar/Hotspot) and the uow-belt-tools library (https://github.com/UoW-ATM/uow_tool_belt), both
released under GPL v3 licence, and the OpenAP library (https://github.com/TUDelft-CNS-ATM/openap), released
under the LGPL v3 licence.

Copyright 2023 Gérald Gurtner, Luis Delgado, University of Westminster,
and Innaxis.

All subsequent copyright belongs to the respective contributors.

.. inclusion-marker-do-not-remove4


