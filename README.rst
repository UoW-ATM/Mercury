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

Quickstart
==========

Docker
------

NEW! Docker versions are now available for:

- the command line interface (CLI) version: `mercury_cli <https://github.com/orgs/UoW-ATM/packages/container/package/mercury%2Fmercury_cli>`_;
- the notebook version: `mercury_nb <https://github.com/orgs/UoW-ATM/packages/container/package/mercury%2Fmercury_nb>`_;
- the GUI version: (coming soon).

Docker allows you to use the model on any OS without installing anything except a docker environment (see https://www.docker.com/get-started/).
With a terminal (e.g. powershell in windows), you can download the docker images like this:

.. code:: bash

    docker pull ghcr.io/uow-atm/mercury/mercury_nb:latest
    docker tag ghcr.io/uow-atm/mercury/mercury_nb:latest mercury_nb
    docker pull ghcr.io/uow-atm/mercury/mercury_cli:latest
    docker tag ghcr.io/uow-atm/mercury/mercury_cli:latest mercury_cli

In a terminal you can then use the docker image like this:

- for the CLI:

.. code:: bash

    docker run mercury_cli -id -1 -cs -1

- for the notebook:

.. code:: bash

    docker run -p 8888:8888 mercury_nb

You probably need to copy the url appearing in the terminal after this command and copy/paste it into your browser.

- for the GUI: (coming soon)



Installation
------------
Mercury has been tested on ubuntu-like machines and to a lesser extent Windows, using anaconda. Python 3.10 is
recommended, Python 3.12 will raise issues.

-  Start by cloning the repository, for instance:

.. code:: bash

    git clone https://github.com/UoW-ATM/Mercury

-  Use this to download the third party libraries:

.. code:: bash

   cd Mercury
   git submodule update --recursive --remote --init

-  In a fresh python environment, install all the required packages:

In Linux, use:

.. code:: bash

   sudo apt-get install libproj-dev libgeos-dev build-essential python3-dev proj-data proj-bin
   python -m pip install shapely cartopy --no-binary shapely --no-binary cartopy
   pip install -r requirements.txt

In Windows, you probably just need to install the requirements:

.. code-block:: bash

    pip install -r requirements.txt

You may also need to install Visual studio C++ built tools if it's not the case already.

-  Download the sample data here:
   https://zenodo.org/records/11384379/files/Mercury_data_sample.zip?download=1. Extract the data. Put the results "input" folder outside of the main Mercury root folder (side by side).

By default, Mercury uses the `OpenAP <https://github.com/TUDelft-CNS-ATM/openap>`_ model for aircraft performance.
However, Mercury also supports the BADA models developed by EUROCONTROL. If you want to use it, you can request a licence
from EUROCONTROL (here: https://www.eurocontrol.int/model/bada), then use the script ``generate_bada3_input.py`` to
transform the AFP, OFP and PTD files from BADA3 into tables (parquet files) that will be read by Mercury.
In the following command, replace ``BADA3_FILES_PATH`` with the location of the downloaded bada files:

.. code:: bash

   python generate_bada3_input.py -s BADA3_FILES_PATH -d .

Ensure you copy the generated parquet files into
``/home/earendil/Documents/Westminster/Mercury/Mercury/libs/performance_models/bada3/data/``.

If you want to use BADA4, please contact us directly and we'll offer general guidance. We are also working on a support
for EUROCONTROL's pyBADA library.

Running the CLI version
-----------------------

You can test the model by running:

.. code:: bash

   ./mercury.py -id -1

Use ``-h`` to have list of all the possible arguments.

Programmatic use of Mercury
---------------------------

Mercury can be used as an object. An example of its use and some
examples to run can be found in the ``Mercury.ipynb`` Jupyter notebook.
The notebook shows the possible uses of Mercury in terms of parameter
setting, scenarios, case study, etc.

Graphical interface
-------------------

You can use a GUI to explore the data input and output structure, create
new scenarios, case studies, etc. Use the following command to start it:

.. code:: bash

   cd dashboard
   python mercury_gui.py

.. inclusion-marker-do-not-remove2

Manual and references
=====================

A more complete manual is in construction and can be found here_.

.. _here: https://uow-atm.github.io/Mercury

The following articles can also be consulted to learn more about
Mercury:

-  the one included in `the repo <https://github.com/UoW-ATM/Mercury/blob/master/docs/SIDs_2023_OpenMercury.pdf>`_, presenting the general approach to Mercury,
-  the one available there_ showing some details of Mercury and some examples of its uses, from a few years back.

.. _there: https://www.sciencedirect.com/science/article/abs/pii/S0968090X21003600

.. inclusion-marker-do-not-remove3

About
=====

Authorship
----------

Up to the open source release, all Mercury code has been written by
Gérald Gurtner and Luis Delgado, to the exception of:

-  The Dynamic Cost Indexing module, written by Damir Valput
-  The GUI, written by Michal Weiszer

We thank also Tanja Bolic for many waves of testing.

Licence and copyright
---------------------

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


