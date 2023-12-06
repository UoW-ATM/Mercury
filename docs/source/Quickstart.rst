Quickstart
===========

Mercury has been tested on ubuntu-like machines and to a lesser extent Windows, using anaconda. Python 3.10 is
recommended, Python 3.12 will raise issues.

Installation
------------

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
   https://zenodo.org/doi/10.5281/zenodo.10211704. Extract the data. Put the results "input" folder outside of the main
Mercury root folder (side by side).

-  Mercury requires aircraft performance models developed by EUROCONTROL, BADA3.
You can request a licence from EUROCONTROL (here: https://www.eurocontrol.int/model/bada), then use the script ``generate_bada3_input.py`` to transform the AFP, OFP
and PTD files from BADA3 into tables (parquet files) that will be read by Mercury. In the following command, replace
``BADA3_FILES_PATH`` with the location of the downloaded bada files:

.. code:: bash

   python generate_bada3_input.py -s BADA3_FILES_PATH -d .

Ensure you copy the generated parquet files into the data/ac_performance/bada/bada3/ of your input folder (per scenario).
The location of the performance files might be modified in the future.

-  Note: support for OpenAP, an open alternative to BADA, is under development.

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
   ./mercury_gui.py