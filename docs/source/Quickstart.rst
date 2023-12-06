Quickstart
===========

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

.. code:: bash

   sudo apt-get install libproj-dev libgeos-dev build-essential python3-dev proj-data proj-bin
   python -m pip install shapely cartopy --no-binary shapely --no-binary cartopy
   pip install -r requirements.txt

-  Download the sample data here:
   https://zenodo.org/doi/10.5281/zenodo.10211704. By default, Mercury
   will look for them just outside the root directory in a folder named
   “input”.

-  To run Mercury, you need to use bada 3 or bada 4 (not supported for
   now). You can request a licence from EUROCONTROL, then use the script
   generate_bada3_input.py to transform the AFP, OFP and PTD files from
   BADA into tables (parquet files) that will be read by Mercury. The
   script has a help function.

.. code:: bash

   ./generate_bada3_input.py -s path_to/bada3_files -d path_to/processed_bada3_files

-  Ensure you copy the generated parquet files into the
   data/ac_performance/bada/bada3/ of your input folder (per scenario).
   The location of the performance files might be modified in the
   future.

-  Note: support for OpenAP, an open alternative to BADA, is under
   development.

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