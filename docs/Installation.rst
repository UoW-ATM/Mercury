.. _installation:

Installation
============

Docker
------

Docker installation
^^^^^^^^^^^^^^^^^^^

`Tested in Windows and Linux, should be running in MacOS.`

An easy way to use Mercury is to use the docker images. Docker is an open-source platform designed to automate the
deployment, scaling, and management of applications. It achieves this by using containerization technology,
which allows applications to run in isolated environments called containers.

Before using Docker, you will need to check if virtualisation is enabled on your machine (see for instance
`here for Windows <https://techviral.net/check-if-virtualization-is-enabled>`_). If virtualisation is disabled,
you can either try to enable it or switch to a full installation of Mercury, see below.

If the virtualisation is enabled, you can install docker in Linux, see
`here <https://www.docker.com/products/docker-desktop/>`_. Once the Docker is installed, you can download the Mercury images
using:

.. code:: bash

    docker pull ghcr.io/uow-atm/mercury/mercury_nb:latest
    docker tag ghcr.io/uow-atm/mercury/mercury_nb:latest mercury_nb
    docker pull ghcr.io/uow-atm/mercury/mercury_cli:latest
    docker tag ghcr.io/uow-atm/mercury/mercury_cli:latest mercury_cli

in a terminal.

Docker usage
^^^^^^^^^^^^

You can then use the CLI versions or launch a notebook from your terminal. For the CLI you need to run:

.. code:: bash

    docker run mercury_cli -id -1 -cs -1

You should now be able to use the CLI exactly like in the full installation version (see :ref:`cli`), just prefacing every command by
``docker run``.

For the notebook, you can run:

.. code:: bash

    docker run -p 8888:8888 mercury_nb

and then copy the url that appears in the terminal to your browser. Then open the Mercury.ipynb and you can use Mercury
normally (see :ref:`notebook`).

However, using docker like this means that input and output folders are also part of the image. You can use bash to
copy data back and forth, but it's not very handy. Instead, you can link folders on your machine to the folders inside
docker. First, download and extract the data as explained in the next section :ref:`installation_data`.
Then use the following command to link this input folder to the image, and link another result folder as well
(note: this method only works for the CLI for now, and might not work in Windows):

.. code:: bash

    docker run --user $(id -u):$(id -g) -v /absolute/path/to/your/host/results:/app/results -v /absolute/path/to/your/host/input:/app/input mercury_cli -id -1 -cs -1

In Linux you can even use an alias to shorten the command. Edit your ``~/.bashrc`` file and add the line:

.. code:: bash

    alias mercury_cli_docker='docker run --user $(id -u):$(id -g) -v /absolute/path/to/your/host/results:/app/results -v /absolute/path/to/your/host/input:/app/input mercury_cli'

Restart your terminal, and you can now run:

.. code:: bash

    mercury_cli_docker -id -1 -cs -1

And the results should appear in the ``/absolute/path/to/your/host/results`` folder.


Full installation
-----------------

`Tested in Linux Mint 21.3, Kubuntu 22.04.4, Windows 10 with miniconda, and Ubuntu 18.04 in Windows with WSL.
Some issues are expected with MacOS. Python 3.12 raises some issues, 3.10 is safe.`

Installing dependencies
^^^^^^^^^^^^^^^^^^^^^^^

To install natively Mercury on your machine, you can follow these steps:

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

In Windows powershell, in the dedicated environment, you need to install the requirements:

.. code-block:: bash

    pip install -r requirements.txt

You may also need to install Visual studio C++ built tools if it's not the case already.


.. _installation_data:

Setting up data and performance models
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A sample of synthetic data is included with Mercury, and can be downloaded
`here <https://zenodo.org/records/11384379/files/Mercury_data_sample.zip?download=1>`_. After extracting the data, put
the "input" folder outside of the main Mercury root folder (side by side) (or you'll need to change the default path to
input data, see :ref:`input_data`).

By default, Mercury uses the `OpenAP <https://github.com/TUDelft-CNS-ATM/openap>`_ model for aircraft performance.
However, Mercury also supports the BADA models developed by EUROCONTROL. If you want to use BADA, you can request a licence
from EUROCONTROL (here: https://www.eurocontrol.int/model/bada), then use the script ``generate_bada3_input.py`` to
transform the AFP, OFP and PTD files from BADA3 into tables (parquet files) that will be read by Mercury.
In the following command, replace ``BADA3_FILES_PATH`` with the location of the downloaded bada files:

.. code:: bash

   python generate_bada3_input.py -s BADA3_FILES_PATH -d .

Ensure you copy the generated parquet files into
``Mercury/libs/performance_models/bada3/data/``.

If you want to use BADA4, please contact us directly and we'll offer general guidance. We are also working on a support
for EUROCONTROL's pyBADA library.




