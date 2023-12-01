![mercury_logo.png](mercury_logo_small.png)


Mercury is a research-oriented air transportation mobility simulator with a strong agent-based paradigm.

# Quick Setup

- Start by cloning the repository, for instance:

```commandline
git clone https://github.com/UoW-ATM/Mercury
```

- Use this to download the third party libraries:
```commandline
cd Mercury
git submodule update --recursive --remote --init
```

- In a fresh python environment, install all the required packages:
```commandline
pip install -r requirements.txt
```

- Download the sample data:
  - By default, Mercury will look for them just outside the root directory in a folder named "input".
  - The most up-to-date dataset (to be used with the current code version) is at https://zenodo.org/doi/10.5281/zenodo.10211704.
  - The dataset to be used with release V3.0 is at: https://zenodo.org/records/10222526
  
- To run Mercury, you need to use BADA 3 or BADA 4 (not supported for now). You can request a licence from EUROCONTROL,
then use the script generate_bada3_input.py to transform the AFP, OFP and PTD files from BADA into tables (parquet files) that Mercury will read. The script has a help function.
```commandline
./generate_bada3_input.py -s path_to/bada3_files -d path_to/processed_bada3_files
```
- Ensure you copy the generated parquet files into the data/ac_performance/bada/bada3/ of your input folder (per scenario). The location of the performance files might be modified in the future.

- Note: support for OpenAP, an open alternative to BADA, is under development.


You can test the model by running:
```commandline
./mercury.py -id -1
```
Or using the jupyter notebook "Mercury.ipynb", see below.

# Graphical interface

You can use a GUI to explore the data input and output structure, create new scenarios, case studies, etc. Use the 
following command to start it:
```commandline
dashbaord/mercury_gui.py
```

# Programmatic use of Mercury

Mercury can be used as an object. An example of its use and some examples of how to run it can be found in 
the Mercury.ipynb Jupyter notebook. The notebook shows the possible uses of Mercury in terms of parameter setting, 
scenarios, case studies, etc.

# Manual and documentation
A more complete manual is in construction. An automatically generated documentation for the repository can be found in 
doc/.

The following articles can also be consulted to learn more about Mercury:
- the one included in doc/SIDs_2023_OpenMercury.pdf, presenting the general approach to Mercury.
- the one available at this address https://www.sciencedirect.com/science/article/abs/pii/S0968090X21003600, shows 
some details of Mercury and some examples of its uses from a few years back.
- Deliverable 4.1-Initial model design from Domino project, where Mercury was reimplemented as an ABM system, describes the main design approaches and the initial version of the roles/agents considered: https://ec.europa.eu/research/participants/documents/downloadPublic?documentIds=080166e5bf17614a&appId=PPGMS


# Authorship

Up to the open source release, all Mercury code has been written by Gérald Gurtner and Luis Delgado, with the exception of:

- The Dynamic Cost Indexing module, written by Damir Valput
- The GUI, written by Michal Weiszer

We also thank Tanja Bolic for many waves of testing.

# Licence and copyright

Mercury is released under the GPL v3 licence. The licence can be found in LICENCE.TXT

Mercury uses the Hotspot library (https://github.com/andygaspar/Hotspot), also released under GPL v3, and
the uow-belt-tools library (https://github.com/UoW-ATM/uow_tool_belt), released under the GPL v3 licence too.

Copyright 2023 Gérald Gurtner, Luis Delgado, University of Westminster, and Innaxis.

All subsequent copyright belongs to the respective contributors.
