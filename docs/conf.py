# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
sys.path.insert(0, os.path.abspath('..'))
sys.path.insert(0, os.path.abspath('../..'))


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Mercury'
copyright = '2023, University of Westminster, Innaxis, Luis Delgado, Gerald Gurtner'
author = 'Mercury team'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.autodoc',
				'sphinx.ext.autosummary',
			  'sphinx.ext.napoleon']

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster' # 'nature'
html_static_path = ['_static']

html_theme_options = {
    'logo': 'mercury_logo_small.png',
    'github_user': 'UoW-ATM',
    'github_repo': 'Mercury',
	'description': 'An open source mobility simulator',
	'github_banner': True,
}

