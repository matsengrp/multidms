# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Extension configuration -------------------------------------------------

# add sourcecode to path
import sys, os

sys.path.insert(0, os.path.abspath("../multidms"))
# sys.path.insert(0, "{0}/..".format(os.path.abspath(".")))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'multidms'
copyright = '2023, Jared Galloway, Hugh Haddox'
author = 'Jared Galloway'
release = '0.0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx_rtd_theme',
    "sphinx.ext.autodoc",
    #"sphinx.ext.doctest",
    "sphinx.ext.mathjax",
    "sphinx.ext.githubpages",
    #"sphinx.ext.viewcode",
    #"sphinx.ext.napoleon",
    # "matplotlib.sphinxext.plot_directive",
    "nbsphinx",
    "nbsphinx_link"
]

templates_path = ['_templates']
# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
# source_suffix = ['.rst', '.md']
source_suffix = ".rst"

# The master toctree document.
master_doc = "index"

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = 'en'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
# html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

autodoc_member_order = 'bysource'

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
#
html_theme_options = {
    "logo": "2020-logo-1000px-transparent.png",
    "logo_name": "true",
    "github_button": "true",
    "github_user": "matsengrp",
    "github_repo": "multidms",
    "github_banner": "true",
    "travis_button": "false",
    "page_width": "1300px",
    "sidebar_width": "250px",
}
