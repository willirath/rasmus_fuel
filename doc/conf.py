# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

sys.path.insert(0, os.path.abspath('../src/rasmus_fuel/'))

import sphinx_autosummary_accessors

import rasmus_fuel  # noqa

# -- Project information -----------------------------------------------------

project = 'rasmus_fuel'
copyright = '2021, Willi Rath, Elena Shchekinova'
author = 'Willi Rath, Elena Shchekinova'


# -- General configuration ---------------------------------------------------

master_doc = 'index'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx.ext.intersphinx',
    'sphinx.ext.extlinks',
    'sphinx_autosummary_accessors',
    'nbsphinx',
    'myst_parser',
]

extlinks = {
    'issue': ('https://github.com/willirath/rasmus_fuel/issues/%s', '#'),
    'pull': ('https://github.com/willirath/rasmus_fuel/pull/%s', '#'),
}

# Add any paths that contain templates here, relative to this directory.
templates_path = [
    '_templates',
    sphinx_autosummary_accessors.templates_path,
]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
# html_theme_options = {
#    "github_url": "https://github.com/willirath/rasmus_fuel",
#    "use_edit_page_button": True,
#    "search_bar_position": "navbar",
# }


html_context = {
    'github_user': 'willirath',
    'github_repo': 'rasmus_fuel',
    'github_version': 'master',
    'doc_path': 'doc',
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']


# Enable notebook execution
# https://nbsphinx.readthedocs.io/en/0.4.2/never-execute.html
# nbsphinx_execute = 'auto'
# Allow errors in all notebooks by
# nbsphinx_allow_errors = True
nbsphinx_kernel_name = 'python3'

# Disable cell timeout
nbsphinx_timeout = -1

nbsphinx_prolog = """
{% set docname = env.doc2path(env.docname, base=None) %}

|Binder|

You can run this notebook in a `live session <https://mybinder.org/v2/gh/willirath/rasmus_fuel/master?filepath=doc/{{
docname }}>`_ or view it `on Github <https://github.com/willirath/rasmus_fuel/blob/master/doc/{{ docname }}>`_.

.. |Binder| image:: https://mybinder.org/badge_logo.svg
   :target: https://mybinder.org/v2/gh/willirath/rasmus_fuel/master?filepath=doc/{{ docname }}
"""


intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'numpy': ('https://numpy.org/', None),
}

autosummary_generate = True

autodoc_typehints = 'none'

napoleon_use_param = True
napoleon_use_rtype = True
