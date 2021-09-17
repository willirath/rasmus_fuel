.. _contribute:

Contributor Guide
=================

Development environment
-----------------------

Fork the repository and download the code
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To further be able to submit modifications, it is preferable to start by
forking the rasmus_fuel repository on GitHub_ (you need to have an account).

Then clone your fork locally::

  $ git clone git@github.com:your_name_here/rasmus_fuel.git

Alternatively, if you don't plan to submit any modification, you can clone the
original rasmus_fuel git repository::

   $ git clone git@github.com:willirath/rasmus_fuel.git

.. _GitHub: https://github.com

Install
~~~~~~~

To install the dependencies, we recommend using the conda_ package manager with
the conda-forge_ channel. For development purpose, you might consider installing
the packages in a new conda environment::

  $ conda create -n rasmus_fuel_dev python numpy
  $ conda activate rasmus_fuel_dev

Then install rasmus_fuel locally (in development mode) using ``pip``::

  $ cd rasmus_fuel
  $ python -m pip install -e .

.. _conda: http://conda.pydata.org/docs/
.. _conda-forge: https://conda-forge.github.io/

Pre-commit
~~~~~~~~~~

rasmus_fuel provides a configuration for `pre-commit <https://pre-commit.com>`_, which
can be used to ensure that code-style and code formatting is consistent.

First install ``pre-commit``::

  $ conda install pre-commit -c conda-forge

Then run the following command to activate it in the current repository::

  $ pre-commit install

From now on ``pre-commit`` will run whenever you commit with ``git``.

Run tests
~~~~~~~~~

To make sure everything behaves as expected, you may want to run rasmus_fuel's unit
tests locally using the `pytest`_ package. You can first install it with conda::

  $ conda install pytest pytest-cov -c conda-forge

Then you can run tests from the main rasmus_fuel directory::

  $ pytest . --verbose

.. _pytest: https://docs.pytest.org/en/latest/

Contributing to code
--------------------

Below are some useful pieces of information in case you want to contribute
to the code.

Local development
~~~~~~~~~~~~~~~~~

Once you have set up the development environment, the next step is to create
a new git branch for local development::

  $ git checkout -b name-of-your-bugfix-or-feature

Now you can make your changes locally.

Submit changes
~~~~~~~~~~~~~~

Once you are done with the changes, you can commit your changes to git and
push your branch to your rasmus_fuel fork on GitHub::

  $ git add .
  $ git commit -m "Your detailed description of your changes."
  $ git push origin name-of-your-bugfix-or-feature

(note: this operation may be repeated several times).

When committing, ``pre-commit`` will re-format the files if necessary.

We you are ready, you can create a new pull request through the GitHub_ website
(note that it is still possible to submit changes after your created a pull
request).

Contributing to documentation
-----------------------------

Documentation is maintained in the RestructuredText markup language (``.rst``
files) in the ``doc`` folder.

To build the documentation locally, first install some extra requirements::

   $ conda install sphinx sphinx_rtd_theme sphinx-autosummary-accessors -c conda-forge

Then build the documentation with ``make``::

   $ cd doc
   $ make html

The resulting HTML files end up in the ``build/html`` directory.

You can now make edits to rst files and run ``make html`` again to update
the affected pages.

.. _Sphinx: http://www.sphinx-doc.org/

Docstrings
~~~~~~~~~~

Everything (i.e., classes, methods, functions...) that is part of the public API
should follow the numpydoc_ standard when possible.

.. _numpydoc: https://numpydoc.readthedocs.io/en/latest/format.html#docstring-standard
