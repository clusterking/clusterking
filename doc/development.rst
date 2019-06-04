Development
===========

Software
--------

Please install the ClusterKinG package with the development packages from the
latest master version on github:

.. code-block:: shell

    git clone https://github.com/clusterking/clusterking
    cd clusterking
    pip3 install --user .[plotting,dev]

This will enable you to run our unittests and more.


Git hooks
---------

Please install our git pre-commit hooks:

.. code-block:: shell

    pip3 install --user pre-commit
    pre-commit install

Now, every time you commit to this package, a number of checks and cleanups
are performed, among them

* Code styling with `black <https://github.com/python/black>`_
* Stripping output of jupyter notebooks with `nbstripout <https://github.com/kynan/nbstripout>`_

Git commit message
------------------

It's recommended to use the following prefixes:

* ``[Fix]``: Fixing a bug
* ``[Int]``: Interface change
* ``[Feat]``: New feature
* ``[Doc]``: Everything regarding documentation
* ``[CI]``: Continuus Integration (unittests and more)
* ``[Ref]``: Code refactoring
* ``[Clean]``: Code cleanup (style improvement etc.)
* ``[Deploy]``: Everything that has to do with releases
* ``[Dev]``: Things that are only relevant to developers

this helps to get an overview over what's happening, e.g. when compiling
release notes.

Unittests
---------

Whenever changing functionality, please run

.. code-block:: shell

    pytest

or alternatively

.. code-block:: shell

    nose2

to run all unittests.
