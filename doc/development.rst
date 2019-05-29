Development
===========

Software
--------

Please run

    pip install --user --upgrade -r requirements-dev.txt

to install additional packages required for development.

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

this helps to get an overview over what's happening, e.g. when compiling
release notes.

Git hooks
---------

Please install out git pre-commit hooks:

    pre-commit install

Now, every time you commit to this package, a number of checks and cleanups
are performed, among them

* Code styling with `black <https://github.com/python/black>`_
* Stripping output of jupyter notebooks with `nbstripout <https://github.com/kynan/nbstripout>`_

Unittests
---------

Whenever changing functionality, please run

    pytest

or alternatively

    nose2

to run all unittests.
