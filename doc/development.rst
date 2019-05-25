Development
===========

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

When developing code for ClusterKinG, please install our
`pre-commit hook <https://git-scm.com/book/en/v2/Customizing-Git-Git-Hooks>`_
which will automatically

* Run all unittests
* Strip output from jupyter notebooks
* Reformats the code using `black <https://github.com/python/black>`_,

whenever you commit to git.

The hooks can be installed by running

.. code-block:: bash

    hoooks/install_hooks.sh
