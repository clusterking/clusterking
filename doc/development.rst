Development
===========

When developing code for ClusterKinG, please install our
`pre-commit hook <https://git-scm.com/book/en/v2/Customizing-Git-Git-Hooks>`_
which will automatically

* Run all unittests
* Strip output from jupyter notebooks
* Reformats the code using `black <https://github.com/python/black>`_

The hooks can be installed by running

.. code-block:: bash

    hoooks/install_hooks.sh
