Installation
============

Basic installation:

.. code:: sh

    pip3 install --user --upgrade clusterking[,plotting]

If you do not require plotting, you can remove ``[plotting]``, which adds
``matplotlib`` as a dependency.
If you are on MaxOS, you might want to check out the
`matplotlib documentation <https://matplotlib.org/3.1.0/faq/osx_framework.html>`_
on how to install matplotlib and install it **prior** to installing matplotlib.

For the latest development version type:

.. code:: sh

    git clone https://github.com/clusterking/clusterking/
    cd clusterking
    pip3 install --user --editable .[plotting,dev]

Here, ``[,plotting,dev]`` also installs additional packages that are needed
for development, such as support for unittests, etc.
