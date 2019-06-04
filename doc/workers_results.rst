Workers and Results
===================

Operations on the data (represented by a :class:`~clusterking.data.Data` object) are
performed by worker classes, which are formally a subclass of the
:class:`~clusterking.worker.Worker` class.

Usually the workflow looks as follows:

1. Initialize the worker class ``w = Worker()``
2. Configure the worker class by applying a set of methods: ``w.set_metric(...)``, ``w.configure_fom(...)``` etc.
3. Run the worker class on a :class:`~clusterking.data.Data` object: ``r = w.run(d)``.
   This returns a result object ``r``.

Running a worker class returns a result class, which is formally a subclass of the
:class:`~clusterking.result.Result`` class.

Most prominently, it has a ``write`` method, that allows to writes the relevant
part of the results back to the :class:`~clusterking.data.Data` object. Thus the
workflow continues as

4. Write back to data object: ``r.write()``.

.. automodule:: clusterking.worker

``Worker``
---------------------

    .. autoclass:: Worker
        :members:
        :undoc-members:

.. automodule:: clusterking.result

``Result``
---------------------

    .. autoclass:: Result
        :members:
        :undoc-members:
