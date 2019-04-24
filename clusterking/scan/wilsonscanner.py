#!/usr/bin/env python3

# std
from typing import Dict
import itertools

# 3rd
import wilson
import numpy as np

# ours
from clusterking.scan.scanner import Scanner


class WilsonScanner(Scanner):
    """ Scans the NP parameter space in a grid and also in the kinematic
    variable.

    Usage example:

    .. code-block:: python

        import flavio
        import functools
        import numpy as np
        import clusterking as ck

        # Initialize Scanner object
        s = ck.scan.WilsonScanner()

        # Sample 4 points for each of the 5 Wilson coefficients
        s.set_spoints_equidist(
            {
                "CVL_bctaunutau": (-1, 1, 4),
                "CSL_bctaunutau": (-1, 1, 4),
                "CT_bctaunutau": (-1, 1, 4)
            },
            scale=5,
            eft='WET',
            basis='flavio'
        )

        # Set function and binning
        s.set_dfunction(
            functools.partial(flavio.np_prediction, "dBR/dq2(B+->Dtaunu)"),
            binning=np.linspace(3.15, 11.66, 10),
            normalize=True
        )

        # Initialize a Data objects to write to
        d = ck.Data()

        # Start running with maximally 3 cores and write the results to Data
        s.run(d)

    """
    def __init__(self):
        """ Initializes the :class:`clusterking.scan.WilsonScanner` class. """
        super().__init__()
        self.imaginary_prefix = "im_"

    @property
    def imaginary_prefix(self) -> str:
        """ Prefix for the name of imaginary parts of coefficients.
        Also see the documentation of :meth:`.set_spoints_equidistant`.
        """
        return self.md["imaginary_prefix"]

    @imaginary_prefix.setter
    def imaginary_prefix(self, value: str) -> None:
        self.md["imaginary_prefix"] = value

    def set_spoints_grid(self, values, scale, eft, basis) -> None:
        """ Set a grid of points in wilson space.

        Args:
            values: A dictionary of the following form:

                .. code-block:: python

                    {
                        <wilson coeff name>: [
                            value_1,
                            ...,
                            value_n
                        ]
                    }

                where ``value_1``, ..., ``value_n`` can be complex numbers in
                general.

            scale: Wilson coeff input scale in GeV
            eft: Wilson coeff input eft
            basis: Wilson coeff input basis
        """

        # Important to remember the order now, because of what we do next.
        # Dicts are NOT ordered
        coeffs = list(values.keys())
        # It's very important to sort the coefficient names here, because when
        # calling wilson.Wilson(...).wc.values() later, these will also
        # be alphabetically ordered.
        coeffs.sort()
        # Nowe we collect all lists of values.
        values_lists = [
            values[coeff] for coeff in coeffs
        ]
        # Now we build the cartesian product, i.e.
        # [a1, a2, ...] x [b1, b2, ...] x ... x [z1, z2, ...] =
        # [(a1, b1, ..., z1), ..., (a2, b2, ..., z2)]
        cartesians = list(itertools.product(*values_lists))

        # And build wilson coefficients from this
        self._spoints = [
            wilson.Wilson(
                wcdict={
                    coeffs[icoeff]: cartesian[icoeff]
                    for icoeff in range(len(coeffs))
                },
                scale=scale,
                eft=eft,
                basis=basis
            )
            for cartesian in cartesians
        ]

        # fixme: why is this flagged by pycharm as defined outside of __init__?
        self.coeffs = list(values.keys())
        md = self.md["spoints"]
        md["values"] = values
        md["scale"] = scale
        md["eft"] = eft
        md["basis"] = basis

    def set_spoints_equidist(self, ranges: Dict[str, tuple], scale: float,
                             eft: str, basis: str) -> None:
        """ Set a list of 'equidistant' points in wilson space.

        Args:
            ranges: A dictionary of the following form:

                .. code-block:: python

                    {
                        <wilson coeff name>: (
                            <Minimum of wilson coeff>,
                            <Maximum of wilson coeff>,
                            <Number of bins between min and max>,
                        )
                    }

            scale: <Wilson coeff input scale in GeV>,
            eft: <Wilson coeff input eft>,
            basis: <Wilson coeff input basis>

        .. note::

            In order to add imaginary parts to your wilson coefficients,
            prepend their name with ``im_`` (you can customize this prefix by
            setting the :attr:`.imaginary_prefix` attribute to a custom value.)

            Example:

            .. code-block:: python

                ws = WilsonScanner()
                ws.set_spoints_equidist(
                    {
                        "a": (-2, 2, 4),
                        "img_a": (-1, 1, 10),
                    },
                    ...
                )

            Will sample the real part of ``a`` in 4 points between -2 and 2 and
            the imaginary part of ``a`` in 10 points between -1 and 1.

        Returns:
            None
        """
        # Because of our hack with the imaginary prefix, let's first see which
        # coefficients we really have

        def is_imaginary(name: str) -> bool:
            return name.startswith(self.imaginary_prefix)

        def real_part(name: str) -> str:
            if is_imaginary(name):
                return name.replace(self.imaginary_prefix, "", 1)
            else:
                return name

        def imaginary_part(name: str) -> str:
            if not is_imaginary(name):
                return self.imaginary_prefix + name
            else:
                return name

        coeffs = list(set(map(real_part, ranges.keys())))

        grid_config = {}
        for coeff in coeffs:
            # Now let's always collect the values of the real part and of the
            # imaginary part
            res = [0.]
            ims = [0.]
            if real_part(coeff) in ranges:
                res = list(np.linspace(*ranges[real_part(coeff)]))
            if imaginary_part(coeff) in ranges:
                ims = list(np.linspace(*ranges[imaginary_part(coeff)]))
            # And basically take their cartesian product, alias initialize
            # the complex number.
            grid_config[coeff] = [
                complex(x, y)
                for x in res
                for y in ims
            ]

        self.set_spoints_grid(
            grid_config,
            scale=scale,
            eft=eft,
            basis=basis,
        )
        # Make sure to do this after set_spoints_grid, so we overwrite
        # the relevant parts.
        md = self.md["spoints"]
        md["sampling"] = "equidistant"
        md["ranges"] = ranges
