""" Miscellaneous utilities """

import collections

def nested_dict():
    """ This is very clever and stolen from
    https://stackoverflow.com/questions/16724788/
    Use it to initialize a dictionary-like object which automatically adds
    levels.
    E.g.
        a = nested_dict()
        a['test']['this']['is']['working'] = "yaaay"
    """
    return collections.defaultdict(nested_dict)