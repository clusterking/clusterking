#!/usr/bin/env python3

""" Miscellaneous utilities """

# std
import collections
from collections.abc import Iterable
import json
import pathlib
import time
from typing import Dict

# 3rd party
try:
    import git
except ImportError:
    git = None


def nested_dict():
    """ This is very clever and stolen from
    https://stackoverflow.com/questions/16724788/
    Use it to initialize a dictionary-like object which automatically adds
    levels.
    E.g.

    .. code-block:: python

        a = nested_dict()
        a['test']['this']['is']['working'] = "yaaay"
    """
    return collections.defaultdict(nested_dict)


def git_info(log=None, path=None) -> Dict[str, str]:
    """ Return dictionary containing status of the git repository (commit hash,
    date etc.

    Args:
        log: logging.Logger object (optional)
        path: path to .git subfolder or search path (optional)

    Returns:
        dictionary
    """
    # Fill in some dummy values first
    git_config = {
        "branch": "unknown",
        "sha": "unknown",
        "msg": "unknown",
        "time": "unknown"
    }

    if git is None:
        msg_warn = "Module 'git' not found, will not add git version " \
                   "information to the output files."
        msg_debug = "Install the 'git' module by running " \
                    "'sudo pip3 install gitpython' or similar. "
        if log:
            log.warning(msg_warn)
            log.debug(msg_debug)
        else:
            print(msg_warn)
            print(msg_debug)
        return git_config

    if not path:
        # give git.Repo the directory that includes this file as directory
        # and let it search
        this_dir = pathlib.Path(__file__)
        path = this_dir
    try:
        repo = git.Repo(path=path, search_parent_directories=True)
    except git.InvalidGitRepositoryError:
        return git_config

    git_config["branch"] = repo.head.name
    hcommit = repo.head.commit
    git_config["sha"] = hcommit.hexsha
    git_config["msg"] = hcommit.message.strip("\n")
    commit_time = hcommit.committed_date
    git_config["time"] = time.strftime("%a %_d %b %Y %H:%M",
                                       time.gmtime(commit_time))
    # todo: also add a nice string representation of git diff?
    return git_config


def save_git_info(output_path=None, *args, **kwargs) -> Dict[str, str]:
    """
    Save output of git_info to a file.
    
    Args:
        output_path: Output path. If None, the default will be 
            bclustering/git_info.json
        *args: Passed on to git_info
        **kwargs: Passed on to git_info

    Returns:
        Output of git_info
    """
    if output_path:
        output_path = pathlib.Path(output_path)
    if not output_path:
        this_dir = pathlib.Path(__file__).parent.resolve()
        output_path = this_dir / ".." / "git_info.json"
    gi = git_info(*args, **kwargs)
    with output_path.open("w") as output_file:
        json.dump(
            gi,
            output_file,
            indent=4,
            sort_keys=True
        )
    return gi


def load_git_info(input_path=None) -> Dict[str, str]:
    """
    Load previously saved output of git_info from a json file.
    
    Args:
        input_path: Input path to json file. If None, the default will be
            bclustering/git_info.json

    Returns:
        Parsed json file (should be identical to saved output of git_info).
    """
    if input_path:
        input_path = pathlib.Path(input_path)
    if not input_path:
        this_dir = pathlib.Path(__file__).parent.resolve()
        input_path = this_dir / ".." / "git_info.json"
    with input_path.open() as input_file:
        info = json.loads(input_file.read())
    return info


# todo: docstring
def failsafe_serialize(obj):
    if isinstance(obj, dict):
        return {key: failsafe_serialize(v) for key, v in obj.items()}
    elif isinstance(obj, Iterable) and not isinstance(obj, str):
        return [failsafe_serialize(v) for v in obj]
    elif isinstance(obj, (int, float)):
        return obj
    else:
        return str(obj)


if __name__ == "__main__":
    print("Testing git_info")
    print(git_info())
    print("Saving git_info")
    save_git_info()
    print("Loading git info again")
    print(load_git_info())
