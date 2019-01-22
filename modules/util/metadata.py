""" Miscellaneous utilities """

import collections
try:
    import git
except ImportError:
    git = None
import time
import pathlib


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


def git_info(log=None, path=None):
    """ Return dictionary containing status of the git repository (commit hash,
    date etc.

    Args:
        log: logging.Logger object (optional)

    Returns:
        dictionary
    """
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
        return

    git_config = {}
    if not path:
        # give git.Repo the directory that includes this file as directory
        # and let it search
        this_dir = pathlib.Path(__file__)
        path = this_dir
    repo = git.Repo(path=path, search_parent_directories=True)
    git_config["branch"] = repo.head.name
    hcommit = repo.head.commit
    git_config["sha"] = hcommit.hexsha
    git_config["msg"] = hcommit.message
    commit_time = hcommit.committed_date
    git_config["time"] = time.strftime("%a %_d %b %Y %H:%M",
                                       time.gmtime(commit_time))
    # todo: also add a nice string representation of git diff?
    return git_config


if __name__ == "__main__":
    print("Testing git_info")
    print(git_info())