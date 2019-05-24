# Hooks for git

This is to set up a hook for git that

* Runs unit tests (using nose or pytest)
* Cleans output from jupyter notebooks (albeit not perfectly)
* Reformats code using black

## Installation:

Prerequisites:

    pip3 install --user black pytest pytest-subtests

Simply run

    ./install_hooks.sh
