#!/bin/bash

# ==============================================================================
# strip output of IPython Notebooks
# FROM: https://gist.github.com/minrk/6176788
# ==============================================================================

if git rev-parse --verify HEAD >/dev/null 2>&1; then
   against=HEAD
else
   # Initial commit: diff against an empty tree object
   against=4b825dc642cb6eb9a060e54bf8d69288fbee4904
fi

# Find notebooks to be committed
(
IFS='
'
NBS=`git diff-index $against --name-only | grep '\.ipynb$' | uniq`

for NB in $NBS ; do
    echo "Removing outputs from $NB"
    hooks/nbstripout "$NB"
    git add "$NB"
done
)


# todo: What did this do?
# exec git diff-index --check --cached $against --