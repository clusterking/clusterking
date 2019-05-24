#!/bin/bash

# ==============================================================================
# Run unit tests
# ==============================================================================

#read -p "Run unittests? [yY]" -n 1 -r
#echo    # (optional) move to a new line
#if [[ $REPLY =~ ^[Yy]$ ]]
#then
#    `nose2 `
#    if [[ $? = 0 ]]
#    then
#        echo "Unittests passed."
#    else
#        echo "Unittests did not pass."
#        exit 1
#    fi
#else
#    echo "Skipping unit tests"
#fi

if [[ ! -x "$(command -v pytest)" ]]; then
    >&2 echo "Please install pytest, see hooks/readme.md. Falling back to nose."
    if [[ ! -x "$(command -v nose2)" ]]; then
        >&2 echo "Please install pytest, see hooks/readme.md. "
    else
        nose2
    fi
else
    pytest -v --tb=line
fi
