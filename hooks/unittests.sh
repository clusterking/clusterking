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

nose2