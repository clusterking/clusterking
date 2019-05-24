#!/usr/bin/env bash

# Run black code formatter

# https://stackoverflow.com/questions/59895/
thisDir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null && pwd )"

sourceDir="${thisDir}/.."

if [[ ! -x "$(command -v black)" ]]; then
    >&2 echo "Please install black code formatter, see hooks/readme.md"
else
    find "${sourceDir}" -name "*.py" | xargs black -l 80 -t py34
fi
