#!/usr/bin/env bash

set -e

thisDir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
sourceDir="${thisDir}/.."

rm -rf "${sourceDir}/*.egg-info"
rm -rf "${sourceDir}/build/**"
