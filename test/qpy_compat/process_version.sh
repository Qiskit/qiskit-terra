#!/bin/bash

# This code is part of Qiskit.
#
# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
set -e
set -x

function usage {
    echo "usage: ${BASH_SOURCE[0]} -p /path/to/qiskit/python <package> <version>" 1>&2
    exit 1
}

python="python"
while getopts "p:" opt; do 
    case "$opt" in
        p)
            python="$OPTARG"
            ;;
        *)
            usage
            ;;
    esac
done
shift "$((OPTIND-1))"
if [[ $# != 2 ]]; then
    usage
fi

# `package` is the name of the Python distribution to install (qiskit or qiskit-terra). `version` is
# the source version: the release with which to generate qpy files with to load with the version
# under test.
package="$1"
version="$2"

our_dir="$(realpath -- "$(dirname -- "${BASH_SOURCE[0]}")")"
cache_dir="$(pwd -P)/qpy_cache/$version"
venv_dir="$(pwd -P)/venvs/$package-$version"

if [[ ! -d $cache_dir ]] ; then
    echo "Building venv for $package==$version"
    "$python" -m venv "$venv_dir"
    "$venv_dir/bin/pip" install -c "${our_dir}/qpy_test_constraints.txt" "${package}==${version}"
    mkdir -p "$cache_dir"
    pushd "$cache_dir"
    echo "Generating QPY files with $package==$version"
    "$venv_dir/bin/python" "${our_dir}/test_qpy.py" generate --version="$version"
else
    echo "Using cached QPY files for $version"
    pushd "${cache_dir}"
fi
echo "Loading qpy files from $version with dev Qiskit"
"$python" "${our_dir}/test_qpy.py" load --version="$version"
popd
rm -rf "$venv_dir"
