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

# version is the source version, this is the release with which to generate
# qpy files with to load with the version under test
version=$1
parts=( ${version//./ } )
# qiskit_version is the version under test, We're testing that we can correctly
# read the qpy files generated with source version with this version.
qiskit_version=`./qiskit_venv/bin/python -c "import qiskit;print(qiskit.__version__)"`
qiskit_parts=( ${qiskit_version//./ } )


# If source version is less than 0.18 QPY didn't exist yet so exit fast
if [[ ${parts[0]} -eq 0 && ${parts[1]} -lt 18 ]] ; then
    exit 0
fi

# Exclude any non-rc pre-releases as they don't have stable API guarantees
if [[ $version == *"b"* || $version == *"a"* ]] ; then
    exit 0
fi

# If the source version is newer than the version under test exit fast because
# there is no QPY compatibility for loading a qpy file generated from a newer
# release with an older release of Qiskit.
if ! ./qiskit_venv/bin/python ./compare_versions.py "$version" "$qiskit_version" ; then
    exit 0
fi

if [[ ! -d qpy_$version ]] ; then
    echo "Building venv for qiskit-terra $version"
    python -m venv $version
    if [[ ${parts[0]} -eq 0 ]] ; then
        ./$version/bin/pip install -c qpy_test_constraints.txt "qiskit-terra==$version"
    else
        ./$version/bin/pip install -c qpy_test_constraints.txt "qiskit==$version"
    fi
    mkdir qpy_$version
    pushd qpy_$version
    echo "Generating qpy files with qiskit-terra $version"
    ../$version/bin/python ../test_qpy.py generate --version=$version
else
    echo "Using cached QPY files for $version"
    pushd qpy_$version
fi
echo "Loading qpy files from $version with dev qiskit-terra"
../qiskit_venv/bin/python ../test_qpy.py load --version=$version
popd
rm -rf ./$version
