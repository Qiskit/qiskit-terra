#!/bin/bash

# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
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

# Set fixed hash seed to ensure set orders are identical between saving and
# loading.
export PYTHONHASHSEED=$(python -S -c "import random; print(random.randint(1, 4294967295))")
echo "PYTHONHASHSEED=$PYTHONHASHSEED"

python -m venv qiskit_venv
qiskit_venv/bin/pip install -c ../../constraints.txt ../..

parallel bash ./process_version.sh ::: `git tag --sort=-creatordate`

# Test dev compatibility
dev_version=`qiskit_venv/bin/python -c 'import qiskit;print(qiskit.__version__)'`
qiskit_venv/bin/python test_qpy.py generate --version=$dev_version
qiskit_venv/bin/python test_qpy.py load --version=$dev_version
