#!/usr/bin/env python3
# This code is part of Qiskit.
#
# (C) Copyright IBM 2024
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Utility script to verify that all images have alt text"""

from pathlib import Path
import multiprocessing
import sys
import glob

# List of allowlist files that the checker will not verify
ALLOWLIST_MISSING_ALT_TEXT = [
    "qiskit/primitives/statevector_estimator.py",
    "qiskit/pulse/builder.py",
    "qiskit/pulse/library/symbolic_pulses.py",
    "qiskit/transpiler/layout.py",
    "qiskit/transpiler/__init__.py",
    "qiskit/transpiler/passes/utils/filter_op_nodes.py",
    "qiskit/transpiler/passes/utils/remove_barriers.py",
    "qiskit/transpiler/passes/scheduling/dynamical_decoupling.py",
    "qiskit/transpiler/passes/scheduling/padding/dynamical_decoupling.py",
    "qiskit/transpiler/passes/routing/star_prerouting.py",
    "qiskit/providers/fake_provider/__init__.py",
    "qiskit/quantum_info/states/statevector.py",
    "qiskit/qasm3/__init__.py",
    "qiskit/converters/dag_to_circuit.py",
    "qiskit/circuit/controlledgate.py",
    "qiskit/circuit/operation.py",
    "qiskit/circuit/quantumcircuit.py",
    "qiskit/circuit/__init__.py",
    "qiskit/circuit/parameter.py",
    "qiskit/circuit/random/utils.py",
    "qiskit/circuit/library/overlap.py",
    "qiskit/circuit/library/iqp.py",
    "qiskit/circuit/library/graph_state.py",
    "qiskit/circuit/library/phase_estimation.py",
    "qiskit/circuit/library/grover_operator.py",
    "qiskit/circuit/library/quantum_volume.py",
    "qiskit/circuit/library/hidden_linear_function.py",
    "qiskit/circuit/library/fourier_checking.py",
    "qiskit/circuit/library/__init__.py",
    "qiskit/circuit/library/boolean_logic/quantum_xor.py",
    "qiskit/circuit/library/boolean_logic/quantum_and.py",
    "qiskit/circuit/library/boolean_logic/inner_product.py",
    "qiskit/circuit/library/boolean_logic/quantum_or.py",
    "qiskit/circuit/library/basis_change/qft.py",
    "qiskit/circuit/library/generalized_gates/gms.py",
    "qiskit/circuit/library/generalized_gates/permutation.py",
    "qiskit/circuit/library/generalized_gates/gr.py",
    "qiskit/circuit/library/generalized_gates/mcmt.py",
    "qiskit/circuit/library/arithmetic/piecewise_chebyshev.py",
    "qiskit/circuit/library/n_local/real_amplitudes.py",
    "qiskit/circuit/library/n_local/evolved_operator_ansatz.py",
    "qiskit/circuit/library/n_local/n_local.py",
    "qiskit/circuit/library/n_local/pauli_two_design.py",
    "qiskit/circuit/library/n_local/qaoa_ansatz.py",
    "qiskit/circuit/library/n_local/excitation_preserving.py",
    "qiskit/circuit/library/n_local/efficient_su2.py",
    "qiskit/synthesis/arithmetic/multipliers/rg_qft_multiplier.py",
    "qiskit/synthesis/arithmetic/multipliers/hrs_cumulative_multiplier.py",
    "qiskit/visualization/dag_visualization.py",
    "qiskit/visualization/gate_map.py",
    "qiskit/visualization/state_visualization.py",
    "qiskit/visualization/counts_visualization.py",
    "qiskit/visualization/__init__.py",
    "qiskit/visualization/timeline/interface.py",
    "qiskit/visualization/circuit/circuit_visualization.py",
    "qiskit/visualization/pulse_v2/interface.py",
]


def is_image(line: str) -> bool:
    """Determine if a line is an image"""
    return line.strip().startswith((".. image:", ".. plot:"))


def is_option(line: str) -> bool:
    """Determine if a line is an option"""
    return line.strip().startswith(":")


def is_valid_image(options: list[str]) -> bool:
    """Validate one single image"""
    alt_exists = any(option.strip().startswith(":alt:") for option in options)
    nofigs_exists = any(option.strip().startswith(":nofigs:") for option in options)

    # Only `.. plot::`` directives without the `:nofigs:` option are required to have alt text.
    # Meanwhile, all `.. image::` directives need alt text and they don't have a `:nofigs:` option.
    return alt_exists or nofigs_exists


def validate_image(file_path: str) -> tuple[str, list[str]]:
    """Validate all the images of a single file"""

    if file_path in ALLOWLIST_MISSING_ALT_TEXT:
        return [file_path, []]

    invalid_images: list[str] = []

    lines = Path(file_path).read_text().splitlines()

    image_found = False
    options: list[str] = []

    for line_index, line in enumerate(lines):
        if image_found:
            if is_option(line):
                options.append(line)
                continue

            # Else, the prior image_found has no more options so we should determine if it was valid.
            #
            # Note that, either way, we do not early exit out of the loop iteration because this `line`
            # might be the start of a new image.
            if not is_valid_image(options):
                image_line = line_index - len(options)
                invalid_images.append(
                    f"- Error in line {image_line}: {lines[image_line-1].strip()}"
                )

        image_found = is_image(line)
        options = []

    return (file_path, invalid_images)


def _main() -> None:
    files = glob.glob("qiskit/**/*.py", recursive=True)

    with multiprocessing.Pool() as pool:
        results = pool.map(validate_image, files)

    failed_files = {file: image_errors for file, image_errors in results if image_errors}

    if not failed_files:
        print("✅ All images have alt text")
        sys.exit(0)

    print("💔 Some images are missing the alt text", file=sys.stderr)

    for file, image_errors in failed_files.items():
        print(f"\nErrors found in {file}:", file=sys.stderr)

        for image_error in image_errors:
            print(image_error, file=sys.stderr)

    print(
        "\nAlt text is crucial for making documentation accessible to all users.",
        "It should serve the same purpose as the images on the page,",
        "conveying the same meaning rather than describing visual characteristics.",
        "When an image contains words that are important to understanding the content,",
        "the alt text should include those words as well.",
        file=sys.stderr,
    )

    sys.exit(1)


if __name__ == "__main__":
    _main()
