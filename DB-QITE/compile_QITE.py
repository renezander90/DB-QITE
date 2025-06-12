"""
\********************************************************************************
* Copyright (c) 2025 the Qrisp authors
*
* This program and the accompanying materials are made available under the
* terms of the Eclipse Public License 2.0 which is available at
* http://www.eclipse.org/legal/epl-2.0.
*
* This Source Code may also be made available under the following Secondary
* Licenses when the conditions for such availability set forth in the Eclipse
* Public License, v. 2.0 are satisfied: GNU General Public License, version 2
* with the GNU Classpath Exception which is
* available at https://www.gnu.org/software/classpath/license.html.
*
* SPDX-License-Identifier: EPL-2.0 OR GPL-2.0 WITH Classpath-exception-2.0
********************************************************************************/
"""

from qrisp.qite import QITE
from qrisp import QuantumVariable


def compile_QITE(H, U_0, exp_H, s_values, steps, method='GC'):

    circuits = []

    N = H.find_minimal_qubit_amount()
    qv = QuantumVariable(N)
    U_0(qv)
    qc = qv.qs.compile()
    circuits.append(qc)

    for k in range(1,steps+1):

        # Perform k steps of QITE
        qv = QuantumVariable(H.find_minimal_qubit_amount())
        QITE(qv, U_0, exp_H, s_values, k, method=method)
        qc = qv.qs.compile()
        circuits.append(qc)

    # Collect data for circuits
    circuit_ops = {}
    circuit_qubits = {}
    circuit_depth = {}

    for k, qc in enumerate(circuits):
        tqc = qc.transpile(basis_gates=["cz","u"])
        circuit_ops[k] = tqc.count_ops()
        circuit_qubits[k] = tqc.num_qubits()
        circuit_depth[k] = tqc.depth()

    circuit_data = [circuit_ops, circuit_qubits, circuit_depth]
    result_dict = {'circuit_data':circuit_data}

    return result_dict
