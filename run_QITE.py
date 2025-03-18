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

import numpy as np
import sympy as sp
import time
from qrisp.qite import QITE
from qrisp import QuantumVariable


def run_QITE(H, U_0, exp_H, s_values, steps, method='GC'):

    start = time.time()

    H_squared = H**2

    theta = sp.Symbol('theta')
    optimal_s = [theta]
    optimal_energies = []
    variances = []
    circuits = []
    runtimes = []

    qv = QuantumVariable(H.find_minimal_qubit_amount())
    U_0(qv)
    qc = qv.qs.compile()
    circuits.append(qc)

    E_0 = H.get_measurement(qv,precision=0.01,precompiled_qc=qc,diagonalisation_method='commuting')
    V_0 = H_squared.get_measurement(qv,precision=0.01,precompiled_qc=qc,diagonalisation_method='commuting') - E_0**2
    optimal_energies.append(E_0)
    variances.append(V_0)

    for k in range(1,steps+1):

        # Perform k steps of QITE
        qv = QuantumVariable(H.find_minimal_qubit_amount())
        QITE(qv, U_0, exp_H, optimal_s, k, method=method)
        qc = qv.qs.compile()
        circuits.append(qc)

        # Find optimal evolution time 
        # Use "precompliled_qc" keyword argument to avoid repeated compilation of the QITE circuit
        energies = [H.get_measurement(qv,precision=0.01,subs_dic={theta:s_},precompiled_qc=qc,diagonalisation_method='commuting') for s_ in s_values]
        index = np.argmin(energies)
        s_min = s_values[index]

        optimal_s.insert(-1,s_min)
        optimal_energies.append(energies[index])
        variances.append(H_squared.get_measurement(qv,precision=0.01,subs_dic={theta:s_min},precompiled_qc=qc,diagonalisation_method='commuting') - energies[index]**2)

        end = time.time()
        runtimes.append(end-start)


    evolution_times = [sum(optimal_s[i] for i in range(k)) for k in range(steps+1)]

    circuit_ops = {}
    circuit_qubits = {}
    circuit_depth = {}

    # Collect data for circuits
    for k, qc in enumerate(circuits):
        # We need to bind the symbolic parameter theta to transpile to specific basis gates
        tqc = qc.bind_parameters(subs_dic={theta:1}).transpile(basis_gates=["cx","u"])
        circuit_ops[k] = tqc.count_ops()
        circuit_qubits[k]=tqc.num_qubits()
        circuit_depth[k]=tqc.depth()

    circuit_data = [circuit_ops, circuit_qubits, circuit_depth, circuit_qubits]

    result_dict = {'evolution_times':evolution_times,'optimal_energies':optimal_energies,'variances':variances,'circuit_data':circuit_data,'runtimes':runtimes}

    return result_dict
