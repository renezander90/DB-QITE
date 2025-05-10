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


def run_QITE(H, U_0, exp_H, s_values, steps, method='GC', use_statevectors=False):

    start = time.time()

    H_squared = H**2
    H_matrix = H.to_array()

    theta = sp.Symbol('theta')
    optimal_s = [theta]
    optimal_energies = []
    variances = []
    circuits = []
    runtimes = []
    statevectors = []

    N = H.find_minimal_qubit_amount()
    qv = QuantumVariable(N)
    U_0(qv)
    qc = qv.qs.compile()
    circuits.append(qc)

    if use_statevectors:
        E_0, _, V_0 = compute_moments(get_statevector(qc,N),H_matrix)
    else:
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
        if use_statevectors:
            energies = [compute_moments(get_statevector(qc,N,subs_dic={theta:s_}),H_matrix)[0] for s_ in s_values]
        else:
            energies = [H.get_measurement(qv,precision=0.01,subs_dic={theta:s_},precompiled_qc=qc,diagonalisation_method='commuting') for s_ in s_values]

        index = np.argmin(energies)
        s_min = s_values[index]

        optimal_s.insert(-1,s_min)
        optimal_energies.append(energies[index])
        if use_statevectors:
            variances.append(compute_moments(get_statevector(qc,N,subs_dic={theta:s_min}),H_matrix)[2])
        else:
            variances.append(H_squared.get_measurement(qv,precision=0.01,subs_dic={theta:s_min},precompiled_qc=qc,diagonalisation_method='commuting') - energies[index]**2)

        end = time.time()
        runtimes.append(end-start)


    evolution_times = [sum(optimal_s[i] for i in range(k)) for k in range(steps+1)]

    circuit_ops = {}
    circuit_qubits = {}
    circuit_depth = {}

    # Collect data for circuits and statevectors
    for k, qc in enumerate(circuits):
        # We need to bind the symbolic parameter theta to transpile to specific basis gates
        if k>0:
            qc = qc.bind_parameters(subs_dic={theta:optimal_s[k-1]})   
 
        tqc = qc.transpile(basis_gates=["cx","u"])
        circuit_ops[k] = tqc.count_ops()
        circuit_qubits[k]=tqc.num_qubits()
        circuit_depth[k]=tqc.depth()

        statevectors.append(get_statevector(qc,N))

    circuit_data = [circuit_ops, circuit_qubits, circuit_depth, circuit_qubits]

    result_dict = {'evolution_times':evolution_times,'optimal_energies':optimal_energies,'variances':variances,'circuit_data':circuit_data,'runtimes':runtimes,'statevectors':statevectors}

    return result_dict


def get_statevector(qc,n,subs_dic=None):
    if subs_dic is not None:
        bqc = qc.bind_parameters(subs_dic) 
    else:
        bqc = qc
    
    for i in range(bqc.num_qubits() - n):
        bqc.qubits.insert(0, bqc.qubits.pop(-1))

    statevector = bqc.statevector_array()[:2**n]
    statevector = statevector/np.linalg.norm(statevector)

    return statevector

def compute_moments(psi, H):
    psi = np.array([psi]).transpose()
    E = (psi.conj().T @ H @ psi)[0,0].real
    S = (psi.conj().T @ H @ H @ psi)[0,0].real
    return E, S, S - E**2