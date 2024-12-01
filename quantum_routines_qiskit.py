from qiskit import QuantumCircuit
from qiskit.circuit.library import QAOAAnsatz
from qiskit.quantum_info import SparsePauliOp, Statevector
import numpy as np
import settings


def generate_mixing_Ham(N):
    """Build the mixing Hamiltonian
    Parameters
    ----------
    N: number of nodes
    coords: list of lists
        coordinates of particles
    Returns
    -------
    h_m: SparsePauliOp
        Mixing Hamiltonian as a summation of Pauli terms, with possible non-zero detuning
    """
    pauli_terms = []
    coefficients = []

    # laser field
    for j in range(N):
        pauli_string_y = ["I"] * N
        # apply Y on the j-th qubit
        pauli_string_y[j] = "Y"
        pauli_terms.append("".join(pauli_string_y))
        coefficients.append(-settings.omega)

        pauli_string_z = ["I"] * N
        # apply Z on the j-th qubit, if delta is non-zero
        pauli_string_z[j] = "Z" 
        coefficients.append(
            0. * settings.delta
        )
        pauli_terms.append("".join(pauli_string_z))

    # create h_m from the terms and coefficients
    h_m = SparsePauliOp.from_list(list(zip(pauli_terms, coefficients)))

    # # TODO: temp code to maintain jupyter workflow
    # matrix_representation = h_m.to_matrix()
    # # reshape to match QuTiP dims
    # reshaped_matrix = matrix_representation.reshape(
    #     (2**N, 2**N)
    # )
    # qutip_h_m = qutip.Qobj(reshaped_matrix, dims=[[2] * N, [2] * N])
    return h_m


def generate_Ham_from_graph(graph, type_h='xy', process_edge=None):
    """Given a connectivity graph, build the Hamiltonian, Ising or XY.
    Parameters
    ----------
    graph: networkx.Graph(), nodes numeroted from 0 to N_nodes
    type_h: str, type of hamiltonian 'xy' or 'ising'
    process_edge: funciton, function to convert the edge attribute into a
    numerical value, add weight to the hamiltonian

    Returns
    -------
    H: SparsePauliOp
        The Hamiltonian for the configuration as a SparsePauliOp.
    """
    assert type_h in ["ising", "xy"]

    N = graph.number_of_nodes()
    pauli_terms = []
    coefficients = []

    # define ising or XY Hamiltonian terms based on edges of graph
    for edge in graph.edges.data():
        node1, node2, edge_attr = edge
        edge_weight = 1

        if len(edge[2]) > 0:
            if process_edge is not None:
                edge_weight = process_edge(edge[2]["attr"])

        if type_h == 'ising':
            z_pauli = ["I"] * N
            z_pauli[node1] = "Z"
            z_pauli[node2] = "Z"
            pauli_terms.append("".join(z_pauli))
            coefficients.append(edge_weight)

        elif type_h == 'xy':
            x_pauli = ["I"] * N
            x_pauli[node1] = "X"
            x_pauli[node2] = "X"
            pauli_terms.append("".join(x_pauli))
            coefficients.append(edge_weight / 2)
            y_pauli = ["I"] * N
            y_pauli[node1] = "Y"
            y_pauli[node2] = "Y"
            pauli_terms.append("".join(y_pauli))
            coefficients.append(edge_weight / 2)

    H = SparsePauliOp.from_list(list(zip(pauli_terms, coefficients)))

    # # TODO: temp code to maintain jupyter workflow
    # matrix_representation = H.to_matrix()
    # # reshape to match QuTiP dims
    # reshaped_matrix = matrix_representation.reshape((2**N, 2**N))
    # qutip_H = qutip.Qobj(reshaped_matrix, dims=[[2] * N, [2] * N])
    return H


def reverse_statevector(statevector):
    """
    Reverses the binary order of a statevector for comparison.
    Needed because qiskit reverses the direction of the qubit register.
    """
    n = int(np.log2(len(statevector)))
    indices = np.arange(len(statevector))
    reversed_indices = np.array(
        [int(bin(i)[2:].zfill(n)[::-1], 2) for i in indices]
    )
    return statevector[reversed_indices]


def generate_empty_initial_state(N):
    """Generates the empty initial wavefunction
    Parameters
    ----------
    N: number of nodes (qubits)
    Returns
    -------
    psi_0: QuantumCircuit
        Initial wavefunction i.e. quantum circuit with all qubits in the |1⟩ state
    """
    qc = QuantumCircuit(N)
    for qubit in range(N):
        qc.x(qubit)

    # # TODO: temp code to maintain jupyter workflow
    # statevector = Statevector(qc).data
    # # reverse the order the statevector qubit register is read
    # statevector = reverse_statevector(statevector)
    # qutip_state = qutip.Qobj(
    #     np.array(statevector).reshape(-1, 1),
    #     dims=[[2] * qc.num_qubits, [1] * qc.num_qubits],
    # )
    return qc


def sesolve(H_evol, H_m, psi0, pulses, times):
    """
    Perform time evolution of a state vector using Qiskit quantum circuits.

    Arguments:
    ----------
    H_evol : Cost Hamiltonian: SparsePauliOp
    H_m : Mixing Hamiltonian: SparsePauliOp
    psi0 : Initial quantum state: QuantumCircuit
    pulses : Angles for mixing Hamiltonian evolution: list of floats
    times : Times for cost Hamiltonian evolution: list of float

    Returns:
    --------
    evolved_states : Final evolved quantum state: Statevector
    """

    reps = len(pulses)
    circuit = QAOAAnsatz(
        cost_operator=H_evol,
        reps=reps,
        initial_state=psi0.copy(),
        mixer_operator=H_m,
    )

    # flatten and bind parameters
    param_values = []
    for time, theta in zip(times, pulses):
        # parameters for cost Hamiltonian
        param_values.append(time)
        # parameters for mixing Hamiltonian
        param_values.append(theta)

    # bind the parameters to the ansatz
    bound_circuit = circuit.assign_parameters(param_values)

    # get statevector for the circuit
    evolved_states = Statevector(bound_circuit)
    return evolved_states
