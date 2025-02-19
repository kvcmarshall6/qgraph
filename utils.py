from qutip import sigmaz, sigmap, qeye, tensor, Options
import settings
import numpy as np
import math
import numba
from tqdm.auto import tqdm, trange
from operator import itemgetter

settings.init()

if settings.qiskit:
    from quantum_routines_qiskit import (
        generate_empty_initial_state,
        generate_mixing_Ham,
        generate_Ham_from_graph,
        sesolve
    )
else:
    from quantum_routines import (
        generate_empty_initial_state,
        generate_mixing_Ham,
        generate_Ham_from_graph,
    )
    from qutip import sesolve


def generate_signal_fourier(G, rot_init=settings.rot_init,
                            N_sample=1000, hamiltonian='xy',
                            tf=100*math.pi):
    """
    Function to return the Fourier transform of the average number of
    excitation signal

    Arguments:
    ---------
    - G: networx.Graph, graph to analyze
    - rot_init: float, initial rotation
    - N_sample: int, number of timesteps to compute the evolution
    - hamiltonian: str 'xy' or 'ising', type of hamiltonian to simulate
    - tf: float, total time of evolution

    Returns:
    --------
    - plap_fft: numpy.Ndarray, shape (N_sample,) values of the fourier spectra
    - freq_normalized: numpy.Ndarray, shape (N_sample,) values of the
    fequencies
    """

    assert hamiltonian in ['ising', 'xy']
    N_nodes = G.number_of_nodes()
    H_evol = generate_Ham_from_graph(G, type_h=hamiltonian)

    rotation_angle_single_exc = rot_init/2.
    tlist = np.linspace(0, rotation_angle_single_exc, 200)

    psi_0 = generate_empty_initial_state(N_nodes)
    H_m = generate_mixing_Ham(N_nodes)

    result = sesolve(H_m, psi_0, tlist)
    final_state = result.states[-1]

    sz = sigmaz()
    si = qeye(2)
    sp = sigmap()
    sz_list = []
    sp_list = []

    for j in range(N_nodes):
        op_list = [si for _ in range(N_nodes)]
        op_list[j] = sz
        sz_list.append(tensor(op_list))
        op_list[j] = sp
        sp_list.append(tensor(op_list))

    tlist = np.linspace(0, tf, N_sample)

    observable = (-2*math.sin(2*rotation_angle_single_exc)
                  * sum(spj for spj in sp_list)
                  + math.cos(2*rotation_angle_single_exc)
                  * sum(szj for szj in sz_list))

    opts = Options()
    opts.store_states = True
    result = sesolve(H_evol, final_state, tlist,
                     e_ops=[observable], options=opts)

    full_signal = result.expect
    signal = full_signal[0].real
    signal_fft = np.fft.fft(signal)
    freq = np.fft.fftfreq(signal.shape[-1])
    freq_normalized = np.abs(freq * N_sample * 2) / (tf / np.pi)

    return signal_fft, freq_normalized


@numba.njit
def entropy(p):
    """
    Returns the entropy of a discrete distribution p

    Arguments:
    ---------
    - p: numpy.Ndarray dimension 1 non-negative floats summing to 1

    Returns:
    --------
    - float, value of the entropy
    """
    assert (p >= 0).all()
    assert abs(np.sum(p)-1) < 1e-6
    return -np.sum(p*np.log(p+1e-12))


@numba.njit
def jensen_shannon(hist1, hist2):
    '''
    Returns the Jensen Shannon divergence between two probabilities
    distribution represented as histograms.

    Arguments:
    ---------
    - hist1: tuple of numpy.ndarray (density, bins),
             len(bins) = len(density) + 1.
             The integral of the density wrt bins sums to 1.
    - hist2: same format.

    Returns:
    --------
    - float, value of the Jensen Shannon divergence.
    '''

    bins = np.sort(np.unique(np.array(list(hist1[1]) + list(hist2[1]))))
    masses1 = []
    masses2 = []

    for i, b in enumerate(bins[1::]):
        if b <= hist1[1][0]:
            masses1.append(0.)
        elif b > hist1[1][-1]:
            masses1.append(0.)
        else:
            j = 0
            while b > hist1[1][j]:
                j += 1
            masses1.append((b-bins[i]) * hist1[0][j-1])

        if b <= hist2[1][0]:
            masses2.append(0.)
        elif b > hist2[1][-1]:
            masses2.append(0.)
        else:
            j = 0
            while b > hist2[1][j]:
                j += 1
            masses2.append((b-bins[i]) * hist2[0][j-1])

    masses1 = np.array(masses1)
    masses2 = np.array(masses2)
    masses12 = (masses1+masses2)/2

    return entropy(masses12) - (entropy(masses1) + entropy(masses2))/2


# @ray.remote
def return_fourier_from_dataset(graph_list, rot_init=settings.rot_init):
    """
    Returns the fourier transform of evolution for a list of graphs for
    the hamiltonian ising and xy.

    Arguments:
    ---------
    - graph_list: list or numpy.Ndarray of networkx.Graph objects

    Returns:
    --------
    - fs_xy: numpy.Ndarray of shape (2, len(graph_list), 1000)
                        [0,i]: Fourier signal of graph i at 1000 points for
                               hamiltonian XY
                        [1,i]: frequencies associated to graph i at 1000 points
                        for hamiltonian XY

    - fs_is: same for the Ising hamiltonian

    """
    fs_xy = np.zeros((2, len(graph_list), 1000))
    fs_is = np.zeros((2, len(graph_list), 1000))

    for i, graph in enumerate(graph_list):
        fs_xy[0][i], fs_xy[1][i] = generate_signal_fourier(graph,
                                                    rot_init=rot_init,
                                                    N_sample=1000,
                                                    hamiltonian='xy')
        fs_is[0][i], fs_is[1][i] = generate_signal_fourier(graph,
                                                    rot_init=rot_init,
                                                    N_sample=1000,
                                                    hamiltonian='ising')

    return fs_xy, fs_is

def return_evolution(G, times, pulses, evol='xy'):
    """
    Returns the final state after the following evolution:
    - start with empty sate with as many qubits as vertices of G
    - uniform superposition of all states
    - alternating evolution of H_evol during times, and H_m during pulses

    Arguments:
    ---------
    - G: graph networkx.Graph objects
    - times: list of times to evolve following H_evol, list or np.ndarray
    - pulses: list of times to evolve following H_m, list or np.ndarray
                same length as times
    - evol: type of evolution for H_evol 'ising' or 'xy'

    Returns:
    --------
    - state: qutip.Qobj final state of evolution

    """
    assert evol in ['xy', 'ising']
    assert len(times) == len(pulses)

    N_nodes = G.number_of_nodes()
    H_evol = generate_Ham_from_graph(G, type_h=evol)
    H_m = generate_mixing_Ham(N_nodes)

    state = generate_empty_initial_state(N_nodes)

    if settings.qiskit: 
        res = sesolve(H_evol, H_m, state, pulses, [0, np.pi / 4])
    else:
        opts = {
            "store_states": True,
        }

        res = sesolve(H_m, state, [0, np.pi / 4], options=opts)
        res = res.states[-1]

        for i, theta in enumerate(pulses):
            if np.abs(times[i]) > 0:
                res = sesolve(H_evol, state, [0, times[i]], options=opts)
                res = res.states[-1]

            if np.abs(theta) > 0:
                res = sesolve(H_m, state, [0, theta], options=opts)
                res = res.states[-1]

    return res

def return_list_of_states(graphs_list,
    times, pulses, evol='xy', verbose=0):
    """
    Returns the list of states after evolution for each graph following
    return_evolution functions.

    Arguments:
    ---------
    - graphs_list: iterator of graph networkx.Graph objects
    - times: list of times to evolve following H_evol, list or np.ndarray
    - pulses: list of times to evolve following H_m, list or np.ndarray
                same length as times
    - evol: type of evolution for H_evol 'ising' or 'xy'
    - verbose: int, display the progression every verbose steps

    Returns:
    --------
    - all_states: list of qutip.Qobj final states of evolution,
                same lenght as graphs_list

    """
    all_states = []
    for G in tqdm(graphs_list, disable=verbose==0):
        all_states.append(return_evolution(G, times, pulses, evol))
    return all_states


def return_energy_distribution(
    graphs_list,
    all_states,
    observable_func=None,
    return_energies=False,
    verbose=0,
):
    """
    Returns all the discrete probability distributions of a diagonal
    observable on a list of states each one associated with a graph. The
    observable can be different for each state. The distribution is taken of
    all possible values of all observables.

    Arguments:
    ---------
    - graphs_list: iterator of graph networkx.Graph objects
    - all_states: list of states (Qutip Qobj or nested lists of arrays for Qiskit)
                  associated with graphs_list
    - observable_func: function(networkx.Graph):
                        returns diagonal observable (qutip.Qobj or Qiskit observable)
    - return_energies: boolean

    Returns:
    --------
    - all_e_masses: numpy.ndarray of shape (len(graphs_list), N_dim)
            all discrete probability distributions
    - e_values_unique: numpy.ndarray of shape (N_dim, )
            if return_energies, all energies

    """

    all_e_distrib = []
    all_e_values_unique = []

    for i, G in enumerate(tqdm(graphs_list, disable=verbose == 0)):
        if observable_func == None:
            observable = generate_Ham_from_graph(G, type_h="ising")
        else:
            observable = observable_func(G)
        if settings.qiskit:
            e_values = observable.to_matrix().diagonal().real
        else:
            e_values = observable.full().diagonal().real
        e_values_unique = np.unique(e_values)
        state = all_states[i]

        if settings.qiskit:
            # flatten nested states for Qiskit
            state = state.data

        e_distrib = np.zeros(len(e_values_unique))

        if settings.qiskit:
            # compute probabilities from the state vector
            probs = (
                np.abs(state) ** 2
            )
            assert len(probs) == len(e_values)

        for j, v in enumerate(e_values_unique):
            if settings.qiskit:
                mask = e_values == v
                assert len(mask) == len(probs)
                probabilities = probs[mask]
            else:
                probabilities = (np.abs(state.full()) ** 2)[e_values == v]
            # sum probabilities
            e_distrib[j] = np.sum(probabilities)

        all_e_distrib.append(e_distrib)
        all_e_values_unique.append(e_values_unique)

    e_values_unique = np.unique(np.concatenate(all_e_values_unique, axis=0))

    all_e_masses = []
    # all_e_distrib is list of probability distributions for each graph - each is pd of observing unique eigenvalues of O for a given graph
    # all_e_values_unique is list of arrays - each has unique eigenvalues of observable for a given graph
    # builds all_e_masses 2D array - each row corresponds to discrete pd over the unique eigenvalues of all observables (different for each graph)
    for e_distrib, e_values in zip(all_e_distrib, all_e_values_unique):
        masses = np.zeros_like(e_values_unique)
        for d, e in zip(e_distrib, e_values):
            masses[e_values_unique == e] = d
        all_e_masses.append(masses)

    all_e_masses = np.array(all_e_masses)

    if return_energies:
        return all_e_masses, e_values_unique
    return all_e_masses


def extend_energies(target_energies, energies, masses):
    """
    Extends masses array with columns of zeros for missing energies.

    Arguments:
    ---------
    - target_energies: numpy.ndarray of shape (N_dim, ) target energies
    - energies: numpy.ndarray of shape (N_dim_init, ) energies of distributions
    - masses: numpy.ndarray of shape (N, N_dim_init) discrete probability distributions

    Returns:
    --------
    - numpy.ndarray of shape (N, N_dim)
            all extended discrete probability distributions

    """
    energies = list(energies)
    N = masses.shape[0]
    res = np.zeros((N, len(target_energies)))
    for i, energy in enumerate(target_energies):
        if energy not in energies:
            res[:, i] = np.zeros((N, ))
        else:
            res[:, i] = masses[:, energies.index(energy)]
    return res


def merge_energies(e1, m1, e2, m2):
    """
    Merge the arrays of energy masses, filling with zeros the missing energies in each.
    N_dim is the size of the union of the energies from the two distributions.

    Arguments:
    ---------
    - e1: numpy.ndarray of shape (N_dim1, ) energies of first distributions
    - m1: numpy.ndarray of shape (N1, N_dim1) first discrete probability distributions
    - e2: numpy.ndarray of shape (N_dim2, ) energies of first distributions
    - m2: numpy.ndarray of shape (N2, N_dim2) first discrete probability distributions

    Returns:
    --------
    - numpy.ndarray of shape (N1, N_dim)
            all extended first discrete probability distributions
    - numpy.ndarray of shape (N2, N_dim)
            all extended second discrete probability distributions

    """
    e = sorted(list(set(e1) | set(e2)))
    return extend_energies(e, e1, m1), extend_energies(e, e2, m2)

def return_js_square_matrix(distributions, verbose=0):
    """
    Returns the Jensen-Shannon distance matrix of discrete
    distributions.

    Arguments:
    ---------
    - distributions: numpy.ndarray of shape (N_sample, N_dim)
        matrix of probability distribution represented on
        each row. Each row must sum to 1.

    Returns:
    --------
    - js_matrix: numpy.ndarray Jensen-Shannon distance matrix
            of shape (N_sample, N_sample)

    """
    js_matrix = np.zeros((len(distributions), len(distributions)))
    for i in range(len(distributions)):
        for j in range(i + 1):
            masses1 = distributions[i]
            masses2 = distributions[j]
            js = entropy((masses1+masses2)/2) -\
                entropy(masses1)/2 - entropy(masses2)/2
            js_matrix[i, j] = js
            js_matrix[j, i] = js
    return js_matrix

def return_js_matrix(distributions1, distributions2, verbose=0):
    """
    Returns the Jensen-Shannon distance matrix between discrete
    distributions.

    Arguments:
    ---------
    - distributions1: numpy.ndarray of shape (N_samples_1, N_dim)
        matrix of probability distribution represented on
        each row. Each row must sum to 1.
    - distributions2: numpy.ndarray of shape (N_samples_2, N_dim)
        matrix of probability distribution represented on
        each row. Each row must sum to 1.

    Returns:
    --------
    - js_matrix: numpy.ndarray Jensen-Shannon distance matrix
            of shape (N_sample, N_sample)

    """
    assert distributions1.shape[1] == distributions2.shape[1], \
        "Distributions must have matching dimensions. Consider using merge_energies"
    js_matrix = np.zeros((len(distributions1), len(distributions2)))
    for i in trange(len(distributions1), desc='dist1 loop', disable=verbose<=0):
        for j in trange(len(distributions2), desc='dist2 loop', disable=verbose<=1):
            masses1 = distributions1[i]
            masses2 = distributions2[j]
            js = entropy((masses1+masses2)/2) -\
                entropy(masses1)/2 - entropy(masses2)/2
            js_matrix[i, j] = js
    return js_matrix

class Memoizer:
    """ 
    Will store results of the provided observable on graphs to avoid recomputing.
    Storage is based on a key computed using get_key

    Attributes:
    -----------
    - observable: function(networkx.Graph):
                    return qtip.Qobj diagonal observable
    - get_key: function(networkx.Graph):
                    return a key used to identify the graph
    """
    def __init__(self, observable, get_key=None):
        self.graphs = {}
        self.observable = observable
        self.get_key = get_key if get_key is not None else Memoizer.edges_key
    
    @staticmethod
    def edges_unique_key(graph):
        """
        Key insensitive to how edges of the graph are returned 
        (order of edges and order of nodes in edges).
        Same result for [(a, b), (c, d)] and [(d, c), (a, b)]
        """
        edges = list(map(sorted, graph.edges))
        return tuple(map(tuple, sorted(edges, key=itemgetter(0,1))))

    @staticmethod
    def edges_key(graph):
        """ Simple key based on the edges list """
        return tuple(graph.edges())

    def get_observable(self, graph):
        """ 
        Gets observable on graph
        Uses memoization to speed up the process if graph has been seen before

        Arguments:
        ---------
        - graph: networkx.Graph to get observable on

        Returns:
        --------
        - qtip.Qobj, diagonal observable

        """
        key = self.get_key(graph)
        if key not in self.graphs:
            self.graphs[key] = self.observable(graph)
        return self.graphs[key]
