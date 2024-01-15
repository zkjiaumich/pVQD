from walsh_transform import *
import numpy as np
from scipy.fft import fft, ifft, fftfreq
from tqdm import tqdm
from sympy.combinatorics import GrayCode
import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8')
from scipy.interpolate import RegularGridInterpolator

from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister, transpile
from qiskit.tools.visualization import circuit_drawer
from qiskit.quantum_info import state_fidelity
from qiskit import Aer
from qiskit.tools.visualization import plot_histogram, plot_state_city
from qiskit.circuit.library import QFT
from qiskit.extensions import Initialize
from qiskit.quantum_info import Statevector

# Function that gives the partitioned sets by the most significant non-zero bit (MSB) from the number of qubits
def gray_partitions(n):
    # List the Gray code
    g = GrayCode(n)
    gray_list = list(g.generate_gray())[1:]

    # Create a dictionary collecting the lists
    partitions = [[] for _ in range(n)]

    # Figure out the MSB and arrange the partiton
    for entry in gray_list:
        index = entry.find('1')
        partitions[n - 1 - index].append(entry)
    return partitions


# Function that gives the position of the targeted part of the CNOT
def get_control(bitstring1, bitstring2, n):
    xor = ''.join([str(int(bit1) ^ int(bit2)) for bit1, bit2 in zip(bitstring1, bitstring2)])
    return n - 1 - xor.find('1')


# Function that implements the unitary diagonals
def unitary_circuit(f, n, dt, x_grid, terms_kept=None, verbose=True):
    circ = QuantumCircuit(n)
    a = wft(f, n, x_grid, verbose=verbose)
    a_kept = np.copy(a)
    if terms_kept is not None:
        sorted_indices = np.argsort(np.abs(a))[::-1]
        dropped_indices = sorted_indices[terms_kept:]
        a_kept[dropped_indices] = 0
    partitions = gray_partitions(n)
    for partition, target in zip(partitions, range(n)):
        if len(partition) == 1:
            index = eval('0b' + partition[0])
            theta = a_kept[index]
            if np.abs(theta) > 0:
                circ.rz(2*theta*dt, target)
            continue
        for i in range(len(partition)):
            index = eval('0b' + partition[i])
            theta = a_kept[index]
            control = get_control(partition[i - 1], partition[i], n)
            circ.cnot(control, target)
            if np.abs(theta) > 0:
                circ.rz(2*theta*dt, target)
    circ = transpile(circ, optimization_level=1)
    return circ


def kinetic(n_q, dx, dt, D):
    """Apply the kinetic term of a single iteration of the Zalka-Wiesner algorithm.
    Args:
        n_q: number of qubits that define the grid.
        d: limits of the grid, i.e., x is defined in [-d, d).
        dt: duration of each discrete time step.

    Returns:
        qc: quantum circuit right after this step.
    """
    qc = QuantumCircuit(n_q)
    N = 2**n_q

    p_vals = (2 * np.pi * np.fft.fftfreq(N, d=dx))[[2**k for k in range(n_q)]]
    p_sum = sum(p_vals)

    for j in range(n_q):
        alpha_j = -(dt*D) * p_vals[j] ** 2 - (dt*D) * p_vals[j] * (p_sum - p_vals[j])
        qc.rz(alpha_j, j)

    for j in range(n_q):
        for l in range(j + 1, n_q):
            gamma_jl = (dt*D) * p_vals[j] * p_vals[l]
            qc.cx(j, l)
            qc.rz(gamma_jl, l)
            qc.cx(j, l)
    return qc


def walsh_quantum_trotter(potential, initial_wave_function, n, L, dt, D=1/2, terms_kept=None, verbose=True, gate_count_only=False):
    N = 2**n
    dx = 2*L/N
    x_grid = np.arange(-L, L, dx)
#     t_grid = np.arange(0, T + dt/2, dt)


    # Initializing the quantum circuit
    qc = QuantumCircuit(n)

    # Initialize the initial wave function
#     desired_vector = initial_wave_function(x_grid)
#     if not gate_count_only: qc.prepare_state(state=desired_vector)

    potential_step = unitary_circuit(potential, n, dt, x_grid, terms_kept=terms_kept, verbose=False)
    kinetic_step = kinetic(n, dx, dt, D)

    iqft = QFT(num_qubits=n, inverse=True).decompose().to_gate()
    qft = QFT(num_qubits=n).decompose().to_gate()

#     if gate_count_only: out = False
#     else:
#         out = True
#         states = [Statevector.from_instruction(qc)]

#     if verbose: progress = tqdm(total=K, desc='working on time evolution')
#     for i in range(K):
        # Kinetic Step
    qc.append(qft, qargs=[i for i in range(n)])
    qc.append(kinetic_step, qargs=[i for i in range(n)])
    qc.append(iqft, qargs=[i for i in range(n)])

    # Potential Step
    qc.append(potential_step, qargs=[i for i in range(n)][::-1])

#         if out:
#             states.append(Statevector.from_instruction(qc))
#         if verbose: progress.update(1)
#     if verbose: progress.close()

#     if out: states = np.array(states, dtype='complex')
#     qc = qc.decompose()
#     if gate_count_only: return qc.count_ops()
#     return states, t_grid, x_grid
    return qc


def walsh_quantum_strang(potential, initial_wave_function, n, L, dt, D=1/2, terms_kept=None, verbose=True, gate_count_only=False):
    N = 2**n
    dx = 2*L/N
    x_grid = np.arange(-L, L - dx/2, dx)
#     t_grid = np.arange(0, T + dt/2, dt)

    # Initializing the quantum circuit
    qc = QuantumCircuit(n)

    # Initialize the initial wave function
#     desired_vector = initial_wave_function(x_grid)
#     if not gate_count_only: qc.prepare_state(state=desired_vector)

    potential_step = unitary_circuit(potential, n, dt, x_grid, terms_kept=terms_kept, verbose=False)
    half_potential_step = unitary_circuit(potential, n, dt/2, x_grid, terms_kept=terms_kept, verbose=False)
    kinetic_step = kinetic(n, dx, dt, D)

    a = wft(potential, n, x_grid, verbose=False)
    potential_walsh = iwft(a, n, terms_kept=terms_kept, verbose=False)

    iqft = QFT(num_qubits=n, inverse=True).decompose().to_gate()
    qft = QFT(num_qubits=n).decompose().to_gate()

#     if gate_count_only: out = False
#     else:
#         out = True
#         states = [Statevector.from_instruction(qc)]

    qc.append(half_potential_step, qargs=[i for i in range(n)][::-1])
    # propagate potential half step to start
#     if verbose: progress = tqdm(total=K, desc='working on time evolution')

    # Propagate kinetic
    qc.append(qft, qargs=[i for i in range(n)])
#     qc.append(kinetic_step, qargs=[i for i in range(n)])
#     qc.append(iqft, qargs=[i for i in range(n)])

    # Propagate potential
#     qc.append(potential_step, qargs=[i for i in range(n)][::-1])
#     if out:
#         states.append(np.array(Statevector.from_instruction(qc))*np.exp(1j*potential_walsh*dt/2))
#     if verbose: progress.update(1)

    # Propagate kinetic
#     qc.append(qft, qargs=[i for i in range(n)])
    qc.append(kinetic_step, qargs=[i for i in range(n)])
    qc.append(iqft, qargs=[i for i in range(n)])

    # Ending potential half step
    qc.append(half_potential_step, qargs=[i for i in range(n)][::-1])

#     if out:
#         states.append(Statevector.from_instruction(qc))
#     if verbose: progress.update(1)
#     if verbose: progress.close()

#     if out: states = np.array(states, dtype='complex')
#     qc = qc.decompose()
#     if gate_count_only: return qc.count_ops()
#     return states, t_grid, x_grid
    return qc

def walsh_quantum_4(potential, initial_wave_function, n, L, dt, D=1/2, terms_kept=None, verbose=True, gate_count_only=False):
    N = 2**n
    dx = 2*L/N
    x_grid = np.arange(-L, L - dx/2, dx)
#     t_grid = np.arange(0, T + dt/2, dt)
    s = 1/(4-4**(1/3))

    # Initializing the quantum circuit
    qc = QuantumCircuit(n)

    # Initialize the initial wave function
#     desired_vector = initial_wave_function(x_grid)
#     if not gate_count_only: qc.prepare_state(state=desired_vector)

    potential_step_s = unitary_circuit(potential, n, s*dt, x_grid, terms_kept=terms_kept, verbose=False)
    half_potential_step_s = unitary_circuit(potential, n, s*dt/2, x_grid, terms_kept=terms_kept, verbose=False)
    interm_potential_step_s = unitary_circuit(potential, n, (1-3*s)*dt/2, x_grid, terms_kept=terms_kept, verbose=False)
    kinetic_step_s = kinetic(n, dx, s*dt, D)
    interm_kinetic_step_s = kinetic(n, dx, (1-4*s)*dt, D)

    a = wft(potential, n, x_grid, verbose=False)
    potential_walsh = iwft(a, n, terms_kept=terms_kept, verbose=False)

    iqft = QFT(num_qubits=n, inverse=True).decompose().to_gate()
    qft = QFT(num_qubits=n).decompose().to_gate()

#     if gate_count_only: out = False
#     else:
#         out = True
#         states = [Statevector.from_instruction(qc)]

    qc.append(half_potential_step_s, qargs=[i for i in range(n)][::-1])
    # propagate potential half step to start
#     if verbose: progress = tqdm(total=K, desc='working on time evolution')

    # Propagate kinetic
    qc.append(qft, qargs=[i for i in range(n)])
    qc.append(kinetic_step_s, qargs=[i for i in range(n)])
    qc.append(iqft, qargs=[i for i in range(n)])

    # Propagate potential
    qc.append(potential_step_s, qargs=[i for i in range(n)][::-1])

    # Propagate kinetic
    qc.append(qft, qargs=[i for i in range(n)])
    qc.append(kinetic_step_s, qargs=[i for i in range(n)])
    qc.append(iqft, qargs=[i for i in range(n)])

    # Propagate potential
    qc.append(interm_potential_step_s, qargs=[i for i in range(n)][::-1])

    # Propagate kinetic
    qc.append(qft, qargs=[i for i in range(n)])
    qc.append(interm_kinetic_step_s, qargs=[i for i in range(n)])
    qc.append(iqft, qargs=[i for i in range(n)])

    # Propagate potential
    qc.append(interm_potential_step_s, qargs=[i for i in range(n)][::-1])

    # Propagate kinetic
    qc.append(qft, qargs=[i for i in range(n)])
    qc.append(kinetic_step_s, qargs=[i for i in range(n)])
    qc.append(iqft, qargs=[i for i in range(n)])

    # Propagate potential
    qc.append(potential_step_s, qargs=[i for i in range(n)][::-1])

    # Propagate kinetic
    qc.append(qft, qargs=[i for i in range(n)])
    qc.append(kinetic_step_s, qargs=[i for i in range(n)])
    qc.append(iqft, qargs=[i for i in range(n)])

#     # Propagate potential
#     qc.append(potential_step_s, qargs=[i for i in range(n)][::-1])

#     # Append the states
# #     if out:
# #         states.append(np.array(Statevector.from_instruction(qc))*np.exp(1j*potential_walsh*s*dt/2))
# #     if verbose: progress.update(1)

#     # Propagate kinetic
#     qc.append(qft, qargs=[i for i in range(n)])
#     qc.append(kinetic_step_s, qargs=[i for i in range(n)])
#     qc.append(iqft, qargs=[i for i in range(n)])

#     # Propagate potential
#     qc.append(potential_step_s, qargs=[i for i in range(n)][::-1])

#     # Propagate kinetic
#     qc.append(qft, qargs=[i for i in range(n)])
#     qc.append(kinetic_step_s, qargs=[i for i in range(n)])
#     qc.append(iqft, qargs=[i for i in range(n)])

#     # Propagate potential
#     qc.append(interm_potential_step_s, qargs=[i for i in range(n)][::-1])

#     # Propagate kinetic
#     qc.append(qft, qargs=[i for i in range(n)])
#     qc.append(interm_kinetic_step_s, qargs=[i for i in range(n)])
#     qc.append(iqft, qargs=[i for i in range(n)])

#     # Propagate potential
#     qc.append(interm_potential_step_s, qargs=[i for i in range(n)][::-1])

#     # Propagate kinetic
#     qc.append(qft, qargs=[i for i in range(n)])
#     qc.append(kinetic_step_s, qargs=[i for i in range(n)])
#     qc.append(iqft, qargs=[i for i in range(n)])

#     # Propagate potential
#     qc.append(potential_step_s, qargs=[i for i in range(n)][::-1])

#     # Propagate kinetic
#     qc.append(qft, qargs=[i for i in range(n)])
#     qc.append(kinetic_step_s, qargs=[i for i in range(n)])
#     qc.append(iqft, qargs=[i for i in range(n)])

    # Propagate potential
    qc.append(half_potential_step_s, qargs=[i for i in range(n)][::-1])

    # Append the states
#     if out:
#         states.append(np.array(Statevector.from_instruction(qc)))
#     if verbose: progress.update(1)
#     if verbose: progress.close()

#     if out: states = np.array(states, dtype='complex')
#     qc = qc.decompose()
#     if gate_count_only: return qc.count_ops()
    return qc
 



