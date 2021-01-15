import math
import numpy as np
import random

from helpers import get_hex_keys, get_binary_keys, generate_mitigator, count_ops
from helpers import clean_result, random_gated_circuit

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import spatial
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPRegressor

from qiskit import IBMQ, Aer
from qiskit.providers.ibmq import least_busy
from qiskit.compiler import transpile, assemble
from qiskit import execute
from qiskit.tools.visualization import plot_histogram
from qiskit.providers.aer import QasmSimulator
from qiskit.ignis.mitigation.measurement import complete_meas_cal



# generate_calibration_circuits uses PCA to identify good calibration circuits
# from the training set, along with binary calibration circuits.
def generate_calibration_circuits(noisy, simulator, max_depth, pc_cirq, plot=0):
    global shots
    num_qubits = len(noisy.properties().qubits)
    training, _ = complete_meas_cal(range(num_qubits))
    for i in range(1, max_depth):
        for _ in range(10 * num_qubits):
            training.append(random_gated_circuit(noisy, i, 0))
    
    experiments = transpile(training, backend=noisy)
    expected_all = execute(experiments, simulator, shots=shots).result()
    actual_all = execute(experiments, noisy, shots=shots).result()

    errors = []
    keys = get_hex_keys(num_qubits)
    for i in range(len(training)):
        expected = clean_result(expected_all.data(i)['counts'], keys, shots)
        actual = clean_result(actual_all.data(i)['counts'], keys, shots)
        errors.append([(expected[key] - actual[key]) ** 2 for key in keys])

    errors = np.array(errors)
    pca = PCA(n_components=math.ceil(2 ** num_qubits))
    pca.fit(errors)

    test_cirq_indexes = {}
    for i in range(num_qubits):
        get_proj = lambda e: np.dot(pca.components_[i], e) / np.linalg.norm(e)
        projs = [abs(get_proj(error)) for error in errors]
        for ind in np.argsort(projs)[-1 - pc_cirq:-1]:
            test_cirq_indexes[ind] = True
    
    if plot != 0:
        plot_errors(num_qubits, pca, pc_cirq, errors, test_cirq_indexes, plot)
    
    return [training[ind] for ind in test_cirq_indexes.keys()]

def plot_errors(num_qubits, pca, pc_cirq, errors, test_circuit_indexes, dim):
    if dim == 2:
        x = [np.dot(pca.components_[0], err) for err in errors]
        y = [np.dot(pca.components_[1], err) for err in errors]

        plt.scatter(x[2 ** num_qubits:], y[2 ** num_qubits:], s=5)
        plt.scatter(x[:2 ** num_qubits], y[:2 ** num_qubits], s=5, color='r')
        for ind in test_circuit_indexes.keys()[:pc_cirq]:
            plt.scatter(x[ind], y[ind], s=5, color='y')
        plt.show()
    elif dim == 3:
        x = [np.dot(pca.components_[0], err) for err in errors]
        y = [np.dot(pca.components_[1], err) for err in errors]
        z = [np.dot(pca.components_[2], err) for err in errors]

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x[:2 ** num_qubits], y[:2 ** num_qubits], z[:2 ** num_qubits], color='r')
        ax.scatter(x[2 ** num_qubits:], y[2 ** num_qubits:], z[2 ** num_qubits:])
        plt.show()
    else:
        raise Exception("Invalid number of dimensions")

def evaluate_circuits(num_qubits, circuits, calibration, noisy, simulator):
    global shots
    basis = noisy.configuration().basis_gates
    keys = [str(hex(i)) for i in range(2 ** num_qubits)]
    string_mod = lambda s: '0' * (7 - len(s)) + s[2:]
    new_keys = [string_mod(str(bin(int(h, 0)))) for h in keys]
    X_set, results, Y_set = [], [], []
    for circuit in circuits:
        circuit_group = calibration + [circuit]
        experiments = transpile(circuit_group, backend=noisy, optimization_level=3)
        circuits_output = execute(experiments, noisy, shots=shots).result()
        training_data = count_ops(experiments[-1], basis)

        # Gather actual circuit output
        out = circuits_output.data(len(circuit_group) - 1)['counts']
        clean_out = clean_result(out, keys, shots)
        result_dict = {}
        for i in range(len(keys)):
            result_dict[new_keys[i]] = clean_out[keys[i]]
        results.append(result_dict)

        for i in range(len(circuit_group)):
            data = circuits_output.data(i)['counts']
            cleaned = clean_result(data, keys, shots)
            training_data += [cleaned[key] for key in keys]
        
        X_set.append(training_data)
        expected = execute(experiments[-1], simulator, shots=shots).result()
        training_result = clean_result(expected.data(0)['counts'], keys, shots)
        Y_set.append([training_result[key] for key in keys])
    return X_set, Y_set, results

def build_model(training, holdout, calibration_circuits, noisy, simulator, num_qubits):
    global shots
    X_train, Y_train, _ = evaluate_circuits(num_qubits, training, calibration_circuits, noisy, simulator, shots)
    X_test, Y_test, results = evaluate_circuits(num_qubits, holdout, calibration_circuits, noisy, simulator, shots)

    clf = MLPRegressor(random_state=1, max_iter=300).fit(X_train, Y_train)
    mitigation = generate_mitigator(num_qubits, noisy, shots)

    net_err = 0
    mit_err = 0
    net_pred = clf.predict(X_test)

    keys = [str(hex(i)) for i in range(2 ** num_qubits)]
    string_mod = lambda s: '0' * (7 - len(s)) + s[2:]
    new_keys = [string_mod(str(bin(int(h, 0)))) for h in keys]

    for i in range(len(holdout)):
        mit_pred = mitigation.apply(results[i])
        mit_pred = np.array([mit_pred[nk] if nk in mit_pred else 0 for nk in new_keys])

        net_err += sum((net_pred[i] - Y_test[i]) ** 2)
        mit_err += sum((mit_pred - Y_test[i]) ** 2)

    print(net_err, mit_err)

def main():
    global shots
    shots = 1024
    provider = IBMQ.load_account()
    num_qubits = 5

    def backendFilter(x):
        nonlocal num_qubits
        a = x.configuration().n_qubits >= num_qubits
        b = x.configuration().simulator
        c = x.status().operational == True
        return a and not b and c

    qc = least_busy(provider.backends(filters=backendFilter))
    simulator = Aer.get_backend('qasm_simulator')
    noisy = QasmSimulator.from_backend(qc)

    circuits = []
    for i in range(1, 3):
        for _ in range(10 * num_qubits):
            circuits.append(random_gated_circuit(qc, i, 0))

    random.shuffle(circuits)
    training = circuits[:math.ceil(.8 * len(circuits))]
    holdout = circuits[math.ceil(.8 * len(circuits)):]

    calibration_circuits = generate_calibration_circuits(noisy, simulator, 3)
    # build_model(training, holdout, calibration_circuits, noisy, simulator, num_qubits, shots)

main()