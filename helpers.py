import math
import random

from qiskit import QuantumCircuit
from qiskit.compiler import transpile, assemble
from qiskit.ignis.mitigation.measurement import complete_meas_cal
from qiskit.ignis.mitigation.measurement import CompleteMeasFitter

# get_hex_keys is a helper function that returns strings of hexadecimal values
# for each measurable state given the number of qubits. 
def get_hex_keys(num_qubits):
    return [str(hex(i)) for i in range(2 ** num_qubits)]

# get_binary_keys is a helper function that returns strings of binary values for
# each measurable state given the number of qubits. 
def get_binary_keys(num_qubits):
    keys = [str(hex(i)) for i in range(2 ** num_qubits)]
    string_mod = lambda s: '0' * (7 - len(s)) + s[2:]
    return [string_mod(str(bin(int(h, 0)))) for h in keys]

# generate_mitigator is a helper function that takes a noisy quantum computer,
# number of qubits and an optional number of shots parameter. It runs naive
# calibration circuits on that quantum computer and returns the results of those
# circuits along with an error mitigation filter.
def generate_mitigator(num_qubits, noisy, shots=1024):
    meas_calibs, state_labels = complete_meas_cal(range(num_qubits))
    experiments = transpile(meas_calibs, backend=noisy, optimization_level=3)
    job = noisy.run(assemble(experiments, shots=shots))
    cal_results = job.result()
    meas_fitter = CompleteMeasFitter(cal_results, state_labels)
    return job, meas_fitter.filter

# count_ops is a helper function that takes a circuit and a gateset to scan for.
# It returns counts of each gate on each qubit.
def count_ops(circuit, keys):
    counts = [{} for _ in range(circuit.num_qubits)]
    for instr, qubits, _ in circuit:
        for qubit in qubits:
            if instr.name in counts[qubit.index]:
                counts[qubit.index][instr.name] += 1
            else:
                counts[qubit.index][instr.name] = 1
    for key in keys:
        for count in counts:
            if key not in count:
                count[key] = 0

    x_data = []
    for count in counts:
        for key in keys:
            x_data.append(count[key])
    return x_data

# clean_result is a helper function. It takes a dictionary of raw data and the
# keys to that dictionary, along with a shot number. It cleans the dictionary
# by:
#   1) Adding a key value pair of zero when a key is not present
#   2) Normalizing by the number of shots
#   3) Optionally replacing the keys of the dictionary with a new key set
# It returns the cleaned data set.
def clean_result(data, keys, shots, new_keys=[]):
    resp = {}
    if len(new_keys) == 0:
        for key in keys:
            if key not in data:
                resp[key] = 0
            else:
                resp[key] = data[key] / shots
    elif len(keys) != len(new_keys):
        raise Exception("Old keys and new keys must be one to one")
    else:
        for i in range(len(keys)):
            key, new_key = keys[i], new_keys[i]
            if key not in data:
                resp[new_key] = 0
            else:
                resp[new_key] = data[key] / shots
    return resp



# random_gated_circuit returns a random circuit at the provided depth. The
# circuit is created in a way that it can be run natively (without transpiling).
def random_gated_circuit(quantumBackend, depth, seed):
    random.seed(seed)
    circuit = QuantumCircuit(len(quantumBackend.properties().qubits))
    gateSet = quantumBackend.properties().gates
    while circuit.depth() < depth:
        gate = random.choice(gateSet)
        try:
            getattr(circuit, gate.gate)(*gate.qubits)
        except:
            # Some gates require an angular parameter, we generate a random one
            # between 0 and 2 pi.
            param = random.random() * 2 * math.pi
            getattr(circuit, gate.gate)(param, *gate.qubits)

    circuit.measure_all()
    return circuit

