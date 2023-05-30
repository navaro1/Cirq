# Copyright 2023 The Cirq Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import List, Sequence, TYPE_CHECKING

from cirq.sim.simulation_state import SimulationState

if TYPE_CHECKING:
    import cirq


class ClassicalBasisState(cirq.qis.QuantumStateRepresentation):
    def __init__(self, initial_state: List[int]):
        self.basis = initial_state

    def copy(self, deep_copy_buffers: bool = True) -> 'ClassicalBasisState':
        return ClassicalBasisState(self.basis)

    def measure(self, axes: Sequence[int], seed: 'cirq.RANDOM_STATE_OR_SEED_LIKE' = None):
        return [self.basis[i] for i in axes]


class ClassicalSimulationState(SimulationState[ClassicalBasisState]):
    def __init__(self, initial_state, qubits, classical_data):
        state = ClassicalBasisState(
            cirq.big_endian_int_to_bits(initial_state, bit_count=len(qubits))
        )
        super().__init__(state=state, qubits=qubits, classical_data=classical_data)

    def _act_on_fallback_(self, action, qubits: Sequence[cirq.Qid], allow_decompose: bool = True):
        gate = action.gate if isinstance(action, cirq.Operation) else action
        if isinstance(gate, cirq.XPowGate):
            i = self.qubit_map[qubits[0]]
            self._state.basis[i] = int(gate.exponent + self._state.basis[i]) % qubits[0].dimension
            return True
