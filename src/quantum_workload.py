# Qiskit Imports
import networkx as nx
from collections import defaultdict
from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag
from qiskit.dagcircuit import DAGCircuit

# ==========================================
# 1. QASM Workload Parser & Analyzer
# ==========================================
class QasmWorkload:
    def __init__(self, qc: QuantumCircuit):
        self.qc = qc
        self.num_qubits = self.qc.num_qubits
        self.qiskit_dag = circuit_to_dag(self.qc)
        self.sim_dag = nx.DiGraph()
        self.gates_info = {}
        self.t_gate_counts = defaultdict(int)
        self._build_dependency_graph()

    def _build_dependency_graph(self):
        qubits = self.qc.qubits
        qubit_map = {q: i for i, q in enumerate(qubits)}
        last_node_on_qubit = {i: -1 for i in range(self.num_qubits)}
        node_id_counter = 0
        
        for node in self.qiskit_dag.topological_op_nodes():
            op_name = node.name
            current_qubits = [qubit_map[q] for q in node.qargs]
            # FIXME: 매직 게이트 정의 필요 (여기서는 T, TDG 만 매직 게이트로 간주)
            is_magic = op_name in ['t', 'tdg']
            
            # Duration: 매직 게이트는 증류 대기에 따라 가변적이므로 여기선 기본값 1
            duration = 1 
            
            if is_magic:
                for q in current_qubits:
                    self.t_gate_counts[q] += 1
            
            self.gates_info[node_id_counter] = {
                'id': node_id_counter,
                'type': 'MAGIC' if is_magic else 'CLIFFORD',
                'qubits': current_qubits,
                'duration': duration
            }
            self.sim_dag.add_node(node_id_counter)
            
            for q in current_qubits:
                prev_node = last_node_on_qubit[q]
                if prev_node != -1:
                    self.sim_dag.add_edge(prev_node, node_id_counter)
                last_node_on_qubit[q] = node_id_counter
                
            node_id_counter += 1