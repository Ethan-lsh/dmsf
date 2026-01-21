# ==========================================
# 2. Smart Compiler (Analysis & Clustering)
# ==========================================
from configuration import Configuration

class SmartCompiler:
    def __init__(self, workload, grid_size):
        self.workload = workload
        self.grid_size = grid_size
        self.mapping = {}

    def perform_clustering_mapping(self):
        all_qubits = range(self.workload.num_qubits)
        sorted_qubits = sorted(all_qubits, key=lambda q: self.workload.t_gate_counts[q], reverse=True)
        
        center = self.grid_size // 2
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        x, y = center, center
        steps = 1; step_count = 0; dir_idx = 0
        visited = set()
        
        # Spiral Mapping
        for q in sorted_qubits:
            if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
                self.mapping[q] = (x, y)
                visited.add((x, y))
            dx, dy = directions[dir_idx]
            x, y = x + dx, y + dy
            step_count += 1
            if step_count == steps:
                dir_idx = (dir_idx + 1) % 4
                if dir_idx % 2 == 0: steps += 1
                step_count = 0
        hotspot_info = self._compute_hotspot_info()
        return self.mapping, hotspot_info

    def _compute_hotspot_info(self):
        top_n = max(1, int(self.workload.num_qubits * Configuration.HOTSPOT_TOP_FRACTION))
        hot_qubits = sorted(
            range(self.workload.num_qubits),
            key=lambda q: self.workload.t_gate_counts[q],
            reverse=True,
        )[:top_n]
        positions = [self.mapping[q] for q in hot_qubits if q in self.mapping]

        return {
            "tiles": positions,
            "buffer": Configuration.HOTSPOT_BUFFER,
        }
