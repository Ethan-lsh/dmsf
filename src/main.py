import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from collections import deque, defaultdict
import random

# Qiskit Imports
from qiskit import QuantumCircuit
from qiskit.converters import circuit_to_dag

from smart_compiler import SmartCompiler
from quantum_workload import QasmWorkload
from elastic_architecture import ElasticArchitecture
from runtime_simulator import RuntimeSimulator
from configuration import Configuration

# TODO: 입력 회로를 추가해서 성능 비교하기
# TODO: 정확한 contribution을 설정하기


# ==========================================
# Helper & Main
# ==========================================
def generate_biased_qasm(num_qubits, num_gates):
    # (이전과 동일한 로직, 생략)
    qc = QuantumCircuit(num_qubits)
    hot_indices = list(range(int(num_qubits * 0.2) + 1))
    for _ in range(num_gates):
        r = np.random.rand()
        if r < 0.4: 
            q = np.random.choice(hot_indices) if np.random.rand() < 0.8 else np.random.randint(0, num_qubits)
            qc.t(q)
        elif r < 0.7:
            q1, q2 = np.random.choice(range(num_qubits), 2, replace=False)
            qc.cx(q1, q2)
        else:
            q = np.random.randint(0, num_qubits)
            qc.h(q)
    return qc

def visualize_hotspot_and_utilization(sim, output_file):
    """
    의도된 핫스팟과 공장 가동률 시각화
    """
    grid_map = np.zeros((sim.arch.size, sim.arch.size))
    for q, count in sim.workload.t_gate_counts.items():
        if q in sim.arch.logical_to_phys:
            x, y = sim.arch.logical_to_phys[q]
            grid_map[x, y] = count
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    im = ax1.imshow(grid_map, cmap="Reds", interpolation="nearest")
    ax1.set_title("Hotspot Zone (T-gate Density)")
    fig.colorbar(im, ax=ax1)
    for q, (x, y) in sim.arch.logical_to_phys.items():
        ax1.text(y, x, f"Q{q}", ha="center", va="center", color="blue", fontsize=8)
    
    ax2.plot(sim.factory_utilization_history)
    ax2.set_title("Dynamic Factory Utilization over Time")
    ax2.set_xlabel("Time (Cycles)")
    ax2.set_ylabel("Active Factories")
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    print(f"[Output] Plot saved to {output_file}")

if __name__ == "__main__":
    # 1. Setup (Monte Carlo)
    GRID_SIZE = 12
    NUM_RUNS = 10
    num_qubits = 5
    num_gates = 10
    distances = [3, 5, 7]
    NUM_FIXED_FACTORIES = 2
    
    results = []
    
    for d in distances:
        Configuration.CODE_DISTANCE = d
        for run_idx in range(NUM_RUNS):
            # 재현성 확보를 위한 시드 고정 (run index 기반)
            np.random.seed(run_idx)
            random.seed(run_idx)
            
            qc = generate_biased_qasm(num_qubits=num_qubits, num_gates=num_gates)
            workload = QasmWorkload(qc)
            
            # 2. Compile
            compiler = SmartCompiler(workload, grid_size=GRID_SIZE)
            mapping, hotspot_info = compiler.perform_clustering_mapping()
            
            # 3. Architecture & Sim (Elastic)
            arch_elastic = ElasticArchitecture(size=GRID_SIZE)
            arch_elastic.apply_mapping(mapping, hotspot_info)
            
            sim_elastic = RuntimeSimulator(workload, arch_elastic, policy_mode="elastic")
            sim_elastic.run()
            if run_idx == 0:
                plot_file = f"output/hotspot_utilization_d{d}.png"
                visualize_hotspot_and_utilization(sim_elastic, plot_file)
            
            results.append({
                "run": run_idx,
                "code_distance": d,
                "policy": "elastic",
                "final_fidelity": sim_elastic.circuit_fidelity,
                "logical_failure_prob": sim_elastic.logical_failure_prob,
                "total_cycles": sim_elastic.current_time,
                "distillation_overhead": sim_elastic.total_distillation_overhead,
                "build_count": sim_elastic.build_count,
                "reuse_count": sim_elastic.reuse_count,
                "stall_count": sim_elastic.stall_count,
                "avg_wait": float(np.mean(sim_elastic.wait_times)) if sim_elastic.wait_times else 0.0,
                "p95_wait": float(np.percentile(sim_elastic.wait_times, 95)) if sim_elastic.wait_times else 0.0,
                "avg_transport_dist": float(np.mean(sim_elastic.transport_distances)) if sim_elastic.transport_distances else 0.0,
            })
            
            # 4. Architecture & Sim (Fixed baseline)
            arch_fixed = ElasticArchitecture(size=GRID_SIZE)
            arch_fixed.apply_mapping(mapping, hotspot_info)
            
            # Hot-qubit 기반 고정 공장 배치
            hot_qubits = sorted(
                range(workload.num_qubits),
                key=lambda q: workload.t_gate_counts[q],
                reverse=True,
            )
            placed = 0
            for q in hot_qubits:
                if placed >= NUM_FIXED_FACTORIES:
                    break
                x, y = mapping[q]
                if arch_fixed.grid[x, y] != 2:
                    arch_fixed.allocate_factory(x, y, force=True)
                    placed += 1
            
            sim_fixed = RuntimeSimulator(workload, arch_fixed, policy_mode="fixed")
            sim_fixed.run()
            if run_idx == 0:
                plot_file = f"output/hotspot_utilization_fixed_d{d}.png"
                visualize_hotspot_and_utilization(sim_fixed, plot_file)

            results.append({
                "run": run_idx,
                "code_distance": d,
                "policy": "fixed",
                "final_fidelity": sim_fixed.circuit_fidelity,
                "logical_failure_prob": sim_fixed.logical_failure_prob,
                "total_cycles": sim_fixed.current_time,
                "distillation_overhead": sim_fixed.total_distillation_overhead,
                "build_count": sim_fixed.build_count,
                "reuse_count": sim_fixed.reuse_count,
                "stall_count": sim_fixed.stall_count,
                "avg_wait": float(np.mean(sim_fixed.wait_times)) if sim_fixed.wait_times else 0.0,
                "p95_wait": float(np.percentile(sim_fixed.wait_times, 95)) if sim_fixed.wait_times else 0.0,
                "avg_transport_dist": float(np.mean(sim_fixed.transport_distances)) if sim_fixed.transport_distances else 0.0,
            })
    
    # 4. CSV Output
    import csv
    output_file = "output/monte_carlo_results.csv"
    with open(output_file, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "run",
                "code_distance",
                "policy",
                "final_fidelity",
                "logical_failure_prob",
                "total_cycles",
                "distillation_overhead",
                "build_count",
                "reuse_count",
                "stall_count",
                "avg_wait",
                "p95_wait",
                "avg_transport_dist",
            ],
        )
        writer.writeheader()
        writer.writerows(results)
    print(f"[Output] Monte Carlo results saved to {output_file}")
    
    # 5. Summary (Mean & 95% CI)
    summary_file = "output/monte_carlo_summary.csv"
    with open(summary_file, "w", newline="") as f:
        fieldnames = [
            "code_distance",
            "metric",
            "mean",
            "ci95_low",
            "ci95_high",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for d in distances:
            for policy in ["elastic", "fixed"]:
                for metric in ["final_fidelity", "logical_failure_prob", "total_cycles", "distillation_overhead", "avg_wait", "p95_wait", "avg_transport_dist"]:
                    subset = [r for r in results if r["code_distance"] == d and r["policy"] == policy]
                    values = np.array([r[metric] for r in subset], dtype=float)
                    mean = float(np.mean(values))
                    # 95% CI (normal approximation)
                    stderr = float(np.std(values, ddof=1) / np.sqrt(len(values)))
                    ci95 = 1.96 * stderr
                    writer.writerow({
                        "code_distance": d,
                        "metric": f"{policy}:{metric}",
                        "mean": mean,
                        "ci95_low": mean - ci95,
                        "ci95_high": mean + ci95,
                    })
    print(f"[Output] Monte Carlo summary saved to {summary_file}")
    
    # 5b. Policy Comparison (elastic vs fixed)
    comparison_file = "output/policy_comparison.csv"
    by_key = {}
    for r in results:
        key = (r["code_distance"], r["run"])
        by_key.setdefault(key, {})[r["policy"]] = r
    
    with open(comparison_file, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "run",
                "code_distance",
                "speedup_total_cycles",
                "delta_fidelity",
                "delta_logical_failure_prob",
                "delta_distillation_overhead",
                "delta_avg_wait",
                "delta_avg_transport_dist",
            ],
        )
        writer.writeheader()
        for (d, run_idx), pair in by_key.items():
            if "elastic" not in pair or "fixed" not in pair:
                continue
            el = pair["elastic"]
            fx = pair["fixed"]
            speedup = fx["total_cycles"] / el["total_cycles"] if el["total_cycles"] > 0 else 0.0
            writer.writerow({
                "run": run_idx,
                "code_distance": d,
                "speedup_total_cycles": speedup,
                "delta_fidelity": el["final_fidelity"] - fx["final_fidelity"],
                "delta_logical_failure_prob": fx["logical_failure_prob"] - el["logical_failure_prob"],
                "delta_distillation_overhead": fx["distillation_overhead"] - el["distillation_overhead"],
                "delta_avg_wait": fx["avg_wait"] - el["avg_wait"],
                "delta_avg_transport_dist": fx["avg_transport_dist"] - el["avg_transport_dist"],
            })
    print(f"[Output] Policy comparison saved to {comparison_file}")
    
    # 6. Trend Visualization (metric vs code distance)
    metrics = ["final_fidelity", "logical_failure_prob", "total_cycles", "distillation_overhead", "avg_wait", "p95_wait", "avg_transport_dist"]
    for metric in metrics:
        plt.figure(figsize=(8, 5))
        for policy in ["elastic", "fixed"]:
            means = []
            ci_lows = []
            ci_highs = []
            for d in distances:
                subset = [r for r in results if r["code_distance"] == d and r["policy"] == policy]
                values = np.array([r[metric] for r in subset], dtype=float)
                mean = float(np.mean(values))
                stderr = float(np.std(values, ddof=1) / np.sqrt(len(values)))
                ci95 = 1.96 * stderr
                means.append(mean)
                ci_lows.append(mean - ci95)
                ci_highs.append(mean + ci95)
            
            plt.plot(distances, means, marker="o", linewidth=2, label=f"{policy}:{metric}")
            plt.fill_between(distances, ci_lows, ci_highs, alpha=0.2)
        
        plt.xlabel("Code Distance (d)")
        plt.ylabel(metric)
        plt.title(f"{metric} vs Code Distance")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend()
        plot_file = f"output/trend_{metric}.png"
        plt.savefig(plot_file, dpi=300)
        print(f"[Output] Trend plot saved to {plot_file}")
