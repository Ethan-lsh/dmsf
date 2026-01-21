import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import random

# Qiskit Imports
from qiskit import QuantumCircuit

from smart_compiler import SmartCompiler
from quantum_workload import QasmWorkload
from elastic_architecture import ElasticArchitecture
from distributed_architecture import DistributedArchitecture
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

def _annotate_qubits(ax, logical_to_phys, color="blue", fontsize=8):
    for q, (x, y) in logical_to_phys.items():
        ax.text(y, x, f"Q{q}", ha="center", va="center", color=color, fontsize=fontsize)


def _plot_factory_utilization(ax, utilization_history):
    ax.plot(utilization_history)
    ax.set_title("Dynamic Factory Utilization over Time")
    ax.set_xlabel("Time (Cycles)")
    ax.set_ylabel("Active Factories")
    ax.grid(True)


def _save_figure(fig, output_file):
    fig.tight_layout()
    fig.savefig(output_file, dpi=300)
    print(f"[Output] Plot saved to {output_file}")


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
    _annotate_qubits(ax1, sim.arch.logical_to_phys, color="blue", fontsize=8)

    _plot_factory_utilization(ax2, sim.factory_utilization_history)
    _save_figure(fig, output_file)

def visualize_hotspot_and_factories(sim, output_file):
    """
    핫스팟/버퍼 영역과 MSF 타일 배치 시각화
    """
    arch = sim.arch
    workload = sim.workload
    error_grid = np.zeros((arch.size, arch.size), dtype=float)
    logical_error = Configuration.logical_error_rate_per_cycle()
    for q, (x, y) in arch.logical_to_phys.items():
        t_count = workload.t_gate_counts.get(q, 0)
        error_grid[x, y] = t_count * logical_error
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    vmax = float(np.max(error_grid)) if np.max(error_grid) > 0 else 1.0
    im = ax1.imshow(error_grid, cmap="Reds", interpolation="nearest", vmin=0.0, vmax=vmax)
    ax1.set_title("Hotspot Tile Error Rate")
    ax1.set_xticks([])
    ax1.set_yticks([])

    _annotate_qubits(ax1, arch.logical_to_phys, color="black", fontsize=6)

    fig.colorbar(im, ax=ax1)

    for (x, y) in arch.factory_zone_tiles:
        buffer_rect = matplotlib.patches.Rectangle(
            (y - 0.5, x - 0.5),
            1,
            1,
            fill=False,
            edgecolor="#1f77b4",
            linewidth=1.0,
        )
        ax1.add_patch(buffer_rect)
    
    for (x, y) in arch.hotspot_tiles:
        hotspot_rect = matplotlib.patches.Rectangle(
            (y - 0.5, x - 0.5),
            1,
            1,
            fill=False,
            edgecolor="#d62728",
            linewidth=1.5,
        )
        ax1.add_patch(hotspot_rect)
    
    for (x, y) in arch.active_factories.keys():
        factory_rect = matplotlib.patches.Rectangle(
            (y - 0.5, x - 0.5),
            1,
            1,
            fill=False,
            edgecolor="#2ca02c",
            linewidth=1.5,
        )
        ax1.add_patch(factory_rect)
    legend_items = [
        matplotlib.lines.Line2D([0], [0], color="#d62728", lw=2, label="Hotspot"),
        matplotlib.lines.Line2D([0], [0], color="#1f77b4", lw=2, label="Buffer"),
        matplotlib.lines.Line2D([0], [0], color="#2ca02c", lw=2, label="MSF"),
    ]
    ax1.legend(handles=legend_items, loc="upper right", fontsize=8, framealpha=0.9)
    
    _plot_factory_utilization(ax2, sim.factory_utilization_history)
    _save_figure(fig, output_file)

if __name__ == "__main__":
    # 1. Setup (Monte Carlo)
    GRID_SIZE = 12
    NUM_RUNS = 10
    num_qubits = 5
    num_gates = 10
    distances = [3, 5, 7]
    
    elastic_results = []
    fixed_results = []
    
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
            
            spacing = Configuration.DISTRIBUTED_FACTORY_SPACING
            rows = (GRID_SIZE + spacing - 1) // spacing
            cols = (GRID_SIZE + spacing - 1) // spacing
            num_factories = rows * cols
            distributed_k = Configuration.distributed_factory_k(
                Configuration.DISTRIBUTED_TOTAL_K,
                num_factories,
                Configuration.DISTRIBUTED_DISTILL_ROUNDS,
            )
            t_gate_latency = Configuration.t_gate_latency_cycles()
            # distill_time_cycles, distill_round_output_error, distill_round_success_prob
            # are intentionally not stored in results.
            distributed_output_capacity = Configuration.distributed_output_capacity(
                Configuration.DISTRIBUTED_TOTAL_K,
                num_factories,
                Configuration.DISTRIBUTED_DISTILL_ROUNDS,
                Configuration.RAW_INJECTION_ERROR,
            )
            distributed_area = Configuration.distributed_factories_area(
                num_factories,
                Configuration.DISTRIBUTED_TOTAL_K,
                Configuration.DISTRIBUTED_DISTILL_ROUNDS,
                Configuration.CODE_DISTANCE,
            )
            
            # 3. Architecture & Sim (Elastic)
            arch_elastic = ElasticArchitecture(size=GRID_SIZE)
            arch_elastic.apply_mapping(mapping, hotspot_info)
            
            sim_elastic = RuntimeSimulator(workload, arch_elastic, policy_mode="elastic")
            sim_elastic.run()
            if run_idx == 0:
                plot_file = f"output/hotspot_factories_elastic_d{d}.png"
                visualize_hotspot_and_factories(sim_elastic, plot_file)
            
            elastic_results.append({
                "run": run_idx,
                "code_distance": d,
                "policy": "elastic",
                "final_fidelity": sim_elastic.circuit_fidelity,
                "logical_failure_prob": sim_elastic.logical_failure_prob,
                "total_cycles": sim_elastic.current_time,
                "t_gate_latency": t_gate_latency,
                "distributed_output_capacity": distributed_output_capacity,
                "distributed_area": distributed_area,
                "avg_wait": float(np.mean(sim_elastic.wait_times)) if sim_elastic.wait_times else 0.0,
                "p95_wait": float(np.percentile(sim_elastic.wait_times, 95)) if sim_elastic.wait_times else 0.0,
                "avg_transport_dist": float(np.mean(sim_elastic.transport_distances)) if sim_elastic.transport_distances else 0.0,
                "avg_t_parallelism": float(np.mean(sim_elastic.t_parallelism_history)) if sim_elastic.t_parallelism_history else 0.0,
                "avg_congestion": float(np.mean(sim_elastic.congestion_factor_history)) if sim_elastic.congestion_factor_history else 0.0,
                "avg_n_distill": float(np.mean(sim_elastic.n_distill_history)) if sim_elastic.n_distill_history else 0.0,
                "t_total_estimate": sim_elastic.t_total_estimate,
            })
            
            # 4. Architecture & Sim (Fixed baseline)
            arch_fixed = DistributedArchitecture(size=GRID_SIZE)
            arch_fixed.apply_mapping(mapping, hotspot_info)
            arch_fixed.configure_distributed_factories(Configuration.DISTRIBUTED_FACTORY_SPACING)
            num_factories = len(arch_fixed.distributed_region_factories)
            
            sim_fixed = RuntimeSimulator(workload, arch_fixed, policy_mode="fixed")
            sim_fixed.run()
            if run_idx == 0:
                plot_file = f"output/hotspot_factories_fixed_d{d}.png"
                visualize_hotspot_and_factories(sim_fixed, plot_file)

            fixed_results.append({
                "run": run_idx,
                "code_distance": d,
                "policy": "fixed",
                "final_fidelity": sim_fixed.circuit_fidelity,
                "logical_failure_prob": sim_fixed.logical_failure_prob,
                "total_cycles": sim_fixed.current_time,
                "t_gate_latency": t_gate_latency,
                "distributed_output_capacity": distributed_output_capacity,
                "distributed_area": distributed_area,
                "avg_wait": float(np.mean(sim_fixed.wait_times)) if sim_fixed.wait_times else 0.0,
                "p95_wait": float(np.percentile(sim_fixed.wait_times, 95)) if sim_fixed.wait_times else 0.0,
                "avg_transport_dist": float(np.mean(sim_fixed.transport_distances)) if sim_fixed.transport_distances else 0.0,
                "avg_t_parallelism": float(np.mean(sim_fixed.t_parallelism_history)) if sim_fixed.t_parallelism_history else 0.0,
                "avg_congestion": float(np.mean(sim_fixed.congestion_factor_history)) if sim_fixed.congestion_factor_history else 0.0,
                "avg_n_distill": float(np.mean(sim_fixed.n_distill_history)) if sim_fixed.n_distill_history else 0.0,
                "t_total_estimate": sim_fixed.t_total_estimate,
            })
    
    all_results = elastic_results + fixed_results

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
                "t_gate_latency",
                "distributed_output_capacity",
                "distributed_area",
                "avg_wait",
                "p95_wait",
                "avg_transport_dist",
                "avg_t_parallelism",
                "avg_congestion",
                "avg_n_distill",
                "t_total_estimate",
            ],
        )
        writer.writeheader()
        writer.writerows(all_results)
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
        
        summary_metrics = [
            "final_fidelity",
            "logical_failure_prob",
            "total_cycles",
            "t_gate_latency",
            "distributed_output_capacity",
            "distributed_area",
            "avg_wait",
            "p95_wait",
            "avg_transport_dist",
            "avg_t_parallelism",
            "avg_congestion",
            "avg_n_distill",
            "t_total_estimate",
        ]
        for d in distances:
            for policy in ["elastic", "fixed"]:
                for metric in summary_metrics:
                    subset = [r for r in all_results if r["code_distance"] == d and r["policy"] == policy]
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
    for r in all_results:
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
                "delta_avg_wait": fx["avg_wait"] - el["avg_wait"],
                "delta_avg_transport_dist": fx["avg_transport_dist"] - el["avg_transport_dist"],
            })
    print(f"[Output] Policy comparison saved to {comparison_file}")
    
    # 6. Trend Visualization (metric vs code distance)
    plot_metrics = [
        "final_fidelity",
        "logical_failure_prob",
        "total_cycles",
        "t_gate_latency",
        "avg_wait",
        "p95_wait",
        "avg_transport_dist",
        "avg_congestion",
        "avg_n_distill",
        "t_total_estimate",
    ]
    for metric in plot_metrics:
        plt.figure(figsize=(8, 5))
        
        for label, dataset in [("elastic", elastic_results), ("fixed", fixed_results)]:
            means = []
            ci_lows = []
            ci_highs = []
            for d in distances:
                subset = [r for r in dataset if r["code_distance"] == d]
                values = np.array([r[metric] for r in subset], dtype=float)
                mean = float(np.mean(values))
                stderr = float(np.std(values, ddof=1) / np.sqrt(len(values)))
                ci95 = 1.96 * stderr
                means.append(mean)
                ci_lows.append(mean - ci95)
                ci_highs.append(mean + ci95)
            
            plt.plot(distances, means, marker="o", linewidth=2, label=f"{label}:{metric}")
            plt.fill_between(distances, ci_lows, ci_highs, alpha=0.2)
        
        plt.xlabel("Code Distance (d)")
        plt.ylabel(metric)
        plt.title(f"{metric} Comparison (Elastic vs Fixed)")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend()
        plot_file = f"output/comparison_{metric}.png"
        plt.savefig(plot_file, dpi=300)
        print(f"[Output] Comparison plot saved to {plot_file}")
