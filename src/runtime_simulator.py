import numpy as np
from configuration import Configuration
from msf import MagicFactoryUnit


class RuntimeSimulator:
    def __init__(self, workload, arch, policy_mode="elastic"):
        self.workload = workload
        self.arch = arch
        self.policy_mode = policy_mode  # "elastic" or "fixed"
        self.ready_queue = []
        self.executed_gates = set()
        self.current_time = 0
        self.gate_finish_time = {}
        
        # Metrics
        self.total_distillation_overhead = 0
        self.circuit_fidelity = 1.0 # 초기 충실도 100%
        self.fidelity_history = [1.0]
        self.logical_failure_prob = 0.0
        self.build_count = 0
        self.reuse_count = 0
        self.stall_count = 0
        self.wait_times = []
        self.transport_distances = []
        self.factory_utilization_history = []
        self.t_parallelism_history = []
        self.congestion_factor_history = []
        self.n_distill_history = []
        self.t_total_estimate = 0.0

    def run(self):
        dag = self.workload.sim_dag
        gates_info = self.workload.gates_info
        
        # 초기 노드 로드
        for node in dag.nodes:
            if dag.in_degree(node) == 0:
                self.ready_queue.append(node)
                
        # print(f"\n[Simulation] Start with Physical Error Rate: {Configuration.PHYS_ERROR_RATE}")
        
        while len(self.executed_gates) < len(dag.nodes):
            candidates = []
            new_ready_queue = []
            
            # 1. 실행 가능 여부 체크
            for node in self.ready_queue:
                parents = list(dag.predecessors(node))
                is_parents_done = True
                for p in parents:
                    # Compare parent finish time with current time
                    if p not in self.gate_finish_time or self.gate_finish_time[p] > self.current_time:
                        is_parents_done = False
                        break
                if is_parents_done: candidates.append(node)
                else: new_ready_queue.append(node)
            
            self.ready_queue = new_ready_queue
            
            t_parallel = sum(1 for node in candidates if gates_info[node]["type"] == "MAGIC")
            num_factories = max(1, len(self.arch.active_factories))
            t_gate_latency = Configuration.t_gate_latency_cycles()
            distill_time = Configuration.distillation_time_cycles(Configuration.DISTRIBUTED_DISTILL_ROUNDS)
            congestion_factor = np.sqrt(t_parallel) if t_parallel > 0 else 0.0
            if t_parallel > 0:
                n_distill = (t_gate_latency / distill_time) * np.sqrt(t_parallel / num_factories)
            else:
                n_distill = 0.0
            self.t_parallelism_history.append(t_parallel)
            self.congestion_factor_history.append(congestion_factor)
            self.n_distill_history.append(n_distill)
            self.t_total_estimate += n_distill * t_parallel
            
            # 2. 게이트 실행 및 Fidelity 계산
            for node in candidates:
                gate = gates_info[node]
                gate_type = gate['type']
                
                # A. Clifford Gate (단순 물리 오류)
                if gate_type == 'CLIFFORD':
                    logical_op_cycles = Configuration.get_logical_cycle_duration(1)
                    logical_error = Configuration.logical_error_rate_per_cycle()
                    op_error = 1.0 - ((1.0 - logical_error) ** logical_op_cycles)
                    time_survival = Configuration.decoherence_survival(logical_op_cycles)
                    self.circuit_fidelity *= (1.0 - op_error) * time_survival
                    
                    finish_time = self.current_time + logical_op_cycles
                    self.gate_finish_time[node] = finish_time
                
                # ====================================================
                # [수정된 MAGIC 게이트 처리 로직: 스마트 혼잡 제어]
                # ====================================================
                elif gate_type == 'MAGIC':
                    # FIXME: 모든 Magic State 판별 기능 추가
                    q_idx = gate['qubits'][0]
                    qx, qy = self.arch.logical_to_phys[q_idx]
                    
                    if self.policy_mode == "fixed":
                        factory = self.arch.get_distributed_factory_for_coord(qx, qy)
                        if factory is None:
                            self.gate_finish_time[node] = self.current_time + 100
                            self.stall_count += 1
                            continue
                        wait_time = max(0, factory.busy_until - self.current_time)
                        target_factory = factory
                        start_delay = wait_time
                        self.reuse_count += 1
                        self.wait_times.append(wait_time)
                    else:
                        # 1. 탐색: 가장 가까운 공장(Reuse)과 빈 땅(Build) 동시 탐색 (BFS)
                        factory, empty_pos = self.arch.find_nearest_factory_or_space(qx, qy)
                    
                        # 2. 비용 계산 및 의사결정 변수 설정
                        # 파라미터 정의 (연구 논문 실험 변수로 활용 가능)
                        CONGESTION_THRESHOLD = 15  # 이보다 오래 기다려서 재사용한다면 새 공장 고려
                        SETUP_COST = 5             # 공장 건설에 드는 추가 비용 (초기화 등)
                        
                        target_factory = None
                        final_overhead = 0
                        
                        # (A) 기존 공장 재사용 비용 계산
                        reuse_cost = float('inf')
                        if factory:
                            wait_time = max(0, factory.busy_until - self.current_time)
                            transport_dist = abs(factory.x - qx) + abs(factory.y - qy)
                            reuse_cost = wait_time + transport_dist
                            
                        # (B) 신규 건설 비용 계산
                        build_cost = float('inf')
                        if empty_pos:
                            ex, ey = empty_pos
                            transport_dist_new = abs(ex - qx) + abs(ey - qy)
                            # 새 공장은 대기 시간 0이지만, 건설/준비 시간(SETUP_COST)이 듦
                            build_cost = 0 + transport_dist_new + SETUP_COST

                        # 3. 의사결정 (Elastic Logic)
                        should_build = False
                        if empty_pos is not None:
                            if factory is None:
                                should_build = True
                            elif wait_time > CONGESTION_THRESHOLD: # 너무 혼잡하면 건설
                                should_build = True
                                # print(f"  [Decide] Congestion (Wait {wait_time}). Building new at {empty_pos}")
                            elif build_cost < reuse_cost: # 짓는게 더 빠르면 건설
                                should_build = True
                        
                        # 4. 액션 실행
                        start_delay = 0
                        if should_build:
                            # 신규 건설
                            ex, ey = empty_pos
                            target_factory = self.arch.allocate_factory(ex, ey)
                            if target_factory is None:
                                self.gate_finish_time[node] = self.current_time + 100
                                self.stall_count += 1
                                continue
                            start_delay = SETUP_COST
                            self.build_count += 1
                        elif factory:
                            # 재사용
                            target_factory = factory
                            start_delay = wait_time
                            self.reuse_count += 1
                            self.wait_times.append(wait_time)
                        else:
                            # 공간 없음 (Stall)
                            self.gate_finish_time[node] = self.current_time + 100
                            self.stall_count += 1
                            continue

                    # 5. 증류 및 완료 처리
                    start_time = self.current_time + start_delay
                    if hasattr(self.arch, "distributed_region_factories") and self.arch.distributed_region_factories:
                        num_factories = len(self.arch.distributed_region_factories)
                    else:
                        num_factories = max(1, len(self.arch.active_factories))
                    k = Configuration.distributed_factory_k(
                        Configuration.DISTRIBUTED_TOTAL_K,
                        num_factories,
                        Configuration.DISTRIBUTED_DISTILL_ROUNDS,
                    )
                    distill_finish, magic_error = target_factory.start_distillation(start_time, k)
                    
                    # 이동 거리 (Manhattan)
                    distance = abs(target_factory.x - qx) + abs(target_factory.y - qy)
                    congestion_multiplier = max(1.0, congestion_factor)
                    transport_time = distance * Configuration.CODE_DISTANCE * congestion_multiplier
                    self.transport_distances.append(distance)
                    
                    finish_time = distill_finish + transport_time
                    
                    # Fidelity Update
                    # 1) Magic State 자체 오류 + 2) 논리 오류 + 3) 시간 감쇠
                    transport_error = distance * Configuration.PHYS_ERROR_RATE
                    logical_error = Configuration.logical_error_rate_per_cycle()
                    magic_cycles = max(1, finish_time - self.current_time)
                    op_error = 1.0 - ((1.0 - logical_error) ** magic_cycles)
                    time_survival = Configuration.decoherence_survival(magic_cycles)
                    total_error = min(1.0, magic_error + transport_error + op_error)
                    self.circuit_fidelity *= (1.0 - total_error) * time_survival
                    
                    # Overhead 측정 (이상적 실행 시간 대비 지연)
                    ideal_duration = Configuration.t_gate_latency_cycles()
                    overhead = (finish_time - self.current_time) - ideal_duration
                    self.total_distillation_overhead += max(0, overhead)
                    
                    self.gate_finish_time[node] = finish_time
           
                self.executed_gates.add(node)
                
                # 자식 노드 예약 (간소화)
                for child in list(dag.successors(node)):
                    if child not in self.executed_gates and child not in self.ready_queue and child not in candidates:
                         if child not in self.ready_queue:
                             self.ready_queue.append(child)

            self.current_time += 1
            self.fidelity_history.append(self.circuit_fidelity)
            self.logical_failure_prob = 1.0 - self.circuit_fidelity
            if self.policy_mode != "fixed":
                self.arch.cleanup_factories(self.current_time, Configuration.FACTORY_IDLE_TIMEOUT)
            self.factory_utilization_history.append(len(self.arch.active_factories))
            
            if not self.ready_queue and len(self.executed_gates) == len(dag.nodes):
                break
            if self.current_time > 50000: 
                print("Timeout Reached")
                break

        # print(f"[Simulation] Finished. Total Cycles: {self.current_time}")
        # print(f"[Metrics] Final Circuit Fidelity: {self.circuit_fidelity:.6f}")
        # print(f"[Metrics] Logical Failure Probability: {self.logical_failure_prob:.6f}")
        # print(f"[Metrics] Total Distillation Latency: {self.total_distillation_overhead}")
        if self.wait_times:
            avg_wait = float(np.mean(self.wait_times))
            p95_wait = float(np.percentile(self.wait_times, 95))
            # print(f"[Metrics] Avg Magic Wait: {avg_wait:.2f}, P95: {p95_wait:.2f}")
        if self.transport_distances:
            avg_dist = float(np.mean(self.transport_distances))
            # print(f"[Metrics] Avg Transport Distance: {avg_dist:.2f}")
