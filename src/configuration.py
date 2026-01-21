# ==========================================
# 0. Physics Constants & Protocols
# ==========================================
class Configuration:
    PHYS_ERROR_RATE = 0.001
    RAW_INJECTION_ERROR = 0.01
    # 시간 기반 오류 모델 파라미터
    CYCLE_TIME_US = 1.0
    T1_US = 200.0
    T2_US = 150.0
    
    # Surface code 근사 계수 (문헌 기반 보정 가능)
    SURFACE_CODE_A = 0.1
    SURFACE_CODE_P_TH = 0.01
    
    # Hotspot + Elastic Factory 정책 파라미터
    HOTSPOT_TOP_FRACTION = 0.3      # 상위 몇 % 핫스팟을 고려할지
    HOTSPOT_RADIUS = 2
    HOTSPOT_BUFFER = 2
    FACTORY_IDLE_TIMEOUT = 50
    
    # Distributed MSF 모델 파라미터 (변경 가능)
    DISTRIBUTED_TOTAL_K = 1
    DISTRIBUTED_DISTILL_ROUNDS = 3
    DISTRIBUTED_FACTORY_SPACING = 4
    DISTILLATION_BLOCK_N = 15
    
    # d가 커질수록 오류는 줄어들지만, 실행 시간(Cycle)은 늘어납니다.
    CODE_DISTANCE = None  # None이면 자동 설정
    
    @staticmethod
    def get_logical_cycle_duration(base_ops):
        """
        Logical operation latency in surface code cycles
        """
        return base_ops * Configuration.CODE_DISTANCE

    @staticmethod
    def t_gate_latency_cycles():
        """
        Eq. (2) T-gate execution latency
        E[T_T] = 4d + 4
        """
        return 4 * Configuration.CODE_DISTANCE + 4

    @staticmethod
    def distillation_time_cycles(rounds):
        """
        Eq. (8) Time per factory cycle
        T_distill = 11 * sum d_r (d_r ~= d)
        """
        return 11 * rounds * Configuration.CODE_DISTANCE
    
    @staticmethod
    def logical_error_rate_per_cycle():
        """
        Logical reliability (p_L)
        Eq. (1) P_L ~ d * (100 * eps_in)^((d+1)/2)
        Purpose: Map physical error rate to logical error rate based on code distance
        """
        d = Configuration.CODE_DISTANCE
        eps_in = Configuration.PHYS_ERROR_RATE
        return d * ((100.0 * eps_in) ** ((d + 1) / 2.0))

    @staticmethod
    def decoherence_survival(time_cycles):
        """
        시간 기반 감쇠 모델: exp(-t/T1)과 exp(-t/T2) 혼합 근사
        """
        import math
        t_us = time_cycles * Configuration.CYCLE_TIME_US
        p_t1 = math.exp(-t_us / Configuration.T1_US)
        p_t2 = math.exp(-t_us / Configuration.T2_US)
        return min(p_t1, p_t2)

    @staticmethod
    def distributed_factory_k(total_k, num_factories, rounds):
        """
        Eq. (9) k = (K / X)^(1/l)
        """
        if num_factories <= 0 or rounds <= 0:
            return 0.0
        return (total_k / num_factories) ** (1.0 / rounds)

    @staticmethod
    def distillation_threshold(k):
        """
        Eq. (4) eps_thresh ≈ 1 / (3k + 1)
        """
        return 1.0 / (3.0 * k + 1.0) if k > 0 else 0.0

    @staticmethod
    def distillation_success_prob(eps_in, k):
        """
        Eq. (5) P_success ≈ 1 - (8 + 3k) * eps_in
        """
        return max(0.0, 1.0 - (8.0 + 3.0 * k) * eps_in)

    @staticmethod
    def distillation_output_error(eps_in, k):
        """
        Eq. (3) eps_out = (1 + 3k) * eps_in^2
        """
        return (1.0 + 3.0 * k) * (eps_in ** 2)

    @staticmethod
    def bravyi_haah_yield_ok(inject_error, total_k, num_factories, rounds):
        """
        Eq. (5) yield condition: P_success > 0
        """
        k = Configuration.distributed_factory_k(total_k, num_factories, rounds)
        return Configuration.distillation_success_prob(inject_error, k) > 0.0

    @staticmethod
    def distributed_output_capacity(total_k, num_factories, rounds, inject_error):
        """
        Eq. (10) K_output = K * Π_r (1 - yield_loss_r)
        """
        k = Configuration.distributed_factory_k(total_k, num_factories, rounds)
        eps = inject_error
        capacity = total_k
        for _ in range(rounds):
            p_success = Configuration.distillation_success_prob(eps, k)
            capacity *= p_success
            eps = Configuration.distillation_output_error(eps, k)
        return capacity

    @staticmethod
    def distillation_module_count(rounds, k, block_n=None):
        """
        Eq. (7) N_distill = sum_{r=1}^l k^{r-1} n^{l-r}
        """
        if block_n is None:
            block_n = Configuration.DISTILLATION_BLOCK_N
        total = 0.0
        for r in range(1, rounds + 1):
            total += (k ** (r - 1)) * (block_n ** (rounds - r))
        return total

    @staticmethod
    def distributed_factories_area(num_factories, total_k, rounds, code_distance):
        """
        Eq. (12) A_r ∝ X * (6k + 14) * d_r^2
        """
        k = Configuration.distributed_factory_k(total_k, num_factories, rounds)
        return num_factories * (6.0 * k + 14.0) * (code_distance ** 2)
    
