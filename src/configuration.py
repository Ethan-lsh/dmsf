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
    HOTSPOT_TOP_FRACTION = 0.3
    HOTSPOT_RADIUS = 2
    HOTSPOT_BUFFER = 2
    FACTORY_IDLE_TIMEOUT = 50
    
    # === [NEW] Code Distance Parameter ===
    # d가 커질수록 오류는 줄어들지만, 실행 시간(Cycle)은 늘어납니다.
    CODE_DISTANCE = 7  # 예: d=3, 5, 7, ...
    
    @staticmethod
    def get_logical_cycle_duration(base_ops):
        """
        논리적 연산이 소요하는 실제 QEC 사이클 수 계산
        Lattice Surgery 등을 고려할 때 시간은 d에 비례함.
        """
        return base_ops * Configuration.CODE_DISTANCE
    
    @staticmethod
    def logical_error_rate_per_cycle():
        """
        Surface code 논리 오류율 근사
        p_L ≈ A * (p_phys / p_th)^((d+1)/2)
        """
        d = Configuration.CODE_DISTANCE
        ratio = Configuration.PHYS_ERROR_RATE / Configuration.SURFACE_CODE_P_TH
        exponent = (d + 1) / 2.0
        return Configuration.SURFACE_CODE_A * (ratio ** exponent)

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
    def distill_15_to_1(input_error):
        # (기존 로직 유지)
        return max(35 * (input_error ** 3), Configuration.PHYS_ERROR_RATE)

    @staticmethod
    def distill_cascade(initial_error, rounds=2):
        """
        다단계 15-to-1 증류 근사
        """
        err = initial_error
        for _ in range(rounds):
            err = Configuration.distill_15_to_1(err)
        return err
