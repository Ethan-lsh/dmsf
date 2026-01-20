import random
from configuration import Configuration

class MagicFactoryUnit:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.busy_until = 0         # 공장이 바쁠 때까지의 시간
        self.produced_quality = 0.0 # Error rate of the produced state
    
    def start_distillation(self, current_time):
        """
        확률적 증류 프로세스 시뮬레이션
        Returns: (completion_time, error_rate)
        """
        # 1. 기본 소요 시간 (15-to-1은 약 10~20 사이클 가정)
        distillation_cycles = Configuration.get_logical_cycle_duration(10)
        
        # 2. 성공 확률 (Yield) 시뮬레이션
        # 15-to-1 프로토콜은 내부 오류 감지 시 폐기하고 다시 시작해야 함
        # 성공 확률 P_success = (1 - 15*p_phys) 대략적 근사
        p_success = 1.0 - (15 * Configuration.PHYS_ERROR_RATE)
        
        total_cycles = distillation_cycles
        attempts = 1
        
        # 성공할 때까지 재시도 (기하 분포)
        while random.random() > p_success:
            total_cycles += distillation_cycles
            attempts += 1
            if attempts > 5: break # 무한 루프 방지 (최악의 경우 Raw State 사용)

        # 3. 품질 계산 (증류 성공 시)
        output_error = Configuration.distill_cascade(Configuration.RAW_INJECTION_ERROR, rounds=2)
        
        finish_time = current_time + total_cycles
        self.busy_until = finish_time
        self.produced_quality = output_error
        
        return finish_time, output_error
