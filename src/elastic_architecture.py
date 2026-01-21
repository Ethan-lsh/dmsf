from collections import deque
import numpy as np
from msf import MagicFactoryUnit

class ElasticArchitecture:
    def __init__(self, size):
        self.size = size
        self.grid = np.zeros((size, size), dtype=int) # 0: Empty, 1: Storage, 2: Factory
        self.logical_to_phys = {} 
        self.active_factories = {} # (x,y) -> MagicFactoryUnit
        self.hotspot_center = None
        self.hotspot_radius = 0
        self.hotspot_buffer = 0
        self.hotspot_tiles = set()
        self.factory_zone_tiles = set()

    def apply_mapping(self, mapping, hotspot_info=None):
        self.logical_to_phys = mapping
        for q, (x, y) in mapping.items():
            self.grid[x, y] = 1 # Storage
        if hotspot_info:
            self.set_hotspot(
                hotspot_info["tiles"],
                hotspot_info["buffer"],
            )

    def set_hotspot(self, tiles, buffer_size):
        self.hotspot_center = None
        self.hotspot_radius = 0
        self.hotspot_buffer = buffer_size
        self.hotspot_tiles = set((x, y) for (x, y) in tiles)
        self.factory_zone_tiles = set()
        
        for hx, hy in self.hotspot_tiles:
            for dx in range(-buffer_size, buffer_size + 1):
                for dy in range(-buffer_size, buffer_size + 1):
                    if max(abs(dx), abs(dy)) > buffer_size:
                        continue
                    nx, ny = hx + dx, hy + dy
                    if 0 <= nx < self.size and 0 <= ny < self.size:
                        if (nx, ny) not in self.hotspot_tiles:
                            self.factory_zone_tiles.add((nx, ny))

    def allocate_factory(self, x, y, force=False):
        if not force and not self.is_factory_zone_tile(x, y):
            return None
        if not force and self.grid[x, y] != 0:
            return None
        self.grid[x, y] = 2
        factory = MagicFactoryUnit(x, y)
        self.active_factories[(x, y)] = factory
        return factory
    
    def is_factory_zone_tile(self, x, y):
        if not self.factory_zone_tiles:
            return True
        return (x, y) in self.factory_zone_tiles
    
    def find_nearest_factory_or_space(self, qx, qy, max_search_dist=None):
        """
        주어진 큐비트 위치(qx, qy)에서 가장 가까운
        1. 기존 가동 중인 공장 (Reuse Candidate)
        2. 공장을 지을 수 있는 빈 공간 (Build Candidate)
        을 동시에 탐색하여 반환합니다.
        
        Returns:
            (nearest_factory, nearest_empty_pos)
            - nearest_factory: MagicFactoryUnit 객체 (없으면 None)
            - nearest_empty_pos: (x, y) 튜플 (없으면 None)
        """
        if max_search_dist is None:
            max_search_dist = self.size  # 전체 그리드 탐색
            
        queue = deque([(qx, qy, 0)]) # (x, y, distance)
        visited = set([(qx, qy)])
        
        nearest_factory = None
        nearest_empty_pos = None
        
        while queue:
            cx, cy, dist = queue.popleft()
            
            # 탐색 거리 제한 (성능 최적화)
            if dist > max_search_dist:
                break
                
            # 타일 상태 확인
            tile_type = self.grid[cx, cy]
            
            # 1. 빈 공간(TileMode.EMPTY = 0) 발견
            if tile_type == 0 and nearest_empty_pos is None and self.is_factory_zone_tile(cx, cy):
                nearest_empty_pos = (cx, cy)
            
            # 2. 공장(TileMode.FACTORY = 2) 발견
            if tile_type == 2 and nearest_factory is None:
                # 좌표로 객체를 찾아서 반환
                if (cx, cy) in self.active_factories:
                    nearest_factory = self.active_factories[(cx, cy)]
            
            # 두 후보를 모두 찾았으면 즉시 조기 종료 (Early Exit)
            if nearest_factory is not None and nearest_empty_pos is not None:
                return nearest_factory, nearest_empty_pos
            
            # 인접 타일 탐색
            for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < self.size and 0 <= ny < self.size:
                    if (nx, ny) not in visited:
                        visited.add((nx, ny))
                        queue.append((nx, ny, dist + 1))
                        
        return nearest_factory, nearest_empty_pos

    def cleanup_factories(self, current_time, idle_timeout):
        to_remove = []
        for (x, y), factory in self.active_factories.items():
            if current_time >= factory.busy_until + idle_timeout:
                to_remove.append((x, y))
        
        for key in to_remove:
            x, y = key
            if self.grid[x, y] == 2:
                self.grid[x, y] = 0
            del self.active_factories[key]
