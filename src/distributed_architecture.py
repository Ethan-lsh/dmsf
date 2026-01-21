from elastic_architecture import ElasticArchitecture


class DistributedArchitecture(ElasticArchitecture):
    def __init__(self, size):
        super().__init__(size)
        self.distributed_region_size = None
        self.distributed_rows = 0
        self.distributed_cols = 0
        self.distributed_region_factories = {}

    def configure_distributed_factories(self, region_size):
        """
        균일 서브영역 분할 후 각 영역 중앙에 factory 배치
        """
        self.distributed_region_size = max(1, int(region_size))
        self.distributed_rows = (self.size + self.distributed_region_size - 1) // self.distributed_region_size
        self.distributed_cols = (self.size + self.distributed_region_size - 1) // self.distributed_region_size
        self.distributed_region_factories = {}

        for ry in range(self.distributed_rows):
            for rx in range(self.distributed_cols):
                x0 = rx * self.distributed_region_size
                y0 = ry * self.distributed_region_size
                x1 = min(x0 + self.distributed_region_size - 1, self.size - 1)
                y1 = min(y0 + self.distributed_region_size - 1, self.size - 1)
                cx = (x0 + x1) // 2
                cy = (y0 + y1) // 2
                factory = self.allocate_factory(cx, cy, force=True)
                self.distributed_region_factories[(rx, ry)] = factory

    def get_distributed_region(self, x, y):
        if self.distributed_region_size is None:
            return None
        rx = min(x // self.distributed_region_size, self.distributed_cols - 1)
        ry = min(y // self.distributed_region_size, self.distributed_rows - 1)
        return (rx, ry)

    def get_distributed_factory_for_coord(self, x, y):
        region = self.get_distributed_region(x, y)
        if region is None:
            return None
        return self.distributed_region_factories.get(region)
