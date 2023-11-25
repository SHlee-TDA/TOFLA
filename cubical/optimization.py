import numpy as np

from converter import GrayscaleConverter
from entropy_calculator import Vectorization


class OptimizationProcess:
    def __init__(self, dataset, search_option='grid'):
        self.dataset = dataset
        self.best_params = (0, 0, 0)
        self.max_entropy = 0.0
        self.search_option = search_option
        # 추가 초기화 로직

    def grid_search(self, resolution):
        # Grid search 로직
        for x1 in np.linspace(0, 1, num=resolution):  # 예시로 num=10을 사용
            for x2 in np.linspace(0, 1, num=resolution):
                for x3 in np.linspace(0, 1, num=resolution):
                    converter = GrayscaleConverter(x1, x2, x3)
                    vectorizer = Vectorization(self.dataset, converter)
                    entropy_avg = np.mean(vectorizer.transform())
                    
                    if entropy_avg > self.max_entropy:
                        self.max_entropy = entropy_avg
                        self.best_params = (x1, x2, x3)
    
    def optimize(self):
        # 최적화 옵션에 따라 메소드 선택
        if self.search_option == 'grid':
            self.grid_search()
        # 여기에 다른 최적화 방법 추가 (예: random search)
        # ...

        return self.best_params, self.max_entropy