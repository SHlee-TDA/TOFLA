from abc import ABC, abstractmethod
import logging

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from gudhi.sklearn.cubical_persistence import CubicalPersistence

from .converter import GrayscaleConverter
from .entropy_calculator import Vectorization


# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


persistence_0d = CubicalPersistence(
    homology_dimensions=0,
    input_type='vertices',
    homology_coeff_field=2,
    n_jobs=-1
)

persistence_1d = CubicalPersistence(
    homology_dimensions=1,
    input_type='vertices',
    homology_coeff_field=2,
    n_jobs=-1
)


class OptimizationProcess(ABC):
    def __init__(self, dataset):
        self.dataset = dataset
        self.best_params = None
        self.max_entropy = -float('inf')

    @abstractmethod
    def optimize(self):
        pass
    

class GridSearch(OptimizationProcess):
    def __init__(self, dataset, steps=10):
        super().__init__(dataset)
        self.steps = steps  # Grid search에서 사용할 각 차원의 스텝 수

    def optimize(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        # 엔트로피 값을 저장할 리스트
        entropies = []
        
        # Grid search 로직
        for x1 in np.linspace(0, 1, num=self.steps):
            for x2 in np.linspace(0, 1, num=self.steps):
                for x3 in np.linspace(0, 1, num=self.steps):
                    # GrayscaleConverter 인스턴스 생성 및 파라미터 설정
                    converter = GrayscaleConverter(dataset=self.dataset, x1=x1, x2=x2, x3=x3)

                    # Vectorization 인스턴스 생성 및 데이터셋 변환
                    vectorizer_0d = Vectorization(self.dataset, converter, persistence_0d)
                    vectorizer_1d = Vectorization(self.dataset, converter, persistence_1d)
                    vectors_0d = vectorizer_0d.transform()
                    vectors_1d = vectorizer_1d.transform()

                    # 평균 엔트로피 계산
                    entropy_avg = np.mean(vectors_0d) + np.mean(vectors_1d)
                    entropies.append(entropy_avg)
                    logger.info(f"Params: ({x1:.4f}, {x2:.4f}, {x3:.4f}), Average Entropy: {entropy_avg:.4f}")


                    # 최대 엔트로피 및 최적 파라미터 업데이트
                    if entropy_avg > self.max_entropy:
                        self.max_entropy = entropy_avg
                        self.best_params = (x1, x2, x3)
                        
        # 점의 색상을 엔트로피 값에 따라 설정
        sc = ax.scatter(*np.meshgrid(*[np.linspace(0, 1, num=self.steps)]*3), c=entropies, cmap='viridis')

        
        ax.set_xlabel('X1')
        ax.set_ylabel('X2')
        ax.set_zlabel('X3')
        plt.colorbar(sc)  # 컬러바 추가
        plt.show()
        
        return self.best_params, self.max_entropy


class RandomSearch(OptimizationProcess):
    def optimize(self):
        # Random search 로직 구현
        pass
