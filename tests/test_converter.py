import sys
import unittest
sys.path.append('../')

import numpy as np
import torch

from cubical.converter import GrayscaleConverter


class TestGrayscaleConverter(unittest.TestCase):
    def test_convert_numpy_array(self):
        # RGB 이미지 생성
        rgb_image = np.random.rand(100, 100, 3)
        rgb_dataset = np.random.rand(1000, 100, 100, 3)
        
        # GrayscaleConverter 인스턴스 생성
        converter = GrayscaleConverter(rgb_dataset)

        # 이미지 변환
        grayscale_image = converter.convert(rgb_image)
        grayscale_dataset = converter.convert_dataset()

        # 변환된 이미지 검증
        self.assertEqual(grayscale_image.shape, (100, 100, 1))
        self.assertEqual(grayscale_dataset.shape, (1000, 100, 100, 3))

    def test_convert_torch_tensor(self):
        # RGB 이미지 생성
        rgb_image = torch.random.rand(3, 100, 100)
        rgb_dataset = torch.random.rand(1000, 3, 100, 100)
        
        # GrayscaleConverter 인스턴스 생성
        converter = GrayscaleConverter()

        # 이미지 변환
        grayscale_image = converter.convert(rgb_image)
        grayscale_dataset = converter.convert_dataset()

        # 변환된 이미지 검증
        self.assertEqual(grayscale_image.shape, (1, 100, 100))
        self.assertEqual(grayscale_dataset.shape, (1000, 1, 100, 100))


if __name__ == '__main__':
    unittest.main()