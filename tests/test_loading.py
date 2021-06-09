import unittest

import sense.loading as loading
from sense import RESOURCES_DIR


class TestLoadWeightsFromResources(unittest.TestCase):

    RELATIVE_PATH = loading.MODELS['StridedInflatedEfficientNet']['lite']['action_recognition']
    ABSOLUTE_PATH = '{}/{}'.format(RESOURCES_DIR, RELATIVE_PATH)

    def test_load_weights_from_resources_on_relative_path(self):
        _ = loading.load_weights_from_resources(self.RELATIVE_PATH)

    def test_load_weights_from_resources_on_absolute_path(self):
        _ = loading.load_weights_from_resources(self.ABSOLUTE_PATH)

    def test_load_weights_from_resources_on_wrong_path(self):
        wrong_path = 'this/path/does/not/exist'
        self.assertRaises(FileNotFoundError, loading.load_weights_from_resources, wrong_path)
