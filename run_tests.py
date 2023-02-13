import unittest

# Run all tests in the 'tests' subfolder
test_suite = unittest.defaultTestLoader.discover('tests')
unittest.TextTestRunner().run(test_suite)
