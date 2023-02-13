import unittest
import sys
sys.path.insert(0, '../')
from common.utilities.project import *
from common.utilities.helper import *

class TestProject(unittest.TestCase):

    def setUp(self):
        self.command_line_arguments = get_arguments()
        self.PROJECT_NAME = "tests"
        self.command_line_arguments['project_name'] = self.PROJECT_NAME

    def test_init_project(self):
        self.command_line_arguments["task"] = SAFE_TRAINING_TASK
        m_project = Project(self.command_line_arguments)
        self.assertTrue(m_project.command_line_arguments['project_name'] == self.PROJECT_NAME)
        self.assertTrue(m_project.command_line_arguments['task'] == SAFE_TRAINING_TASK)
        self.assertTrue(m_project.agent == None)
        self.assertTrue(m_project.mlflow_bridge == None)



