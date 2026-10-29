import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from curriculum import command_frontier


class CommandFrontierTests(unittest.TestCase):
    def test_full_goal_identity_and_old_regression(self):
        goals=torch.tensor([[.09,0,0],[.12,0,0],[0,0,0],[-.04,0,0],[0,.04,0],[0,0,.35]])
        torch.testing.assert_close(command_frontier(goals,1.),goals,rtol=0,atol=0)
        torch.testing.assert_close(command_frontier(goals,.25)[:3],goals[:3],rtol=0,atol=0)
        expected=torch.tensor([[.06875,0,0],[.07875,.01,0],[.07875,0,.0875]])
        torch.testing.assert_close(command_frontier(goals,.25)[3:],expected)

    def test_mirror_and_no_input_mutation(self):
        goals=torch.tensor([[-.04,.02,.3],[0,-.04,0],[0,0,-.35]])
        saved=goals.clone();sign=torch.tensor([1.,-1.,-1.])
        torch.testing.assert_close(command_frontier(goals*sign,.5),command_frontier(goals,.5)*sign)
        torch.testing.assert_close(goals,saved,rtol=0,atol=0)
        for alpha in [-.1,1.1]:
            with self.assertRaises(ValueError):command_frontier(goals,alpha)


if __name__=='__main__':unittest.main()
