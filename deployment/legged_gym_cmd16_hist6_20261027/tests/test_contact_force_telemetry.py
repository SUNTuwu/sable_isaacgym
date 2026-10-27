import json
from types import SimpleNamespace
import unittest
import torch
from contact_force_analysis import analyze_contact_forces
from evaluation_contact_forces import ContactForceTelemetry


class ForceTelemetryTests(unittest.TestCase):
    def test_original_episodes_terminal_samples_and_missing_fields(self):
        scales=SimpleNamespace(contact_force_wrong=-1.,contact_force_cancellation=0.,loaded_contact_slip=-1.)
        env=SimpleNamespace(dt=.02,sim_params=SimpleNamespace(dt=.0005),contact_nominal_weight_n=8.,cfg=SimpleNamespace(rewards=SimpleNamespace(scales=scales)))
        telemetry=ContactForceTelemetry(env,[('reverse',None),('forward105',None)],1)
        state=torch.zeros(2,4,13);state[...,6]=1
        force=torch.zeros(2,4,3);force[...,2]=2.
        data=analyze_contact_forces(force,torch.zeros(2,4,3),state,torch.zeros(2,3),torch.zeros(2,2),torch.zeros(2),torch.zeros(2,4),8.,point_velocity=torch.zeros(2,4,3),resultant_force=torch.tensor([[0.,0.,8.]]).repeat(2,1),resultant_valid=torch.ones(2,dtype=torch.bool),tangential_available=False)
        data.update(point_velocity_valid=torch.ones(2,dtype=torch.bool),normal_balance_error_n=torch.zeros(2))
        snapshot=dict(contact_force_analysis=data,xyz=torch.zeros(2,3),base_quat=torch.tensor([[0.,0.,0.,1.]]).repeat(2,1))
        telemetry.update(snapshot,torch.zeros(2,16),torch.tensor([True,True]),torch.tensor([True,False]),1)
        telemetry.update(snapshot,torch.zeros(2,16),torch.tensor([False,True]),torch.tensor([False,True]),2)
        result=telemetry.finish();json.dumps(result,allow_nan=False)
        self.assertEqual(result['scenarios']['reverse']['original_control_samples'],1)
        self.assertEqual(result['scenarios']['forward105']['original_control_samples'],2)
        self.assertEqual(len(result['traces']['reverse']),1)
        self.assertTrue(result['traces']['reverse'][0]['terminal'])
        self.assertIsNone(result['traces']['reverse'][0]['tangential_force_world_n'])
        self.assertIsNone(result['scenarios']['reverse']['mean_cancelled_equivalent_force_n'])
        self.assertFalse(result['availability']['force_cancellation'])


if __name__=='__main__':unittest.main()
