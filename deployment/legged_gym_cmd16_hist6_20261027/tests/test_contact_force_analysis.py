import sys
from pathlib import Path
import unittest
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from contact_force_analysis import analyze_contact_forces,material_point_velocity,material_point_velocity_from_poses


class ContactForces(unittest.TestCase):
    def fixture(self):
        point=torch.tensor([[[.08,.04,0.],[.08,-.04,0.],[-.06,.04,0.],[-.06,-.04,0.]]])
        state=torch.zeros(1,4,13);state[...,:3]=point;state[...,6]=1
        force=torch.zeros(1,4,3);force[...,2]=2.
        return force,point,state

    def result(self,force,point,state,linear=(.1,0.),yaw=0.):
        return analyze_contact_forces(force,point,state,torch.zeros(1,3),
            torch.tensor([linear]),torch.tensor([yaw]),torch.zeros(1,4),8.)

    def test_support_and_translation(self):
        f,p,s=self.fixture();r=self.result(f,p,s)
        self.assertEqual(float(r['wrong_cost']+r['cancellation_cost']+r['loaded_slip_cost']),0.)
        f[...,0]=1.;r=self.result(f,p,s);self.assertEqual(float(r['wrong_cost']+r['cancellation_cost']),0.)
        f[...,0]=-1.;r=self.result(f,p,s);self.assertGreater(float(r['wrong_cost']),0.)
        f[:,::2,0]=1.;r=self.result(f,p,s)
        self.assertEqual(float(r['wrong_cost']),0.);self.assertGreater(float(r['cancellation_cost']),0.)

    def test_correct_yaw_couple_is_not_cancellation(self):
        f,p,s=self.fixture();f[...,:2]=torch.stack((-p[...,1],p[...,0]),-1)*10
        r=self.result(f,p,s,linear=(0.,0.),yaw=.3)
        self.assertGreater(float(r['sum_yaw_moment_nm']),0.)
        self.assertEqual(float(r['wrong_cost']+r['cancellation_cost']),0.)
        f[...,:2]*=-1.;r=self.result(f,p,s,linear=(0.,0.),yaw=.3)
        self.assertGreater(float(r['wrong_cost']),0.)

    def test_braking_and_steady_tracking(self):
        f,p,s=self.fixture();f[...,0]=-1.
        r=self.result(f,p,s,linear=(-.1,0.));self.assertEqual(float(r['wrong_cost']),0.)
        r=self.result(f,p,s,linear=(.005,0.),yaw=.02)
        self.assertEqual(float(r['wrong_cost']+r['cancellation_cost']),0.)

    def test_rolling_and_loaded_sliding(self):
        f,p,s=self.fixture();s[...,2]=.005;s[...,7]=.05;s[...,11]=10.
        self.assertTrue(torch.allclose(material_point_velocity(s,p),torch.zeros(1,4,3),atol=1e-7))
        r=self.result(f,p,s);self.assertEqual(float(r['loaded_slip_cost']),0.)
        s[...,11]=0.;r=self.result(f,p,s);self.assertGreater(float(r['loaded_slip_cost']),0.)
        f[...,2]=0.;r=self.result(f,p,s)
        self.assertEqual(float(r['loaded_slip_cost']+r['wrong_cost']+r['cancellation_cost']),0.)

    def test_bounded_and_off_ground(self):
        f,p,s=self.fixture();f[...,0]=-1e6;s[...,7]=1e6
        r=self.result(f,p,s)
        self.assertLessEqual(float(r['wrong_cost']),2.);self.assertLessEqual(float(r['loaded_slip_cost']),4.)
        r=analyze_contact_forces(f,p,s,torch.zeros(1,3),torch.tensor([[.1,0.]]),
            torch.zeros(1),torch.full((1,4),.02),8.)
        self.assertEqual(float(r['wrong_cost']+r['loaded_slip_cost']),0.)

    def test_same_material_point_pose_difference(self):
        f,p,s=self.fixture();dt=.0005
        after=s[...,:7].clone();after[...,2]=.005
        before=after.clone();before[...,0]-=.05*dt
        angle=-10*dt;before[...,4]=torch.sin(torch.tensor(angle/2));before[...,6]=torch.cos(torch.tensor(angle/2))
        # Rolling sphere: center moves5cm/s, radius5mm, angular rate10rad/s.
        velocity=material_point_velocity_from_poses(before,after,p,dt)
        self.assertLess(float(velocity[...,:2].norm(dim=-1).max()),1e-5)
        before[...,4]=0.;before[...,6]=1
        velocity=material_point_velocity_from_poses(before,after,p,dt)
        self.assertTrue(torch.allclose(velocity[...,0],torch.full((1,4),.05),atol=1e-5))

    def test_normal_only_backend_does_not_invent_cancellation(self):
        f,p,s=self.fixture();kwargs=dict(point_velocity=torch.zeros(1,4,3),
            resultant_force=torch.tensor([[-4.,0.,8.]]),resultant_valid=torch.tensor([True]),tangential_available=False)
        r=analyze_contact_forces(f,p,s,torch.zeros(1,3),torch.tensor([[.1,0.]]),torch.zeros(1),torch.zeros(1,4),8.,**kwargs)
        self.assertAlmostEqual(float(r['wrong_cost']),.5,places=6)
        self.assertFalse(bool(r['tangential_available'][0]));self.assertEqual(float(r['cancellation_cost']),0.)
        kwargs['resultant_valid'][:]=False
        r=analyze_contact_forces(f,p,s,torch.zeros(1,3),torch.tensor([[.1,0.]]),torch.zeros(1),torch.zeros(1,4),8.,**kwargs)
        self.assertEqual(float(r['wrong_cost']),0.)

    def test_com_reference_and_large_environment_offset(self):
        f,p,s=self.fixture();s[...,2]=.005;s[...,7]=.15;s[...,11]=10.
        com=torch.tensor([[0.,0.,.01]]).expand(4,-1)
        # COM height15mm, bottom0; vCOM=.15 and omegaY=10 give zero slip.
        self.assertLess(float(material_point_velocity(s,p,com).norm(dim=-1).max()),1e-6)
        s[...,:2]+=100.;p[...,:2]+=100.
        self.assertLess(float(material_point_velocity(s,p,com).norm(dim=-1).max()),1e-6)


if __name__=='__main__':unittest.main()
