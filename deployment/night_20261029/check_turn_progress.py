"""CPU checks of discounted cycle progress, plus exact saved-cycle evidence."""
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
import torch

HERE=Path(__file__).resolve().parent


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


new=load('new_cycles',HERE/'teacher_turn_cycles_reviewed.py')
old=load('old_cycles',HERE.parent/'turn_task_20261029/teacher_turn_cycles.py')
sys.modules['teacher_turn_cycles']=new
before=load('old_rewards',HERE/'audited_rewards.py')
after=load('new_rewards',HERE/'audited_rewards_turnprogress.py')
os.environ.setdefault('SABLE_GYM_ROOT',str(HERE))
night=load('night',HERE/'night_tuning_reviewed.py')


class ProgressTests(unittest.TestCase):
    def make(self,direction=1,n=1):
        c=new.TurnCycles(n,'cpu',torch.ones(n),torch.ones(n,4))
        self.command=torch.tensor([[0.,0,direction*math.pi/6]]*n)
        self.yaw=torch.zeros(n);self.n=n
        for phase in (.25,.75,0.):self.tick(c,0,phase)
        return c

    def tick(self,c,increment,phase,failed=False,terminal=False):
        increment=torch.as_tensor(increment).expand(self.n).float()
        rate=increment*self.command[:,2]/.02
        self.yaw+=rate*.02
        result=c.update(self.yaw,torch.zeros(self.n,2),torch.full((self.n,),float(phase)),
            self.command,torch.ones(self.n),torch.zeros(self.n,4,dtype=torch.bool),
            torch.full((self.n,),failed,dtype=torch.bool),.02,yaw_rate=rate,
            episode_end=torch.full((self.n,),terminal,dtype=torch.bool))
        return c.progress_reward_rate.clone()*.02,result*.02

    def sequence(self,progress,direction=1,terminal_index=None,fail=False):
        c=self.make(direction);previous=0.;rewards=[]
        for i,value in enumerate(progress,1):
            terminal=i==terminal_index
            reward,_=self.tick(c,value-previous,(i%50)/50,failed=terminal and fail,terminal=terminal)
            rewards.append(float(reward));previous=value
        return c,rewards,sum(.99**i*r for i,r in enumerate(rewards))

    def test_discounted_telescoping_and_target_peak(self):
        for direction in [1,-1]:
            values={}
            for endpoint in [-100,-1,0,.2,1,1.8,10]:
                _,rewards,total=self.sequence([endpoint*i/50 for i in range(1,51)],direction)
                expected=6*.99**50*float(new.progress_potential(torch.tensor(endpoint)))
                self.assertAlmostEqual(total,expected,places=4)
                self.assertTrue(all(math.isfinite(r) for r in rewards))
                values[endpoint]=total
            self.assertGreater(values[1],values[1.8]);self.assertGreater(values[1.8],values[.2])
            self.assertEqual(values[0],0)

    def test_closed_excursions_have_zero_discounted_return(self):
        for sign in [1,-1]:
            for wait in [0,15]:
                path=[0.]*wait+[sign*.8]*(25-wait)+[0.]*25
                _,_,value=self.sequence(path)
                self.assertAlmostEqual(value,0.,places=5)
        # Bare Delta-Phi would incorrectly reward this early-positive / late-negative excursion.
        naive=6*.8*(1-.99**25)
        self.assertGreater(naive,1.)

    def test_constant_potential_hold_pays_discount_correction(self):
        c=self.make();self.tick(c,.5,.02)
        reward,_=self.tick(c,0.,.04)
        self.assertAlmostEqual(float(reward),6*(.99*.5-.5),places=5)

    def test_wrap_does_not_charge_artificial_reset_transition(self):
        c=self.make();cycles=[]
        for cycle in range(2):
            total=0.
            for i in range(1,51):
                reward,_=self.tick(c,.02,(i%50)/50);total+=.99**(i-1)*float(reward)
            cycles.append(total)
            self.assertAlmostEqual(float(c.progress_angle),0.,places=7)
        self.assertAlmostEqual(cycles[0],cycles[1],places=5)

    def test_initial_partial_cycle_and_reset_are_unrewarded(self):
        self.n=1;self.command=torch.tensor([[0.,0,math.pi/6]]);self.yaw=torch.zeros(1)
        c=new.TurnCycles(1,'cpu',torch.ones(1),torch.ones(1,4))
        for phase in [.25,.5,.75,0.]:
            reward,_=self.tick(c,.5,phase);self.assertEqual(float(reward),0.)
        self.tick(c,.1,.02);c.reset(torch.tensor([0]))
        self.assertEqual(float(c.progress_angle),0.);self.assertEqual(float(c.progress_reward_rate),0.)
        reward,_=self.tick(c,.5,.25);self.assertEqual(float(reward),0.)

    def test_failed_and_partial_timeout_cancel_shaping(self):
        for direction in [1,-1]:
            for fail in [False,True]:
                c,rewards,value=self.sequence([direction*i/50 for i in range(1,21)],terminal_index=20,fail=fail)
                self.assertAlmostEqual(value,0.,places=5)
                self.assertTrue(bool(c.diagnostics['progress_terminal_cancelled'][0]))
                self.assertFalse(bool(c.diagnostics['complete'][0]))
        # Failure exactly at a would-be wrap must not receive a completed-cycle payoff.
        c,rewards,value=self.sequence([i/50 for i in range(1,51)],terminal_index=50,fail=True)
        self.assertAlmostEqual(value,0.,places=5)
        self.assertFalse(bool(c.diagnostics['complete'][0]))
        # A normal timeout coincident with a successful wrap retains the complete cycle.
        _,_,value=self.sequence([i/50 for i in range(1,51)],terminal_index=50,fail=False)
        self.assertAlmostEqual(value,6*.99**50,places=4)

    def test_existing_cycle_auxiliary_and_six_translation_terms_unchanged(self):
        torch.manual_seed(71);n=8;period=torch.ones(n)
        a=old.TurnCycles(n,'cpu',period,torch.ones(n,4));b=new.TurnCycles(n,'cpu',period,torch.ones(n,4))
        command=torch.tensor([[.075,0,0],[-.075,0,0],[0,.038,0],[0,-.038,0],[.075,0,0],[-.075,0,0],[0,0,.5236],[0,0,-.5236]])
        symmetric=torch.tensor([1,1,0,0,1,1,0,0],dtype=torch.bool)
        ra=before.ReferenceRewards(n,'cpu',.02,period,torch.ones(n,4)*.3,symmetric)
        rb=after.ReferenceRewards(n,'cpu',.02,period,torch.ones(n,4)*.3,symmetric)
        for tick in range(151):
            phase=torch.full((n,),(tick%50)/50);yaw=torch.randn(n);xy=torch.randn(n,2)*.03
            lifts=torch.rand(n,4)>.85;fall=torch.rand(n)>.97;agreement=torch.rand(n)
            args=(yaw,xy,phase,command,agreement,lifts,fall,.02)
            self.assertTrue(torch.equal(a.update(*args),b.update(*args,yaw_rate=torch.randn(n))))
            self.assertTrue(torch.equal(b.progress_reward_rate[:6],torch.zeros(6)))
            s=dict(normal_force=torch.rand(n,4)*3,clearance=torch.rand(n,4)*.02,phase=phase,command=command,
                velocity=torch.randn(n,3),foot_local=torch.randn(n,4,3)*.02,reference_local=torch.zeros(n,4,3),
                foot_sigma=torch.tensor([[.04,.04,.02]]*n),velocity_sigma=torch.ones(n)*.03,yaw_sigma=torch.ones(n)*.1,
                target=torch.randn(n,15)*.05,contact_probability=torch.rand(n,4),contact_velocity=torch.randn(n,4,3)*.04,
                gravity=torch.tensor([[0.,0,-1]]*n),reference_gravity=torch.tensor([[0.,0,-1]]*n),height_error=torch.zeros(n),
                limit_excess=torch.zeros(n,15),joint_acceleration=torch.randn(n,15),normalized_torque=torch.randn(n,15)*.1,fall=fall)
            left=ra.update(s);right=rb.update(s)
            right['v2_yaw']=torch.where(command[:,2].abs()>.001,b.progress_reward_rate,right['v2_yaw'])
            for key in left:self.assertTrue(torch.equal(left[key][:6],right[key][:6]),key)

    def test_frozen_989_vs_1038_cycle_ordering(self):
        evidence=json.loads((HERE/'turn_progress_evidence.json').read_text())
        rows={}
        for key,item in evidence.items():
            angle=item['evaluation']['turn_cycles']['mean_signed_angle_deg']
            # Evaluated mean cycle angle: illustrative Phi(mean p), not mean Phi(p).
            p=angle/30.;phi=float(new.progress_potential(torch.tensor(p)))
            sparse_yaw=[]
            for frame in item['frames']:
                if frame['active']:
                    w=frame['actual_velocity'][2];target=-math.pi/6
                    sparse_yaw.append(.4*6*math.tanh(1-abs(w-target)/abs(target))/math.tanh(1))
            rows[key]=dict(mean_cycle_angle_deg=angle,phi_of_mean_progress=phi,
                illustrative_discounted_cycle_return=.4*6*.99**50*phi,
                old_sparse_yaw_rate=sum(sparse_yaw)/len(sparse_yaw))
        self.assertLess(rows['989']['old_sparse_yaw_rate'],rows['1038']['old_sparse_yaw_rate'])
        self.assertGreater(rows['989']['illustrative_discounted_cycle_return'],rows['1038']['illustrative_discounted_cycle_return'])
        print('Frozen-trace evidence (Phi(mean p) illustration, not full reward A/B):',json.dumps(rows,sort_keys=True))

    def test_fall_dominates_bounded_terminal_potential_refund(self):
        # Both deployed source-trot turns have T=1s. With Phi >= -2, the
        # largest terminal refund is 12 unscaled, or 4.8 at the frozen x0.4.
        names=['04_TURN_LEFT','05_TURN_RIGHT']
        keys=['v2_yaw','v2_phase_symmetry','v2_velocity','v2_cross_motion']
        env=SimpleNamespace(dt=.02,teacher_ids=torch.arange(2),root_states=torch.zeros(2,13),
            cfg=SimpleNamespace(teacher_bank={'clips':[{'id':x} for x in names]}),
            night_control={'teachers':{x:{'multipliers':{k:.4 for k in keys}} for x in names}})
        terms={'v2_yaw':torch.tensor([12./.02,-6./.02]),
               'v2_phase_symmetry':torch.zeros(2),'v2_fall_event':torch.zeros(2),
               'v2_cross_motion':torch.zeros(2),'v2_velocity':torch.zeros(2)}
        result=night.apply(env,terms,torch.ones(2,dtype=torch.bool))
        costs=sum((-v).clamp(min=0) for k,v in result.items() if k!='v2_fall_event')*.02
        self.assertTrue(torch.allclose(-result['v2_fall_event']*.02,20+costs))
        self.assertTrue(bool(((sum(result.values())*.02)<-15).all()))


if __name__=='__main__':unittest.main(verbosity=2)
