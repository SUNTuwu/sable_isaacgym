"""Bounded remote comparison of zero, deterministic and exploratory actions."""
import argparse,sys,json,types
from pathlib import Path
from isaacgym import gymapi
import torch,numpy as np
p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--runtime',type=Path);p.add_argument('--noise',type=float,default=.35);a=p.parse_args()
sys.path.insert(0,str(a.runtime or a.run/'runtime'))
from preflight import create_env
from sable_config import SableCfg
from rsl_rl.modules import ActorCritic
from run_io import atomic_json,read_json
torch.manual_seed(108);np.random.seed(108)
env=create_env(96,noise=False);env.reset_noise=0.;env.fixed_command=.08
cfg=read_json(a.run/'config.json');actor=ActorCritic(48,48,12,**cfg['train']['policy']).to('cuda:0')
actor.load_state_dict(torch.load(str(a.run/'checkpoints/model_000000.pt'))['model_state_dict']);actor.eval();actor.std.data[:]=a.noise
env.reset();env.last_episode_records=[];obs=env.get_observations();counts=np.zeros(3);components={};samples=[];reasons={}
original_termination=env.check_termination
def termination():
    original_termination()
    checks={'tilt':(env.projected_gravity*env.reference_gravity).sum(1)<.55,'low':env.root_states[:,2]<.055,
        'body_contact':torch.any(torch.linalg.norm(env.contact_forces[:,env.termination_contact_indices],dim=-1)>2,dim=1),
        'high_speed':env.dof_vel.abs().max(1).values>30}
    for key,value in checks.items():reasons[key]=reasons.get(key,np.zeros(3))+value.view(3,32).sum(1).cpu().numpy()
env.check_termination=termination
original=env.compute_reward
def reward():
    before={k:v.clone() for k,v in env.episode_sums.items()};original()
    for k,v in env.episode_sums.items():
        value=(v-before[k]).view(3,32).mean(1).cpu().numpy()
        components[k]=components.get(k,np.zeros(3))+value
env.compute_reward=reward
for k in range(100):
    with torch.no_grad():
        actions=actor.act_inference(obs);actions[:32]=0.;actions[64:]+=torch.randn_like(actions[64:])*actor.std
        obs,_,r,done,_=env.step(actions)
        counts+=done.view(3,32).sum(1).cpu().numpy()
        if k%25==0:samples.append({'time':k*env.dt,'height':env.root_states[:,2].view(3,32).mean(1).tolist(),
            'reward':r.view(3,32).mean(1).tolist(),'max_contact':env.contact_forces.norm(dim=2).view(3,-1).max(1).values.tolist(),
            'rms_action':actions.square().mean(1).sqrt().view(3,32).mean(1).tolist()})
report={'groups':['zero','initial_deterministic','initial_exploration'],'noise_std':a.noise,'falls':counts.tolist(),'termination_reasons':{k:v.tolist() for k,v in reasons.items()},
    'components_sum_mean_per_env':{k:v.tolist() for k,v in components.items()},'samples':samples}
atomic_json(a.run/('initial_policy_diagnostic_'+str(a.noise)+'.json'),report);print('DIAGNOSTIC',json.dumps(report),flush=True)
env.gym.destroy_sim(env.sim)
