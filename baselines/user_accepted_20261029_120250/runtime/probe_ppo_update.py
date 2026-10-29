"""Measure policy changes around real PPO updates without changing the update."""
import argparse
import sys
from pathlib import Path
from isaacgym import gymapi
import torch
from rsl_rl.algorithms import PPO
from ppo_history import HistoryPPO
from run_io import atomic_json


def main():
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--diagnostic-output',type=Path,required=True)
    args,remaining=parser.parse_known_args()
    rows=[]
    original_ppo=PPO.update
    original_history=HistoryPPO.update
    sample={}

    def measurement(algorithm):
        policy=algorithm.actor_critic
        with torch.no_grad():
            raw=sample['raw'];mu=policy.act_inference(raw);sigma=policy.distribution_std(raw)
            # KL(old || new), excluding the frozen waist dimension.
            kl=((torch.log(sigma/sample['sigma'])+
                (sample['sigma'].square()+(sample['mu']-mu).square())/(2*sigma.square())-.5)*policy.execution_mask).sum(-1)
            logp=(torch.distributions.Normal(mu,sigma).log_prob(sample['actions'])*policy.execution_mask).sum(-1)
            ratio=(logp-sample['logp']).exp()
            vx,vy,yaw=raw[:,306]*.2,raw[:,307]*.1,raw[:,308]
            novel=(vx<.03)&((torch.sqrt(vx.square()+vy.square())>.01)|(yaw.abs()>.08))
            out={}
            for name,mask in [('all',torch.ones_like(novel)),('novel',novel),('known',~novel)]:
                if not mask.any():continue
                out[name]=dict(count=int(mask.sum()),kl_mean=float(kl[mask].mean()),
                    kl_p95=float(torch.quantile(kl[mask],.95)),
                    likelihood_ratio_outside_clip_fraction=float(((ratio[mask]-1).abs()>algorithm.clip_param).float().mean()))
            return out

    def ppo_update(algorithm):
        storage=algorithm.storage
        count=storage.observations.shape[0]*storage.observations.shape[1]
        # Deterministic coverage of time and environment; no RNG draws.
        ids=torch.linspace(0,count-1,min(4096,count),device=algorithm.device).long()
        for key,source in [('raw',storage.observations),('mu',storage.mu),('sigma',storage.sigma),
                           ('actions',storage.actions),('logp',storage.actions_log_prob)]:
            sample[key]=source.flatten(0,1)[ids].clone()
        sample['logp']=sample['logp'].flatten()
        result=original_ppo(algorithm)
        rows.append(dict(update=len(rows)+1,learning_rate=algorithm.learning_rate,
            after_ppo=measurement(algorithm)))
        return result

    def history_update(algorithm):
        result=original_history(algorithm)
        rows[-1]['after_auxiliary_and_std_clamp']=measurement(algorithm)
        rows[-1]['auxiliary']=dict(algorithm.last_auxiliary_metrics)
        atomic_json(args.diagnostic_output,dict(rows=rows,scope='Stochastic training update diagnostic; no independent task acceptance'))
        return result

    PPO.update=ppo_update
    HistoryPPO.update=history_update
    sys.argv=[sys.argv[0]]+remaining
    import train_ppo
    train_ppo.main()


if __name__=='__main__':main()
