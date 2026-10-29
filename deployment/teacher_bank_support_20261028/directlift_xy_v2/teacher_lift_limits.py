"""Literal foot-tip upstroke counts, including brief and airborne jitter."""
import math
import torch

RISE_M=.0005


def limit_turn_offset(previous,requested,max_delta,enabled,first_step):
    limited=previous+(requested-previous).clamp(-max_delta,max_delta)
    return torch.where((enabled&~first_step)[:,None],limited,requested)


class DirectLiftCounter:
    def __init__(self,n,device):
        self.ready=torch.zeros(n,4,device=device,dtype=torch.bool)
        self.up=torch.zeros_like(self.ready)
        self.low=torch.zeros(n,4,device=device);self.high=torch.zeros_like(self.low)
    def reset(self,ids):
        for v in (self.ready,self.up,self.low,self.high):v[ids]=0
    def update(self,z):
        self.low=torch.where(self.ready,self.low,z)
        self.high=torch.where(self.ready,self.high,z)
        self.low=torch.where(~self.up,torch.minimum(self.low,z),self.low)
        event=self.ready&~self.up&(z-self.low>=RISE_M-1e-8)
        self.up|=event
        self.high=torch.where(self.up,torch.maximum(self.high,z),z)
        down=self.up&(self.high-z>=RISE_M-1e-8)
        self.low=torch.where(down,z,self.low);self.up&=~down;self.ready[:]=True
        return event


def calibrate(raw):
    frames=raw['frames'];n=len(frames[0]['toe']);counter=DirectLiftCounter(n,'cpu')
    counts=torch.zeros(n,4)
    for frame in frames:counts+=counter.update(torch.tensor(frame['toe'])[:,:,2])
    horizon=float(raw['horizon_s']);result={}
    for i,ident in enumerate(raw['teacher_ids']):
        mean=counts[i*4:(i+1)*4].mean(0)
        cap=min(float(mean.max())/horizon/3,.5)
        total_cap=min(float(mean.sum())/horizon/3,2.)
        assert cap>0 and total_cap>0,'Cannot calibrate a zero baseline'
        result[ident]={'baseline_mean_counts_per_foot':mean.tolist(),'baseline_total_count':float(mean.sum()),
           'baseline_horizon_s':horizon,'per_foot_max_hz':cap,'total_max_hz':total_cap,
           'per_foot_min_interval_s':1/cap,'window_s':4.}
    return {'schema':'direct-tip-upstrokes/v1','source_bank':raw['bank'],'source_iteration':raw['iteration'],
        'rise_hysteresis_m':RISE_M,'teachers':result,
        'counter':'Each>=0.5mm tip rise after a>=0.5mm descent counts once. No contact, flight-duration or stride-distance prerequisite. Also counts shaking while held in air; first sample initializes height only.',
        'cap_rule':'At most baseline/3, further bounded by previous0.5Hz/foot low-cadence goal and2Hz total. Total-foot budget prevents redistributing excessive steps across legs.',
        'acceptance':'Full16s first episodes must all survive, yaw RMSE<=0.20rad/s, correct commanded signed yaw, and total upstroke count<=the same-duration baseline/3. Early failure or standing still never passes.'}


class LiftCountGuard:
    def __init__(self,n,device,dt,teacher_ids,clip_ids,config):
        self.dt=dt;self.window=4.;self.steps=int(round(self.window/dt));self.index=0
        rows=(config or {}).get('teachers',{})
        self.enabled=torch.tensor([name in rows for name in clip_ids],device=device)[teacher_ids]
        caps=torch.tensor([rows.get(name,{}).get('per_foot_max_hz',.5) for name in clip_ids],device=device)
        totals=torch.tensor([rows.get(name,{}).get('total_max_hz',2.) for name in clip_ids],device=device)
        self.cap=caps[teacher_ids];self.total_cap=totals[teacher_ids];self.min_interval=1/self.cap
        self.counter=DirectLiftCounter(n,device)
        self.history=torch.zeros(self.steps,n,4,device=device,dtype=torch.bool)
        self.count=torch.zeros(n,4,device=device);self.last=torch.full_like(self.count,-1000.)
        self.previous_z=torch.zeros_like(self.count);self.velocity=torch.zeros_like(self.count)
        self.event=torch.zeros(n,4,device=device,dtype=torch.bool)
        self.interval_failure=torch.zeros(n,device=device,dtype=torch.bool)
        self.window_failure=torch.zeros_like(self.interval_failure);self.violation=torch.zeros_like(self.interval_failure)
    def reset(self,ids):
        self.counter.reset(ids);self.history[:,ids]=False;self.count[ids]=0;self.last[ids]=-1000.
        for v in (self.previous_z,self.velocity,self.event,self.interval_failure,self.window_failure,self.violation):v[ids]=0
    def update(self,z,age):
        was_ready=self.counter.ready.clone()
        self.velocity=(z-self.previous_z)/self.dt*was_ready*self.enabled[:,None]
        self.previous_z.copy_(z)
        self.event=self.counter.update(z)&self.enabled[:,None]
        gap=age[:,None]-self.last
        self.interval_failure=(self.event&(self.last>-999)&(gap<self.min_interval[:,None]-1e-6)).any(1)
        self.count+=self.event.float()-self.history[self.index].float()
        self.history[self.index]=self.event;self.index=(self.index+1)%self.steps
        self.last=torch.where(self.event,age[:,None],self.last)
        self.window_failure=((self.count>torch.floor(self.cap[:,None]*self.window+1e-6)).any(1)|
            (self.count.sum(1)>torch.floor(self.total_cap*self.window+1e-6)))&self.enabled
        self.violation=(self.interval_failure|self.window_failure)&self.enabled
        return self.violation
    def observations(self,age):
        available=((age[:,None]-self.last)/self.min_interval[:,None]).clamp(0,1)
        budget=(1-self.count.sum(1)/(self.total_cap*self.window)).clamp(-1,1)
        return torch.cat((available,budget[:,None]),1)*self.enabled[:,None]
    def frequency_cost(self):
        hz=self.count/self.window
        per_foot=((hz/self.cap[:,None]-1).clamp(min=0).square()).mean(1)
        total=(hz.sum(1)/self.total_cap-1).clamp(min=0).square()
        return -12*torch.maximum(per_foot,total).clamp(max=4)*self.enabled


def turn_acceptance(report,config):
    limit=(config or {}).get('teachers',{}).get(report['teacher_id'])
    if limit is None:return None
    counts=report.get('quality_direct_lifts_per_foot')
    if counts is None:return dict(passed=False,reason='missing direct-lift measurements')
    observed=sum(counts);expected=limit['baseline_total_count']*report['horizon_s']/limit['baseline_horizon_s']*report['num_envs']
    ratio=observed/expected if expected else None
    complete=report['horizon_s']>=16 and report['survival_fraction']==1.
    command=report['mean_command'][2];actual=report['mean_actual'][2]
    tracking=report['yaw_rmse_rad_s']<=.20 and actual*command>0 and abs(actual)>=abs(command)*.75
    hard_budget=all(c<=limit['per_foot_max_hz']*report['horizon_s']*report['num_envs']+1e-6 for c in counts)
    hard_budget=hard_budget and observed<=limit['total_max_hz']*report['horizon_s']*report['num_envs']+1e-6
    passed=complete and tracking and hard_budget and ratio is not None and ratio<=1/3+1e-8
    return dict(passed=passed,full_horizon=complete,yaw_tracking=tracking,
        hard_budget=hard_budget,direct_lift_count=observed,same_duration_baseline_count=expected,count_ratio=ratio,maximum_ratio=1/3)
