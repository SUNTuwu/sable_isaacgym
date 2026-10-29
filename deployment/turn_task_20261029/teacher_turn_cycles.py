"""Full authored-cycle turn progress, drift and contact-period diagnostics."""
import math
import torch

def signed_tracking(actual,target):
    # Zero at rest, +6 at the desired rate, negative for reverse or >2x overspeed.
    # Smooth saturation avoids the narrow Gaussian's near-zero signal far away.
    return 6*torch.tanh(1-(actual-target).abs()/target.abs().clamp(min=.01))/math.tanh(1.)

class TurnCycles:
    def __init__(self,n,device,period,expected_lifts):
        self.dt=None;self.period=period;self.expected_lifts=expected_lifts
        self.previous_yaw=torch.zeros(n,device=device);self.previous_phase=torch.zeros_like(self.previous_yaw)
        self.start_xy=torch.zeros(n,2,device=device);self.net=torch.zeros_like(self.previous_yaw)
        self.travel=torch.zeros_like(self.net);self.max_drift=torch.zeros_like(self.net)
        self.phase_sum=torch.zeros_like(self.net);self.steps=torch.zeros_like(self.net)
        self.lifts=torch.zeros(n,4,device=device)
        self.ready=torch.zeros(n,device=device,dtype=torch.bool);self.armed=self.ready.clone()
        self.diagnostics={}
    def reset(self,ids):
        for k in ('previous_yaw','previous_phase','start_xy','net','travel','max_drift','phase_sum','steps','lifts','ready','armed'):
            getattr(self,k)[ids]=0
    def update(self,yaw,xy,phase,command,phase_agreement,liftoff,failed,dt):
        turn=command[:,2].abs()>.001
        delta=torch.atan2(torch.sin(yaw-self.previous_yaw),torch.cos(yaw-self.previous_yaw))
        wrapped=(phase<self.previous_phase)&self.ready&turn
        active=self.ready&self.armed&turn
        self.net+=delta*active;self.travel+=delta.abs()*active
        displacement=(xy-self.start_xy).norm(dim=1)
        self.max_drift=torch.maximum(self.max_drift,displacement*active)
        self.phase_sum+=phase_agreement*active;self.steps+=active.float()
        self.lifts+=liftoff.float()*active[:,None]
        complete=wrapped&self.armed&~failed
        goal=command[:,2].abs()*self.period
        signed_angle=self.net*command[:,2].sign()
        progress=signed_angle/goal.clamp(min=.01)
        agreement=self.phase_sum/self.steps.clamp(min=1)
        monotonicity=self.net.abs()/self.travel.clamp(min=1e-6)
        cadence_error=(self.lifts-self.expected_lifts).abs().sum(1)
        passed=complete&(progress>=.7)&(progress<=1.3)&(displacement<=.05)&(self.max_drift<=.075)&(monotonicity>=.8)&(agreement>=.85)&(cadence_error==0)
        quality=torch.tanh(1-(progress-1).abs())/math.tanh(1.)
        # Do not attenuate a reverse-turn penalty by moving farther away.
        gate=torch.exp(-(self.max_drift/.05).square())*agreement
        event=(2*(quality.clamp(min=0)*gate+quality.clamp(max=0))-.1*cadence_error)*complete
        self.diagnostics=dict(complete=complete.clone(),passed=passed.clone(),signed_angle_deg=signed_angle*180/math.pi,
            target_angle_deg=goal*180/math.pi,net_angle_error_deg=(signed_angle-goal).abs()*180/math.pi,
            displacement_m=displacement.clone(),max_drift_m=self.max_drift.clone(),contact_agreement=agreement.clone(),
            angular_monotonicity=monotonicity.clone(),liftoffs=self.lifts.clone(),cadence_error=cadence_error.clone())
        # First partial cycle after a reset never earns a cycle reward.
        begin=wrapped|~self.ready
        self.start_xy[begin]=xy[begin]
        for k in ('net','travel','max_drift','phase_sum','steps','lifts'):getattr(self,k)[begin]=0
        self.armed|=wrapped;self.ready[:]=True
        self.previous_yaw=yaw.clone();self.previous_phase=phase.clone()
        return event/dt

def cycle_report(count,passed,angle_error,displacement,max_drift,phase,cadence):
    n=max(1.,count)
    return dict(completed_cycles=int(count),passed_cycles=int(passed),pass_fraction=passed/n,
        mean_net_angle_error_deg=angle_error/n,mean_net_displacement_m=displacement/n,
        mean_cycle_max_drift_m=max_drift/n,mean_contact_phase_agreement=phase/n,mean_liftoff_count_error=cadence/n,
        acceptance='net angle 70-130% of commanded cycle angle; net XY<=5cm, max XY<=7.5cm; angular monotonicity>=0.8; contact phase>=85%; per-foot lift count matches reference')
