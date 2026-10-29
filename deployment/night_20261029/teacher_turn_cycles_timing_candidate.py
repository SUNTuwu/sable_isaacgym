"""Full authored-cycle turn progress, drift and contact-period diagnostics."""
import math
import torch

def signed_tracking(actual,target):
    # Zero at rest, +6 at the desired rate, negative for reverse or >2x overspeed.
    # Smooth saturation avoids the narrow Gaussian's near-zero signal far away.
    return 6*torch.tanh(1-(actual-target).abs()/target.abs().clamp(min=.01))/math.tanh(1.)

DISCOUNT = .99
PROGRESS_SCALE = 6.
POTENTIAL_FLOOR = -2.


def progress_potential(progress):
    """Peak at one target cycle; moderate overspeed beats almost no rotation."""
    return (torch.minimum(progress, torch.ones_like(progress))
            - .15 * (progress - 1).clamp(min=0).square()).clamp(min=POTENTIAL_FLOOR)


class TurnCycles:
    def __init__(self,n,device,period,expected_lifts):
        self.dt=None;self.period=period;self.expected_lifts=expected_lifts
        self.previous_yaw=torch.zeros(n,device=device);self.previous_phase=torch.zeros_like(self.previous_yaw)
        self.start_xy=torch.zeros(n,2,device=device);self.net=torch.zeros_like(self.previous_yaw)
        self.travel=torch.zeros_like(self.net);self.max_drift=torch.zeros_like(self.net)
        self.phase_sum=torch.zeros_like(self.net);self.steps=torch.zeros_like(self.net)
        self.lifts=torch.zeros(n,4,device=device)
        self.ready=torch.zeros(n,device=device,dtype=torch.bool);self.armed=self.ready.clone()
        self.progress_angle=torch.zeros_like(self.net)
        self.progress_reward_rate=torch.zeros_like(self.net)
        self.diagnostics={}
    def reset(self,ids):
        for k in ('previous_yaw','previous_phase','start_xy','net','travel','max_drift','phase_sum','steps','lifts','ready','armed','progress_angle','progress_reward_rate'):
            getattr(self,k)[ids]=0
    def update(self,yaw,xy,phase,command,phase_agreement,liftoff,failed,dt,yaw_rate=None,episode_end=None,bootstrap_timeouts=True):
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
        # Dense potential differences use the very same discount as PPO.
        # Each complete reference cycle is a separate progress objective:
        # sum(gamma**t * r_t*dt) = 6*T*gamma**N*Phi(final_progress).
        # No reward is assigned to the first incomplete cycle after a reset.
        rate=delta/dt if yaw_rate is None else yaw_rate
        old_progress=self.progress_angle*command[:,2].sign()/goal.clamp(min=.01)
        old_potential=progress_potential(old_progress)
        self.progress_angle+=rate*dt*active
        new_progress=self.progress_angle*command[:,2].sign()/goal.clamp(min=.01)
        new_potential=progress_potential(new_progress)
        ended=failed if episode_end is None else (episode_end|failed)
        timeout=ended&~failed
        # A physical fall is terminal and has no value bootstrap: cancel the
        # unfinished potential. An artificial timeout keeps Phi(next), because
        # the core adds gamma*V(pre_reset_obs). Clearing it here too would charge
        # the potential twice. Finite-horizon, no-bootstrap consumers may pass
        # bootstrap_timeouts=False explicitly. A successful wrap keeps its
        # cycle payoff and clears live state before pre_reset_obs is captured.
        terminal=failed if bootstrap_timeouts else ended
        unfinished_terminal=terminal&~complete
        settled_potential=torch.where(unfinished_terminal,torch.zeros_like(new_potential),new_potential)
        self.progress_reward_rate=PROGRESS_SCALE*self.period*(DISCOUNT*settled_potential-old_potential)/dt*active
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
            angular_monotonicity=monotonicity.clone(),liftoffs=self.lifts.clone(),cadence_error=cadence_error.clone(),
            progress_fraction=new_progress.clone(),progress_potential=new_potential.clone(),
            progress_reward_rate=self.progress_reward_rate.clone(),progress_terminal_cancelled=unfinished_terminal.clone(),
            progress_timeout_continuation=(timeout&active&~complete&bootstrap_timeouts).clone(),
            progress_old_potential=old_potential.clone(),progress_settled_potential=settled_potential.clone())
        # First partial cycle after a reset never earns a cycle reward.
        begin=wrapped|~self.ready
        self.start_xy[begin]=xy[begin]
        for k in ('net','travel','max_drift','phase_sum','steps','lifts','progress_angle'):getattr(self,k)[begin]=0
        self.armed|=wrapped;self.ready[:]=True
        self.previous_yaw=yaw.clone();self.previous_phase=phase.clone()
        return event/dt

def cycle_report(count,passed,angle_error,displacement,max_drift,phase,cadence):
    n=max(1.,count)
    return dict(completed_cycles=int(count),passed_cycles=int(passed),pass_fraction=passed/n,
        mean_net_angle_error_deg=angle_error/n,mean_net_displacement_m=displacement/n,
        mean_cycle_max_drift_m=max_drift/n,mean_contact_phase_agreement=phase/n,mean_liftoff_count_error=cadence/n,
        acceptance='net angle 70-130% of commanded cycle angle; net XY<=5cm, max XY<=7.5cm; angular monotonicity>=0.8; contact phase>=85%; per-foot lift count matches reference')
