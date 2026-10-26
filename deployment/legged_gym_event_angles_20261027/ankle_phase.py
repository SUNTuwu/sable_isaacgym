"""Phase-dependent ankle rotation shortfall cost; no positive motion reward."""
import math,torch
class PhaseAnkleMotion:
    def __init__(self,n,device,dt,period=1.,stance_fraction=.5,landing_deg=30.,takeoff_deg=110.,lag_tolerance_deg=5.,lag_scale_deg=30.,lag_max_cost=16.):
        self.lag_tolerance_deg=lag_tolerance_deg;self.lag_scale_deg=lag_scale_deg;self.lag_max_cost=lag_max_cost
        self.dt=dt;self.period=period;self.stance_duration=period*stance_fraction;self.swing_duration=period*(1-stance_fraction)
        self.landing_deg=landing_deg;self.takeoff_deg=takeoff_deg;self.range_deg=takeoff_deg-landing_deg;self.factor=period/(4*dt)
        for key in ['last_q','last_angle','start_q','start_angle','required_progress_deg','progress_deg','lag_deg','air','maxair','stance_age','swing_age','target_deg','dq_deg']:
            setattr(self,key,torch.zeros(n,2,device=device))
        for key in ['initialized','previous','swing_armed','stance_active','landed','lifted','valid','contact']:
            setattr(self,key,torch.zeros(n,2,device=device,dtype=torch.bool))
        for key in ['lag_cost','angle_cost','event_cost','reverse_cost']:setattr(self,key,torch.zeros(n,device=device))
    def reset(self,ids):
        for value in vars(self).values():
            if torch.is_tensor(value):value[ids]=0
    def update(self,q,contact,clearance,angle,command,vx,age):
        self.takeoff_angle_deg=self.last_angle.clone()
        dq=(q-self.last_q)*180/math.pi;dq=torch.where(self.initialized,dq,torch.zeros_like(dq));self.dq_deg=dq
        departing=self.previous&~contact&self.initialized
        arriving=~self.previous&contact&self.initialized
        self.lifted=departing&self.stance_active&(self.stance_age>=.08-1e-7)
        self.swing_armed=torch.where(departing,(self.stance_age>=.08-1e-7)&(age[:,None]>=1),self.swing_armed)
        self.maxair=torch.where(departing,torch.zeros_like(self.maxair),self.maxair)
        self.air=torch.where(~contact&(clearance>=.002),self.air+self.dt,torch.zeros_like(self.air));self.maxair=torch.maximum(self.maxair,self.air)
        self.landed=arriving&self.swing_armed&(self.maxair>=.06-1e-7)
        self.stance_active=(self.stance_active&contact)|self.landed
        edge=departing|arriving|~self.initialized
        self.start_q=torch.where(edge,torch.where(departing,self.last_q,q),self.start_q);self.start_angle=torch.where(edge,torch.where(departing,self.last_angle,angle),self.start_angle)
        self.stance_age=torch.where(contact,self.stance_age+self.dt,torch.zeros_like(self.stance_age));self.swing_age=torch.where(~contact,self.swing_age+self.dt,torch.zeros_like(self.swing_age))
        stance_progress=(self.stance_age/self.stance_duration).clamp(0,1);swing_progress=(self.swing_age/self.swing_duration).clamp(0,1)
        self.target_deg=torch.where(contact,self.landing_deg+self.range_deg*stance_progress,self.takeoff_deg-self.range_deg*swing_progress)
        movement=(command>.02)&(vx>.005)&(age>=1.)
        sign=torch.where(contact,-torch.ones_like(q),torch.ones_like(q))
        joint_progress=(q-self.start_q)*180/math.pi*sign
        ground_progress=torch.where(contact,angle.clamp(max=self.takeoff_deg)-self.start_angle,self.start_angle-angle.clamp(min=self.landing_deg))
        progress=torch.minimum(joint_progress,ground_progress).clamp(0,self.range_deg)
        self.valid=torch.where(contact,self.stance_active&(self.stance_age>=.08-1e-7),self.swing_armed&(self.maxair>=.06-1e-7))
        # Requested motion starts at the actual phase-boundary foot angle. Taking
        # the lesser of joint and ground-angle travel prevents whole-leg motion
        # from masquerading as ankle articulation. No positive credit is paid.
        self.required_progress_deg=torch.where(contact,self.target_deg-self.start_angle,self.start_angle-self.target_deg).clamp(0,self.range_deg)
        self.progress_deg=progress
        self.lag_deg=(self.required_progress_deg-progress).clamp(min=0.)
        # A commanded robot that stalls or drags must still incur tracking cost;
        # do not gate this on achieved forward speed or successful flight.
        commanded=(command>.02)&(age>=1.)
        self.lag_cost=(((self.lag_deg-self.lag_tolerance_deg).clamp(min=0.)/self.lag_scale_deg).square().clamp(max=self.lag_max_cost)).mean(1)*commanded
        self.reverse_cost=(((-dq*sign).clamp(min=0.)/self.range_deg)*self.valid*movement[:,None]*~edge).sum(1)*self.factor
        commanded=(command>.02)&(age>=1.)
        self.angle_cost=((((angle-self.target_deg).abs()-5.).clamp(min=0.)/30.).square().clamp(max=16)).mean(1)*commanded
        landing_error=((angle-self.landing_deg).abs()-10.).clamp(min=0.)/30.
        takeoff_error=((self.last_angle-self.takeoff_deg).abs()-10.).clamp(min=0.)/30.
        self.event_cost=(landing_error.square()*self.landed+takeoff_error.square()*self.lifted).sum(1)*self.factor*commanded
        self.swing_armed&=~arriving;self.previous.copy_(contact);self.contact.copy_(contact);self.last_q.copy_(q);self.last_angle.copy_(angle);self.initialized[:]=True
