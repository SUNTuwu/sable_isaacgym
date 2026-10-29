"""One landing after real flight, then one retrospectively confirmed takeoff.

Contact chatter cannot rearm the detector. A new flight requires continuous
2mm clearance for60ms and5mm horizontal foot-center travel. Takeoff angle/time
are latched at the last contact sample, never at the later confirmation.
"""
import torch

class AnkleStepEvents:
    def __init__(self,n,device,dt,period=1.,landing_deg=30.,takeoff_deg=110.,
                 clearance_m=.002,air_time_s=.060,travel_m=.005,error_scale_deg=30.,max_exponent=6.):
        self.dt=dt;self.factor=period/(4*dt);self.landing_deg=landing_deg;self.takeoff_deg=takeoff_deg
        self.clearance_m=clearance_m;self.air_time_s=air_time_s;self.travel_m=travel_m
        self.error_scale_deg=error_scale_deg;self.max_exponent=max_exponent
        for k in ['initialized','previous_contact','candidate','ready_for_landing','await_liftoff','landed','lifted','contact']:
            setattr(self,k,torch.zeros(n,2,device=device,dtype=torch.bool))
        for k in ['last_angle','last_age','candidate_angle','candidate_time','clear_run','max_clear','landing_angle_deg','takeoff_angle_deg','landing_time_s','takeoff_time_s','confirmation_time_s']:
            setattr(self,k,torch.zeros(n,2,device=device))
        for k in ['flight_id','landing_id','takeoff_pair_id','last_landing_id']:
            setattr(self,k,torch.zeros(n,2,device=device,dtype=torch.long))
        self.last_xy=torch.zeros(n,2,2,device=device);self.start_xy=torch.zeros_like(self.last_xy)
        self.event_cost=torch.zeros(n,device=device)
    def reset(self,ids):
        for v in vars(self).values():
            if torch.is_tensor(v):v[ids]=0
    def angle_cost(self,angle,target):
        # No dead band. The numerical guard lies beyond every valid0..180deg error.
        return torch.expm1(((angle-target).abs()/self.error_scale_deg).clamp(max=self.max_exponent))
    def update(self,contact,clearance,angle,xy,command,age):
        eligible=((command>.02)&(age>=1.))[:,None]
        for k in ['candidate','ready_for_landing','await_liftoff']:
            getattr(self,k).logical_and_(eligible)
        departure=self.initialized&self.previous_contact&~contact&eligible&~self.ready_for_landing
        self.candidate|=departure
        self.start_xy=torch.where(departure[:,:,None],self.last_xy,self.start_xy)
        self.candidate_angle=torch.where(departure,self.last_angle,self.candidate_angle)
        self.candidate_time=torch.where(departure,self.last_age,self.candidate_time)
        self.clear_run=torch.where(departure,torch.zeros_like(self.clear_run),self.clear_run)
        self.max_clear=torch.where(departure,torch.zeros_like(self.max_clear),self.max_clear)
        self.clear_run=torch.where(self.candidate&~contact&(clearance>=self.clearance_m),self.clear_run+self.dt,torch.zeros_like(self.clear_run))
        self.max_clear=torch.maximum(self.max_clear,self.clear_run)
        travel=(xy-self.start_xy).norm(dim=-1)
        qualified=self.candidate&eligible&(self.max_clear>=self.air_time_s-1e-7)&(travel>=self.travel_m-1e-7)
        new_flight=qualified&~self.ready_for_landing
        self.lifted=new_flight&self.await_liftoff
        self.takeoff_angle_deg=torch.where(self.lifted,self.candidate_angle,self.takeoff_angle_deg)
        self.takeoff_time_s=torch.where(self.lifted,self.candidate_time,self.takeoff_time_s)
        self.takeoff_pair_id=torch.where(self.lifted,self.last_landing_id,self.takeoff_pair_id)
        self.confirmation_time_s=torch.where(self.lifted,age[:,None],self.confirmation_time_s)
        self.flight_id+=new_flight.long();self.ready_for_landing|=new_flight;self.await_liftoff&=~self.lifted
        # Only the first contact of this qualified flight is eligible to land.
        self.landed=contact&self.ready_for_landing&eligible
        self.landing_angle_deg=torch.where(self.landed,angle,self.landing_angle_deg)
        self.landing_time_s=torch.where(self.landed,age[:,None],self.landing_time_s)
        self.landing_id=torch.where(self.landed,self.flight_id,self.landing_id)
        self.last_landing_id=torch.where(self.landed,self.flight_id,self.last_landing_id)
        self.await_liftoff|=self.landed
        self.ready_for_landing&=~contact;self.candidate&=~contact
        self.event_cost=(self.angle_cost(self.landing_angle_deg,self.landing_deg)*self.landed+
                         self.angle_cost(self.takeoff_angle_deg,self.takeoff_deg)*self.lifted).sum(1)*self.factor
        self.previous_contact.copy_(contact);self.contact.copy_(contact);self.last_angle.copy_(angle)
        self.last_xy.copy_(xy);self.last_age.copy_(age[:,None].expand_as(self.last_age));self.initialized[:]=True
