"""Fixed-physical-metric evaluation; a flat return is never a success criterion."""
import math

def summarize(base,foot_square,count,liftoffs,slip_square,saturated,history):
    feet=[math.sqrt(max(0,x)/max(1,count)) for x in foot_square]
    cmd=base['mean_command'];speed=math.hypot(*cmd[:2])
    limits={'velocity_rmse_m_s':max(.015,.25*speed),'yaw_rmse_rad_s':max(.08,.25*abs(cmd[2])),
            'worst_foot_rmse_m':.020,'contact_slip_rmse_m_s':.03}
    slip=math.sqrt(max(0,slip_square)/max(1,count*4))
    passed=(base['survival_fraction']==1 and base['xy_rmse_m_s']<=limits['velocity_rmse_m_s']
            and base['yaw_rmse_rad_s']<=limits['yaw_rmse_rad_s'] and max(feet)<=.020 and slip<=.03 and base.get('gravity_rmse_deg',0)<=5. and base.get('contact_phase_agreement',1.)>=.90)
    errors=[base['xy_rmse_m_s']/limits['velocity_rmse_m_s'],base['yaw_rmse_rad_s']/limits['yaw_rmse_rad_s'],
            max(feet)/.020,slip/.03]
    score=base['survival_fraction']*(sum(math.exp(-min(x*x,30)) for x in errors)+2*math.exp(-min((base.get('gravity_rmse_deg',0)/5)**2,30))+base.get('contact_phase_agreement',1.))/7
    history.append(score);recent=history[-5:]
    plateau=len(recent)==5 and max(recent)-min(recent)<.01
    return {'accepted':passed,'acceptance_scope':'preliminary: four held-out envs, 16 seconds; not hardware validation',
            'foot_rmse_m_per_foot':feet,'contact_slip_rmse_m_s':slip,'actual_liftoffs_per_foot':liftoffs,
            'foot_reward_saturated_fraction':saturated/max(1,count),'physical_score':score,
            'evaluation_limits':limits,'plateau_status':('stable_and_passing' if passed else 'stalled_not_solved') if plateau else 'not_established',
            'plateau_definition':'five fixed-seed evaluations, physical score range <0.01; no automatic stopping or curriculum change'}
