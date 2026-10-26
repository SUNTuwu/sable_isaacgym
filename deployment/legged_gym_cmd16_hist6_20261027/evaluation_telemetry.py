"""Pre-reset50Hz diagnostics; mechanical power and saturation are sampled proxies."""
import math
import torch


def distribution(values):
    values=values.flatten()
    if not values.numel():return {'count':0,'mean':None,'p95':None,'p99':None,'max':None}
    return {'count':values.numel(),'mean':float(values.mean()),'p95':float(torch.quantile(values,.95)),
            'p99':float(torch.quantile(values,.99)),'max':float(values.max())}


class EvaluationTelemetry:
    pose_names=('shoulder_height_m','hip_height_m','chest_roll_rad','waist_roll_rad','stance_width_m','waist_pitch_rad')

    def __init__(self, env):
        self.env=env
        self.values=[]
        self.valid=[]
        self.alive=[]
        self.motion=[]

    def update(self,snapshot,commands,alive,valid):
        e=self.env;s=snapshot
        q=s['q'];v=s['joint_velocity'][:,e.active];tau=s['motor_torque'][:,e.active]
        a=s['joint_acceleration'][:,e.active]
        heights=s['body_xyz'][:,e.height_body_indices,2].mean(-1)
        x,y,z,w=s['base_quat'].unbind(-1)
        roll=torch.atan2(2*(y*z+w*x),1-2*(x*x+y*y))
        forward=torch.stack((1-2*(y*y+z*z),2*(x*y+w*z)),1)
        forward/=forward.norm(dim=-1,keepdim=True).clamp(min=1e-6)
        left=torch.stack((-forward[:,1],forward[:,0]),1)
        widths=((s['foot_centers'][:,[0,2],:2]-s['foot_centers'][:,[1,3],:2])*left[:,None]).sum(-1).mean(-1)
        pose=torch.cat((heights,roll[:,None],q[:,e.dof_names.index('spine_roll'):e.dof_names.index('spine_roll')+1],
                        widths[:,None],q[:,e.waist:e.waist+1]),1)
        pose_error=pose-commands[:,[3,4,11,12,13,14]]
        contact=s['force']>1.
        stance_speed=s['foot_link_origin_velocity_xy'].norm(dim=-1)
        desired=e.p_gains[e.active]*(s['executed_target']-q[:,e.active])-e.d_gains[e.active]*v
        rpm=v.abs()*60/(2*math.pi)
        big=torch.where(rpm<=245.1,1.95-(1.95-.78)*rpm/245.1,.78*(421.3-rpm)/(421.3-245.1)).clamp(min=0)
        small=e.torque_limits[e.active]*(1-v.abs()/e.max_speed[e.active]).clamp(min=0)
        drive=torch.where(e.large[e.active],big,small)
        available=torch.where(desired*v<0,e.torque_limits[e.active],drive)
        saturation=(desired.abs()>available+1e-5).float()
        speed_saturation=(v.abs()>=e.max_speed[e.active]*.95).float()
        power=tau*v
        # Fixed shapes avoid a host synchronization for every alive mask.
        self.values.append({'pose':pose,'pose_error':pose_error,'contact':contact.float(),
            'stance_speed':stance_speed,'velocity':v,'acceleration':a,'torque':tau,
            'torque_demand_saturated':saturation,'speed_saturated':speed_saturation,
            'positive_power':power.clamp(min=0).sum(-1),'signed_power':power.sum(-1),
            'coupling':s['coupling'],'waist_acceleration':s['waist_vertical_acceleration_m_s2']})
        self.valid.append(valid.clone());self.alive.append(alive.clone())

    def finish(self,groups,envs_per_scenario):
        if not self.values:return {}
        data={key:torch.stack([row[key]for row in self.values])for key in self.values[0]}
        valid=torch.stack(self.valid);alive=torch.stack(self.alive)
        reports={}
        for group,(name,_) in enumerate(groups):
            sl=slice(group*envs_per_scenario,(group+1)*envs_per_scenario)
            mask=valid[:,sl];original=alive[:,sl]
            selected={key:value[:,sl][mask]for key,value in data.items()}
            count=int(mask.sum())
            pose={}
            for i,key in enumerate(self.pose_names):
                pose[key]={'mean_actual':float(selected['pose'][:,i].mean()) if count else None,
                    'rmse':float(selected['pose_error'][:,i].square().mean().sqrt()) if count else None}
            contact=selected['contact'].bool()
            energy=(data['positive_power'][:,sl]*original).sum(0)*self.env.dt
            signed_energy=(data['signed_power'][:,sl]*original).sum(0)*self.env.dt
            reports[name]={'steady_samples':count,'pose':pose,
                'actual_contact_fraction_per_foot':selected['contact'].mean(0).cpu().tolist() if count else None,
                'foot_link_origin_stance_speed_m_s':distribution(selected['stance_speed'][contact]),
                'joint_abs_velocity_rad_s':distribution(selected['velocity'].abs()),
                'joint_abs_acceleration_rad_s2':distribution(selected['acceleration'].abs()),
                'motor_abs_torque_nm':distribution(selected['torque'].abs()),
                'endpoint_torque_demand_saturation_fraction':float(selected['torque_demand_saturated'].mean()) if count else None,
                'endpoint_speed95_fraction':float(selected['speed_saturated'].mean()) if count else None,
                'positive_mechanical_power_w':distribution(selected['positive_power']),
                'positive_mechanical_energy_j_per_original_episode':energy.cpu().tolist(),
                'signed_mechanical_energy_j_per_original_episode':signed_energy.cpu().tolist(),
                'opposed_hip_coupling_error_rad':distribution(selected['coupling']),
                'waist_abs_vertical_acceleration_m_s2':distribution(selected['waist_acceleration'].abs())}
        return {'control_sample_period_s':self.env.dt,'scenarios':reports,
            'joint_order':[self.env.dof_names[int(i)]for i in self.env.active],
            'scope':'Original episodes only, pre-reset50Hz endpoint samples. Pose/distribution metrics exclude startup and0.75s command settling; energy includes all original samples. Foot link-origin velocity is a slip proxy, not exact capsule contact slip. Endpoint torque demand uses the declared envelope but is not a physics-substep saturation counter. Mechanical work is a sampled approximation, not electrical consumption or temperature.'}
