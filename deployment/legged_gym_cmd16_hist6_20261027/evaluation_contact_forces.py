"""Original-episode normal force, reconstructed resultant and point-slip evidence."""
import torch


class ContactForceTelemetry:
    trace_names={'forward105','reverse','left','turn_left'}

    def __init__(self,env,groups,per):
        self.env,self.groups,self.per=env,groups,per
        self.values=[];self.alive=[]
        self.traces={name:[]for name,_ in groups if name in self.trace_names}
        self.indices={name:g*per for g,(name,_)in enumerate(groups)if name in self.trace_names}

    def update(self,snapshot,commands,alive,done,tick):
        data=snapshot['contact_force_analysis']
        self.values.append(data);self.alive.append(alive.clone())
        # Include the terminal sample even when it falls between10Hz display ticks.
        for name,index in self.indices.items():
            if not bool(alive[index])or(tick%5 and not bool(done[index])):continue
            valid=bool(data['resultant_valid'][index]);pv=bool(data['point_velocity_valid'][index])
            self.traces[name].append(dict(time_s=(tick+1)*self.env.dt,
                terminal=bool(done[index]),root_world=snapshot['xyz'][index].cpu().tolist(),
                base_quat=snapshot['base_quat'][index].cpu().tolist(),
                command=commands[index,:3].cpu().tolist(),
                contact=data['contact'][index].cpu().tolist(),
                points_world=data['contact_point_world'][index].cpu().tolist(),
                normal_force_n=data['force_world'][index,:,2].cpu().tolist(),
                tangential_force_world_n=None,yaw_moment_nm=None,cancelled_force_n=None,
                point_velocity_world_m_s=data['point_velocity_world'][index].cpu().tolist()if pv else None,
                resultant_force_world_n=data['sum_force_world'][index].cpu().tolist()if valid else None,
                wrong_equivalent_force_n=float(data['wrong_equivalent_force_n'][index])if valid else None,
                costs={key:float(data[key][index])for key in ['wrong_cost','loaded_slip_cost']}))

    def finish(self):
        if not self.values:return {}
        data={key:torch.stack([row[key]for row in self.values])for key in self.values[0]}
        alive=torch.stack(self.alive);reports={}
        for g,(name,_)in enumerate(self.groups):
            sl=slice(g*self.per,(g+1)*self.per);mask=alive[:,sl]
            selected={key:value[:,sl][mask]for key,value in data.items()}
            n=int(mask.sum());contact=selected['contact']&selected['point_velocity_valid'][:,None]
            normal=selected['force_world'][...,2];load=normal.clamp(min=0)*contact
            valid=selected['resultant_valid'];nv=int(valid.sum());foot=[]
            for i,label in enumerate(['FL','FR','HL','HR']):
                speed=selected['slip_speed_m_s'][:,i][contact[:,i]]
                foot.append(dict(name=label,contact_samples=int(contact[:,i].sum()),
                    mean_normal_force_n=float(normal[:,i].mean())if n else None,
                    normal_force_p95_n=float(torch.quantile(normal[:,i],.95))if n else None,
                    contact_point_slip_rms_m_s=float(speed.square().mean().sqrt())if len(speed)else None,
                    contact_point_slip_p95_m_s=float(torch.quantile(speed,.95))if len(speed)else None,
                    tangential_force_world_n=None,opposing_correction_fraction=None))
            load_sum=load.sum()
            reports[name]=dict(original_control_samples=n,feet=foot,resultant_valid_samples=nv,
                mean_resultant_force_world_n=selected['sum_force_world'][valid].mean(0).cpu().tolist()if nv else None,
                resultant_horizontal_rms_n=float(selected['sum_force_world'][valid,:2].square().sum(-1).mean().sqrt())if nv else None,
                normal_balance_rms_error_n=float(selected['normal_balance_error_n'][valid].square().mean().sqrt())if nv else None,
                load_weighted_contact_point_slip_rms_m_s=float((
                    (selected['slip_speed_m_s'].square()*load).sum()/load_sum).sqrt())if load_sum>0 else None,
                mean_yaw_moment_nm=None,mean_cancelled_equivalent_force_n=None,
                mean_wrong_equivalent_force_n=float(selected['wrong_equivalent_force_n'][valid].mean())if nv else None,
                mean_costs={key:float(selected[key].mean())if n else None for key in ['wrong_cost','loaded_slip_cost']})
        return dict(schema='sable-four-terminal-forces/2',foot_order=['FL','FR','HL','HR'],
            sample_period_s=self.env.dt,physics_window_s=self.env.sim_params.dt,trace_period_s=5*self.env.dt,
            nominal_weight_n=self.env.contact_nominal_weight_n,scenarios=reports,
            traces=self.traces,trace_environment_indices=self.indices,
            availability=dict(per_foot_normal=True,per_foot_tangential=False,
                whole_body_resultant=True,yaw_moment=False,force_cancellation=False),
            configured_reward_weights={name:getattr(self.env.cfg.rewards.scales,name)for name in
                ['contact_force_wrong','contact_force_cancellation','loaded_contact_slip']},
            force_scope='Per-foot measured ground-normal force; whole-body external resultant reconstructed from mass-weighted COM momentum change minus gravity across the final physics step. Resultant valid only without other-body contact; no per-foot tangential force, yaw moment or cancellation measurement is available.',
            point_scope='Estimated capsule-lowest point. Native COM linear velocity plus angular velocity crossed with the COM-to-point lever; installed Gym COM convention verified using an isolated off-center rigid body. Includes rolling; no world-position finite difference at large environment offsets. Not the velocity of the moving geometric minimum or a resolved solver contact patch.',
            cost_scope='Bounded negative-only cost for resultant force opposite pre-action horizontal velocity correction; correct braking and vertical support exempt. Separate load-weighted material-point sliding cost. Per-foot cancellation disabled(weight0); useful yaw couples are not treated as cancellation.',
            scope='Original episodes including startup and terminal samples. Fixed first environment per displayed scenario; no successful-episode selection. Last-physics-step endpoint samples at50Hz, not control-period mean or exact solver work. No policy update or task acceptance implied.')
