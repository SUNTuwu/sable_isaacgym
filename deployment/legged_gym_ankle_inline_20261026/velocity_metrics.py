"""Measured body-frame velocity tracking, separate from training return."""
import numpy as np

class VelocityMetrics:
    def __init__(self,n,dt):
        self.n=n;self.dt=dt;self.rows={}
    def observe(self,t,active,commands,velocity,angular_velocity):
        if t<=1.:return
        for i in np.flatnonzero(active):
            key=round(float(commands[i,0]),6)
            if key not in self.rows:self.rows[key]={'count':0,'error2':np.zeros(3),'velocity':np.zeros(3),'yaw_error2':0.,'per_env_count':np.zeros(self.n,dtype=int)}
            row=self.rows[key];error=velocity[i]-np.array([commands[i,0],commands[i,1],0.])
            row['count']+=1;row['error2']+=error**2;row['velocity']+=velocity[i]
            row['yaw_error2']+=(angular_velocity[i,2]-commands[i,2])**2;row['per_env_count'][i]+=1
    def result(self):
        groups=[]
        for speed,r in sorted(self.rows.items()):
            groups.append({'command_forward_m_s':speed,'samples':r['count'],
                'observed_environment_seconds':r['count']*self.dt,'environments_observed':int((r['per_env_count']>0).sum()),
                'mean_actual_velocity_xyz_m_s':(r['velocity']/r['count']).tolist(),
                'rmse_velocity_xyz_m_s':np.sqrt(r['error2']/r['count']).tolist(),
                'forward_bias_m_s':float(r['velocity'][0]/r['count']-speed),
                'yaw_rate_rmse_rad_s':float(np.sqrt(r['yaw_error2']/r['count']))})
        return {'definition':'Body-frame measured velocity vs actual commands after initial1s, only while the original episode is alive. Group metrics are descriptive, time samples correlated.',
                'groups':groups}
