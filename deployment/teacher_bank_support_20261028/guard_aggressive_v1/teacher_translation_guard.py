"""Episode-relative path/heading boundaries for pure translation skills."""
import math
import torch

METADATA={'enabled':True,'heading_limit_deg':10.,'cross_track_limit_m':.10,
 'scope':'All six pure translation teachers; turns exempt. No warmup or grace period.',
 'heading':'Actual root heading relative to episode reset heading, wrapped signed angle; not roll/pitch and not angular-velocity magnitude.',
 'path':'Fixed world line through actual reset rootXY, aligned with commanded body-frame XY rotated by reset heading. Perpendicular XY error, not height or along-path progress.',
 'termination':'Immediately terminate on abs(heading)>10deg OR abs(cross_track)>.100m; mark task failure, not timeout, so no timeout-value bootstrap. Existing physical failures retained.',
 'observation':'Append heading_error/10deg and cross_track/.100m, clipped+/-3; zero for turns. Actor/critic89D,15actions; migrate87D network and Adam moments with zero columns.'}


def root_heading(quaternion):
    x,y,z,w=quaternion.unbind(-1)
    return torch.atan2(2*(w*z+x*y),1-2*(y*y+z*z))


class TranslationGuard:
    def __init__(self,n,device,command,enabled=True):
        self.enabled=enabled&(command[:,:2].norm(dim=1)>.001)&(command[:,2].abs()<.001)
        self.command=command
        self.ready=torch.zeros(n,device=device,dtype=torch.bool)
        self.start_xy=torch.zeros(n,2,device=device);self.start_yaw=torch.zeros(n,device=device)
        self.axis=torch.zeros(n,2,device=device)
        self.heading_error=torch.zeros(n,device=device);self.cross_track=torch.zeros(n,device=device)
        self.heading_failure=torch.zeros(n,device=device,dtype=torch.bool);self.cross_failure=torch.zeros_like(self.heading_failure)

    def reset(self,ids,root):
        yaw=root_heading(root[ids,3:7]);self.start_yaw[ids]=yaw;self.start_xy[ids]=root[ids,:2]
        c=self.command[ids,:2];cs,sn=yaw.cos(),yaw.sin()
        axis=torch.stack((cs*c[:,0]-sn*c[:,1],sn*c[:,0]+cs*c[:,1]),-1)
        self.axis[ids]=axis/axis.norm(dim=1,keepdim=True).clamp(min=1e-8)
        self.ready[ids]=True
        for v in (self.heading_error,self.cross_track,self.heading_failure,self.cross_failure):v[ids]=0

    def evaluate(self,root):
        active=self.enabled&self.ready
        delta=root_heading(root[:,3:7])-self.start_yaw
        self.heading_error=torch.atan2(delta.sin(),delta.cos())*active
        xy=root[:,:2]-self.start_xy
        self.cross_track=(self.axis[:,0]*xy[:,1]-self.axis[:,1]*xy[:,0])*active
        self.heading_failure=(self.heading_error.abs()>math.radians(10)+1e-7)&active
        self.cross_failure=(self.cross_track.abs()>.10+1e-7)&active
        return self.heading_failure|self.cross_failure

    def observations(self,root):
        # Refresh also for the post-reset observation and for the pre-action read.
        self.evaluate(root)
        return torch.stack((self.heading_error/math.radians(10),self.cross_track/.10),-1).clamp(-3,3)
