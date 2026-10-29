"""Exact authored hinge kinematics; NumPy only, shared by preparation and Blender.

Foot handles use the physical distal tip of the terminal capsule.
All units SI. The one opposed rear hip coordinate is eliminated analytically.
"""
import json
import numpy as np

LEGS = ('FL', 'FR', 'HL', 'HR')
ENDS = ('L_wrist_fixed', 'R_wrist_fixed', 'L_ankle_pitch', 'R_ankle_pitch')
ANCHORS = ('L_shoulder_pitch', 'R_shoulder_pitch', 'L_hip_pitch', 'R_hip_pitch')

def ry(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0., s], [0., 1., 0.], [-s, 0., c]])

class Kinematics:
    def __init__(self, snapshot, authored):
        self.control_point='toe_tip'
        self.bones = snapshot['bones']
        self.names = [b['name'] for b in self.bones if b['limit'] and b['name'] not in ('R_hip_roll_mimic', 'L_wrist_fixed', 'R_wrist_fixed')]
        self.index = {n: i for i, n in enumerate(self.names)}
        self.parents = {b['name']: b['parent'] for b in self.bones}
        rests = {b['name']: np.array(b['rest']) for b in self.bones}
        self.rel = {n: np.linalg.inv(rests[p]) @ rests[n] if p else rests[n] for n, p in self.parents.items()}
        by_name = {b['name']: b for b in self.bones}
        self.limits = np.array([by_name[n]['limit'][0] for n in self.names])
        self.default = np.array([by_name[n]['q'] for n in self.names])
        self.free = np.array([i for i,n in enumerate(self.names) if n != 'spine_pitch'])
        caps = {c['bone']: c for c in authored['capsules']}
        self.caps = [caps[n] for n in ENDS]
        self.chains = []
        for n in ENDS:
            chain=[]
            while n and n!='ROOT':
                if n=='R_hip_roll_mimic': chain.append((n,self.index['hip_roll'],-1.))
                elif n in self.index: chain.append((n,self.index[n],1.))
                n=self.parents[n]
            self.chains.append(chain)

    def fk(self, q, root, jacobian=False):
        mats={'ROOT':np.asarray(root).copy()}; pivots={}; axes={}
        for b in self.bones[1:]:
            n=b['name']; m=mats[b['parent']] @ self.rel[n]
            pivots[n]=m[:3,3].copy();axes[n]=m[:3,1].copy()
            a=-q[self.index['hip_roll']] if n=='R_hip_roll_mimic' else q[self.index[n]] if n in self.index else 0.
            m[:3,:3]=m[:3,:3] @ ry(a);mats[n]=m
        feet=[];centres=[];J=np.zeros((12,len(self.names)))
        for k,(end,c) in enumerate(zip(ENDS,self.caps)):
            m=mats[end];centre=m[:3,:3]@np.array(c['center_bone_m'])+m[:3,3]
            axis=m[:3,:3]@np.array(c['axis_bone']);half=c['cylinder_length_m']*.5
            p=centre+axis*(half+c['radius_m'])
            if self.control_point=='legacy_support_projection':
                p=centre.copy();p[2]-=abs(axis[2])*half+c['radius_m']
            feet.append(p);centres.append(centre)
            if jacobian:
                for n,i,sign in self.chains[k]:
                    dp=np.cross(axes[n],p-pivots[n])
                    if self.control_point=='legacy_support_projection':
                        dp=np.cross(axes[n],centre-pivots[n])
                        dp[2]-=np.sign(axis[2])*half*np.cross(axes[n],axis)[2]
                    J[3*k:3*k+3,i]+=dp*sign
        return np.array(feet),mats,J if jacobian else np.array(centres)

    def surface_z(self,q,root):
        _,mats,_=self.fk(q,root)
        z=[]
        for end,c in zip(ENDS,self.caps):
            m=mats[end];centre=m[:3,:3]@np.array(c['center_bone_m'])+m[:3,3]
            axis=m[:3,:3]@np.array(c['axis_bone'])
            z.append(centre[2]-abs(axis[2])*c['cylinder_length_m']*.5-c['radius_m'])
        return np.array(z)

    def solve(self, targets, root, seed, prior=None, iterations=45, regularization=.00035):
        q=np.clip(np.array(seed,float),self.limits[:,0],self.limits[:,1]); prior=q.copy() if prior is None else np.asarray(prior)
        f=self.free;lam=regularization**2
        for _ in range(iterations):
            p,_,jac=self.fk(q,root,True);err=(np.asarray(targets)-p).reshape(-1)
            J=jac[:,f];g=J.T@err+lam*(prior[f]-q[f]);H=J.T@J+np.eye(len(f))*lam
            delta=np.linalg.solve(H,g); delta=np.clip(delta,-.18,.18)
            old=float(err@err+lam*np.sum((q[f]-prior[f])**2));accepted=False
            for factor in (1.,.5,.25,.1):
                candidate=q.copy();candidate[f]=np.clip(q[f]+delta*factor,self.limits[f,0],self.limits[f,1])
                e=(np.asarray(targets)-self.fk(candidate,root)[0]).reshape(-1)
                loss=float(e@e+lam*np.sum((candidate[f]-prior[f])**2))
                if loss<old:
                    q=candidate;accepted=True;break
            if not accepted or np.linalg.norm(delta)<1e-6:break
        p,_,_=self.fk(q,root)
        return q, np.linalg.norm(p-targets,axis=1)

def load_model(folder):
    from pathlib import Path
    p=Path(folder)
    return Kinematics(json.loads((p/'rig_snapshot.json').read_text()),json.loads((p/'authored_model.json').read_text()))
