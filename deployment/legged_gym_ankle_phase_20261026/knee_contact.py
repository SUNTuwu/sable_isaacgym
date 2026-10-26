"""Ground contact envelope of the named knee motor collision shapes.

Uses the existing native plane z=0 and imported primitive geometry, not joint
center height. A cylinder support bound is conservative for PhysX's tessellation.
"""
import math
import xml.etree.ElementTree as ET
import torch


def rotate(q, v):
    xyz, w = q[..., :3], q[..., 3:4]
    return v + 2 * torch.cross(xyz, torch.cross(xyz, v, dim=-1) + w*v, dim=-1)


class KneeGroundContact:
    def __init__(self, env):
        self.env = env
        tree = ET.parse(env.cfg.asset.file)
        owners, centers, quats, sizes, kinds, sides, names = [], [], [], [], [], [], []
        for side, prefix in enumerate(['L_knee_pitch_visual_collision_', 'R_knee_pitch_visual_collision_']):
            for body in tree.iter('body'):
                for geom in body.findall('geom'):
                    if not geom.get('name', '').startswith(prefix):
                        continue
                    assert body.get('name') == ['L_hip_pitch', 'R_hip_pitch'][side]
                    kind = geom.get('type')
                    assert kind in ['box', 'cylinder'], kind
                    size = list(map(float, geom.get('size').split()))
                    w, x, y, z = map(float, geom.get('quat', '1 0 0 0').split())
                    owners.append(env.body_names.index(body.get('name')))
                    centers.append(list(map(float, geom.get('pos', '0 0 0').split())))
                    quats.append([x, y, z, w]); sizes.append((size + [0, 0, 0])[:3])
                    kinds.append(kind == 'box'); sides.append(side); names.append(geom.get('name'))
        assert sides.count(0) == 3 and sides.count(1) == 3
        self.indices = torch.tensor(owners, device=env.device, dtype=torch.long)
        self.centers = torch.tensor(centers, device=env.device)
        self.quats = torch.tensor(quats, device=env.device)
        self.quats /= self.quats.norm(dim=-1, keepdim=True)
        self.sizes = torch.tensor(sizes, device=env.device)
        self.boxes = torch.tensor(kinds, device=env.device)
        self.side_ids = [torch.tensor([i for i, s in enumerate(sides) if s == side], device=env.device) for side in range(2)]
        self.names = names
        self.local_axes = torch.stack([rotate(self.quats, torch.eye(3, device=env.device)[i].expand(len(names), -1)) for i in range(3)], dim=1)

    def clearance(self, states):
        state = states[:, self.indices]
        q = state[:, :, 3:7]
        center = state[:, :, :3] + rotate(q, self.centers[None].expand(state.shape[0], -1, -1))
        axes = rotate(q[:, :, None, :].expand(-1, -1, 3, -1), self.local_axes[None].expand(state.shape[0], -1, -1, -1))
        vertical = axes[..., 2]
        box_support = (vertical.abs() * self.sizes[None]).sum(-1)
        axial = vertical[:, :, 2].clamp(-1, 1)
        cylinder_support = self.sizes[None, :, 1] * axial.abs() + self.sizes[None, :, 0] * (1-axial.square()).clamp(min=0).sqrt()
        low = center[:, :, 2] - torch.where(self.boxes[None], box_support, cylinder_support)
        return torch.stack([low[:, ids].min(1).values for ids in self.side_ids], dim=1)

    def update(self):
        env = self.env
        env.knee_clearance = self.clearance(env.rigid_states)
        env.knee_ground_contact = env.knee_clearance <= env.cfg.termination.knee_contact_margin_m
