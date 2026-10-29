"""FK and named body references from the unchanged physical MJCF asset."""
import json
import xml.etree.ElementTree as ET
from pathlib import Path
import torch
from action_schema import ACTION_NAMES, FOOT_NAMES


def quat_multiply(a, b):
    av, aw, bv, bw = a[..., :3], a[..., 3:], b[..., :3], b[..., 3:]
    return torch.cat((aw * bv + bw * av + torch.cross(av, bv, dim=-1),
                      aw * bw - (av * bv).sum(-1, keepdim=True)), -1)


def quat_apply(q, v):
    qv = q[..., :3]
    uv = torch.cross(qv, v, dim=-1)
    return v + 2 * (q[..., 3:] * uv + torch.cross(qv, uv, dim=-1))


def quat_inverse(q):
    return torch.cat((-q[..., :3], q[..., 3:]), -1)


def heading_velocity(root_quat, world_velocity, world_omega):
    """Heading aligned horizontal velocity and true heading derivative.

    yaw_dot = (f_x * f_dot_y - f_y * f_dot_x) / ||f_xy||^2,
    where f is ROOT's world forward axis and f_dot=omega_world cross f.
    """
    local_x = torch.zeros_like(world_velocity)
    local_x[:, 0] = 1.
    forward = quat_apply(root_quat, local_x)
    norm2 = forward[:, :2].square().sum(-1).clamp(min=1e-6)
    flat = forward[:, :2] / norm2.sqrt()[:, None]
    xy = torch.stack(((world_velocity[:, :2] * flat).sum(-1),
                       -world_velocity[:, 0] * flat[:, 1] + world_velocity[:, 1] * flat[:, 0]), -1)
    fdot = torch.cross(world_omega, forward, dim=-1)
    yaw_dot = (forward[:, 0] * fdot[:, 1] - forward[:, 1] * fdot[:, 0]) / norm2
    return xy, yaw_dot


class ModelMapping:
    anchor_definitions = {
        'shoulder_height': [('L_shoulder_pitch', [0., 0., 0.]), ('R_shoulder_pitch', [0., 0., 0.])],
        'hip_height': [('L_hip_pitch', [0., 0., 0.]), ('R_hip_pitch', [0., 0., 0.])],
        'chest_roll': {'body': 'ROOT', 'definition': 'atan2(world_up dot body_left, world_up dot body_up)'},
    }

    def __init__(self, asset, manifest, device='cpu'):
        self.manifest = json.loads(Path(manifest).read_text())
        self.device = device
        self.nodes = []
        self.body_names = []
        root = ET.parse(asset).find('worldbody/body')

        def visit(body, parent):
            index = len(self.nodes)
            name = body.get('name')
            pos = [float(x) for x in body.get('pos', '0 0 0').split()]
            w, x, y, z = [float(x) for x in body.get('quat', '1 0 0 0').split()]
            joints = []
            for j in body.findall('joint'):
                axis = torch.tensor([float(x) for x in j.get('axis').split()], device=device)
                pivot = torch.tensor([float(x) for x in j.get('pos', '0 0 0').split()], device=device)
                joints.append((j.get('name'), axis / axis.norm(), pivot))
            self.nodes.append((name, parent, torch.tensor(pos, device=device),
                               torch.tensor([x, y, z, w], device=device), joints))
            self.body_names.append(name)
            for child in body.findall('body'):
                visit(child, index)

        visit(root, -1)
        self.capsules = []
        for name in FOOT_NAMES:
            body = next(b for b in root.iter('body') if b.get('name') == name)
            geom = next(g for g in body.findall('geom') if g.get('type') == 'capsule')
            points = torch.tensor([float(x) for x in geom.get('fromto').split()], device=device).reshape(2, 3)
            self.capsules.append((self.body_names.index(name), points, float(geom.get('size'))))

    def forward(self, root_position, root_quat, q, dof_names):
        indices = {name: i for i, name in enumerate(dof_names)}
        positions, rotations = [], []
        n = len(q)
        for name, parent, offset, rotation, joints in self.nodes:
            pp = root_position if parent < 0 else positions[parent]
            pq = root_quat if parent < 0 else rotations[parent]
            pos = pp + quat_apply(pq, offset.expand(n, -1))
            rot = quat_multiply(pq, rotation.expand(n, -1))
            for joint, axis, pivot in joints:
                half = q[:, indices[joint]:indices[joint]+1] * .5
                delta = torch.cat((axis[None, :] * half.sin(), half.cos()), -1)
                p = pivot.expand(n, -1)
                pos = pos + quat_apply(rot, p - quat_apply(delta, p))
                rot = quat_multiply(rot, delta)
            positions.append(pos)
            rotations.append(rot)
        return torch.stack(positions, 1), torch.stack(rotations, 1)

    def task_measurements(self, root_position, root_quat, q, dof_names):
        pos, rot = self.forward(root_position, root_quat, q, dof_names)
        heights = []
        for name in ('shoulder_height', 'hip_height'):
            ids = [self.body_names.index(body) for body, _ in self.anchor_definitions[name]]
            heights.append(pos[:, ids, 2].mean(1))
        x, y, z, w = root_quat.unbind(-1)
        chest_roll = torch.atan2(2 * (y*z + w*x), 1 - 2 * (x*x + y*y))
        feet, lowest = [], []
        for index, points, radius in self.capsules:
            ends = pos[:, index, None, :] + quat_apply(rot[:, index, None, :].expand(-1, 2, -1),
                                                       points[None, :, :].expand(len(q), -1, -1))
            feet.append(ends.mean(1))
            lowest.append(ends[:, :, 2].min(1).values - radius)
        return torch.stack(heights, -1), chest_roll, torch.stack(feet, 1), torch.stack(lowest, 1)

    def default_measurements(self):
        m = self.manifest
        names = list(m['default_angles'])
        q = torch.tensor([[m['default_angles'][n] for n in names]], device=self.device)
        p = torch.tensor([m['init_position']], device=self.device)
        r = torch.tensor([m['init_quat_xyzw']], device=self.device)
        heights, roll, feet, _ = self.task_measurements(p, r, q, names)
        widths = feet[0, [0, 2], 1] - feet[0, [1, 3], 1]
        return {'shoulder_height_m': float(heights[0, 0]), 'hip_height_m': float(heights[0, 1]),
                'chest_roll_rad': float(roll[0]), 'front_width_m': float(widths[0]),
                'hind_width_m': float(widths[1]), 'width0_m': float(widths.mean()),
                'source': 'unchanged physical asset FK at saved frame5; dynamic baseline reported separately'}
