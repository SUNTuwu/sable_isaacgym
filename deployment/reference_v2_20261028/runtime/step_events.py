"""Command-direction-independent actual flight and landing measurements."""
import torch


class StepEvents:
    def __init__(self, count, device, dt):
        self.dt = dt
        self.previous_contact = torch.ones(count, 4, dtype=torch.bool, device=device)
        self.armed = torch.zeros_like(self.previous_contact)
        self.takeoff_xy = torch.zeros(count, 4, 2, device=device)
        self.clear_duration = torch.zeros(count, 4, device=device)
        self.longest_clear = torch.zeros_like(self.clear_duration)
        self.landed = torch.zeros_like(self.previous_contact)
        self.travel_m = torch.zeros_like(self.clear_duration)

    def reset(self, ids):
        self.previous_contact[ids] = True
        for value in (self.armed, self.takeoff_xy, self.clear_duration, self.longest_clear, self.landed, self.travel_m):
            value[ids] = 0

    def update(self, xy, contact, clearance, age, moving):
        takeoff = self.previous_contact & ~contact
        landing = ~self.previous_contact & contact
        self.takeoff_xy[takeoff] = xy[takeoff]
        self.armed[takeoff] = ((age >= 1.) & moving)[:, None].expand_as(takeoff)[takeoff]
        self.clear_duration[takeoff] = 0.
        self.longest_clear[takeoff] = 0.
        self.clear_duration = torch.where(~contact & (clearance >= .002), self.clear_duration+self.dt,
                                          torch.zeros_like(self.clear_duration))
        self.longest_clear = torch.maximum(self.longest_clear, self.clear_duration)
        self.travel_m = (xy-self.takeoff_xy).norm(dim=-1)
        self.landed = landing & self.armed & (self.longest_clear >= .060-1e-7) & (self.travel_m >= .005)
        self.armed[landing] = False
        self.previous_contact.copy_(contact)
        return self.landed
