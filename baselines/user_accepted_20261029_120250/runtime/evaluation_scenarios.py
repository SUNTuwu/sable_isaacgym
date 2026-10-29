"""Named commands and transition times shared by evaluation and acceptance."""
def scenarios(suite):
    old = [('forward09', [.09, 0., 0., 1.]), ('forward105', [.105, 0., 0., 1.]),
           ('forward12', [.12, 0., 0., 1.])]
    if suite == 'forward':
        return old
    base = old + [('stop', [0., 0., 0., 1.]), ('start_stop', [0., 0., 0., 1.]),
                  ('speed_switch', [.09, 0., 0., 1.])]
    if suite == 'S2':
        return base
    base += [('reverse', [-.04, 0., 0., 1.]), ('left', [0., .04, 0., 1.]),
             ('right', [0., -.04, 0., 1.]), ('turn_left', [0., 0., .35, 1.]),
             ('turn_right', [0., 0., -.35, 1.]), ('combined', [.07, .03, .25, 1.]),
             ('lateral_switch',[0.,.04,0.,1.]),('yaw_switch',[0.,0.,.35,1.])]
    if suite == 'S5':
        base += [('frequency08', [.09, 0., 0., .8]), ('frequency12', [.09, 0., 0., 1.2]),
                 ('frequency_switch', [.09, 0., 0., .8])]
    return base


def scheduled_commands(name, initial, t):
    values = list(initial)
    changed = 0.
    if name == 'start_stop':
        if t >= 11: values[0], changed = .09, 11.
        elif t >= 8: values[0], changed = 0., 8.
        elif t >= 2: values[0], changed = .105, 2.
    if name == 'speed_switch':
        if t >= 10: values[0], changed = .09, 10.
        elif t >= 5: values[0], changed = .12, 5.
    if name == 'frequency_switch':
        if t >= 20: values[3], changed = .8, 20.
        elif t >= 8: values[3], changed = 1.2, 8.
    if name == 'lateral_switch':
        if t >= 10:values[1],changed=0.,10.
        elif t >= 5:values[1],changed=-.04,5.
    if name == 'yaw_switch':
        if t >= 10:values[2],changed=0.,10.
        elif t >= 5:values[2],changed=-.35,5.
    return values, changed


def transition_segments(suite):
    result={'start_stop':[0.,2.,8.,11.],'speed_switch':[0.,5.,10.]}
    if suite in ('S3','S5'):
        result.update(lateral_switch=[0.,5.,10.],yaw_switch=[0.,5.,10.])
    if suite=='S5':result['frequency_switch']=[0.,8.,20.]
    return result

