"""Soft peak shaping of measured control-step joint acceleration."""
def peak_cost(acceleration,threshold,maximum_excess=10.):
    excess=((acceleration.abs()-threshold)/threshold).clamp(0.,maximum_excess)
    return excess.square().mean(1)
