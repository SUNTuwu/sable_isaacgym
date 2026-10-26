"""Bounded event scores: signed hind stride, and low landing-angle cost."""
import math
import torch


def signed_hind_stride(distance, zero=.04, cap=.10, exponent=4.):
    positive=((distance-zero)/(cap-zero)).clamp(0.,1.)
    deficit=((zero-distance)/zero).clamp(0.,1.)
    return (torch.expm1(exponent*positive)-torch.expm1(exponent*deficit))/math.expm1(exponent)


def landing_shortfall(angle, minimum=45., scale=10., maximum_exponent=3.):
    return torch.expm1(((minimum-angle)/scale).clamp(0.,maximum_exponent))
