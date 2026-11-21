"""Gripper edge on a constant-radius arc about the intended fold line."""
import math

def fold_offset(t,start_s,duration_s,terminal_clearance_m,radius_m=.14):
    if not all(math.isfinite(x) for x in (t,start_s,duration_s,terminal_clearance_m,radius_m)) or duration_s<=0 or not 0<terminal_clearance_m<radius_m:raise ValueError('Invalid fold motion')
    x=min(1.,max(0.,(t-start_s)/duration_s));q=x*x*x*(10+x*(-15+6*x))
    angle=(math.pi-math.asin(terminal_clearance_m/radius_m))*q
    return radius_m*math.cos(angle),radius_m*math.sin(angle)
