"""v5.9 prespecified core-preserving shapes; all coordinates in nominal sigma."""
from common import *
from scipy.integrate import quad
from scipy.special import ndtr
from numpy.polynomial.legendre import leggauss

A = 2.25
H = 0.5
NODES, WEIGHTS = leggauss(32)
SCENARIOS = [('gaussian', 1.0)] + [(f, k) for f in ('dilation', 'curvature') for k in (1.1, 1.2, 1.3)]
WINDOWS = {'primary': 2.25, 'guard': 4.5}

def density(z, kappa, family='dilation'):
    z = np.abs(np.asarray(z, dtype=float))
    t = np.maximum(z-A, 0.)
    if family == 'gaussian' or kappa == 1:
        return np.exp(-.5*z*z)
    if family == 'dilation':
        u = np.where(z <= A, z, A + t/(1+(kappa-1)*(-np.expm1(-t/H))))
        return np.exp(-.5*u*u)
    if family == 'curvature':
        return np.exp(np.where(z <= A, -.5*z*z, -.5*A*A-A*t-.5*(t/kappa)**2))
    raise ValueError(family)

def full_integral(kappa, family):
    return 2*(quad(lambda z: float(density(z,kappa,family)),0,A,epsabs=1e-12)[0]
              +quad(lambda z: float(density(z,kappa,family)),A,np.inf,epsabs=1e-12)[0])

def integrated_bins(edges, kappa, family):
    if family == 'gaussian' or kappa == 1:
        return np.sqrt(2*np.pi)*np.diff(ndtr(edges))
    # Split the only C1 (not necessarily C2) joins before Gaussian quadrature.
    bins = np.zeros(len(edges)-1)
    for low, high in ((-np.inf,-A),(-A,A),(A,np.inf)):
        left = np.maximum(edges[:-1],low); right=np.minimum(edges[1:],high)
        valid = right > left
        center = (right[valid]+left[valid])/2; half=(right[valid]-left[valid])/2
        bins[valid] += half * (density(center[:,None]+half[:,None]*NODES,kappa,family) @ WEIGHTS)
    return bins

def weights(year,mass,kappa,family='dilation'):
    d=DATA[year]; z=(d['edges']-mass/1000.)/sigma(year,mass)
    raw=integrated_bins(z,kappa,family); total=float(raw.sum()); full=full_integral(kappa,family)
    return raw/total, dict(normalization_integral=total,full_line_integral=full,
                         outside_support_fraction=max(0.,1-total/full))

def signal_scale(year,mass):
    # Preserve the parent physical-density window and conversion exactly.
    return float(continuous_signal(year,mass).sum())

def branching_factor(mass):
    if mass <= 2*105.6583745:return 1.
    r=(105.6583745/mass)**2
    return 1+np.sqrt(1-4*r)*(1+2*r)

def shape_metrics():
    rows=[]
    for family,k in SCENARIOS:
        total=full_integral(k,family)
        tail=2*quad(lambda z:float(density(z,k,family)),A,np.inf,epsabs=1e-12)[0]/total
        leakage=2*quad(lambda z:float(density(z,k,family)),4.5,np.inf,epsabs=1e-12)[0]/total
        moment=2*(quad(lambda z:z*z*float(density(z,k,family)),0,A)[0]+quad(lambda z:z*z*float(density(z,k,family)),A,np.inf)[0])/total
        rows.append(dict(family=family,kappa=k,tail_fraction=tail,rms_ratio=np.sqrt(moment),
                         guard_leakage_fraction=leakage,full_line_integral=total,
                         ideal_core_limit_ratio=total/np.sqrt(2*np.pi)))
    return rows
