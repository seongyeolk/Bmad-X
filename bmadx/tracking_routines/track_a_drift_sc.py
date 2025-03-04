from bmadx.structures import Particle
from bmadx.low_level.sqrt_one import make_sqrt_one
import numpy as np
import torch

def make_track_a_drift_sc(lib):
    """Makes track_a_drift given the library lib."""
    sqrt = lib.sqrt
    sqrt_one = make_sqrt_one(lib)
    e   = 1.602e-19   # Electron charge, Coulomb
    m_p = 1.672e-27   # Proton mass
    e0  = 8.85e-12    # Electric permittivity of the free space
    c   = 299792458   # Speed of Light [m/s]
    PMASS = 938.272e+6# Proton rest mass energy

    def track_a_drift_sc(p_in, drift_sc):
        """Tracks the incoming Particle p_in though drift element
        and returns the outgoing particle. 
        See Bmad manual section 24.9 
        """
        L = drift_sc.L
        
        # Proton beam current
        I = drift_sc.I
        fscc = drift_sc.fscc
        
        s = p_in.s
        p0c = p_in.p0c
        mc2 = p_in.mc2
        
        x, px, y, py, z, pz = p_in.x, p_in.px, p_in.y, p_in.py, p_in.z, p_in.pz
        
        P = 1 + pz            # Particle's total momentum over p0
        Px = px / P           # Particle's 'x' momentum over p0
        Py = py / P           # Particle's 'y' momentum over p0
        Pxy2 = Px**2 + Py**2  # Particle's transverse mometum^2 over p0^2
        Pl = sqrt(1-Pxy2)     # Particle's longitudinal momentum over p0
        
        x = x + L * Px / Pl
        y = y + L * Py / Pl
        
        # z = z + L * ( beta/beta_ref - 1.0/Pl ) but numerically accurate:
        dz = L * (sqrt_one((mc2**2 * (2*pz+pz**2))/((p0c*P)**2 + mc2**2))
                  + sqrt_one(-Pxy2)/Pl)
        z = z + dz
        s = s + L

        # ==============================================
        # Update phase space due to space charge kick (zero length)
        sigx = sqrt( torch.mean( (x - torch.mean(x))**2) )
        sigy = sqrt( torch.mean( (y - torch.mean(y))**2) )

        # Energy and gamma
        gam   = (p0c+PMASS)/PMASS  
        bet    = sqrt(1-(1/gam**2))

        # SC kick where f_scc = 0
        Fx = e*I*(1-fscc) / (np.pi * e0 * m_p * (bet * gam * c)**3 * 3*sigx*(3*sigx + 3*sigy)) 
        Fy = e*I*(1-fscc) / (np.pi * e0 * m_p * (bet * gam * c)**3 * 3*sigy*(3*sigx + 3*sigy))

        # ==============================================
        # Space charge kick update (zero-length matrix block)
        px = Fx*x + px
        py = Fy*y + py
        
        return Particle(x, px, y, py, z, pz, s, p0c, mc2)
    
    return track_a_drift_sc