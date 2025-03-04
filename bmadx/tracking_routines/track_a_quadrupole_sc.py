from bmadx.structures import Particle

from bmadx.low_level.offset_particle import make_offset_particle
from bmadx.low_level.low_energy_z_correction import make_low_energy_z_correction
from bmadx.low_level.quad_mat2_calc import make_quad_mat2_calc
import numpy as np
import torch

def make_track_a_quadrupole_sc(lib):
    """Makes track_a_quadrupole_sc given the library lib."""
    quad_mat2_calc = make_quad_mat2_calc(lib)
    offset_particle_set = make_offset_particle(lib, 'set')
    offset_particle_unset = make_offset_particle(lib, 'unset')
    low_energy_z_correction = make_low_energy_z_correction(lib)

    sqrt = lib.sqrt
    e   = 1.602e-19   # Electron charge, Coulomb
    m_p = 1.672e-27   # Proton mass
    e0  = 8.85e-12    # Electric permittivity of the free space
    c   = 299792458   # Speed of Light [m/s]
    PMASS = 938.272e+6# Proton rest mass energy
    
    def track_a_quadrupole_sc(p_in, quad_sc):
        """Tracks the incoming Particle p_in though quad element and
        returns the outgoing particle.
        See Bmad manual section 24.15
        """
        l = quad_sc.L
        I = quad_sc.I # Proton beam current
        k1 = quad_sc.K1
        fscc = quad_sc.fscc # Proton beam SC kick ratio
        n_step = quad_sc.NUM_STEPS  # number of divisions
        step_len = l / n_step  # length of division
        
        x_off = quad_sc.X_OFFSET
        y_off = quad_sc.Y_OFFSET
        tilt = quad_sc.TILT
        
        b1 = k1 * l # For electron
        #b1 = -k1 * l # For proton
        
        s = p_in.s
        p0c = p_in.p0c
        mc2 = p_in.mc2
        
        # --- TRACKING --- :
        
        par = offset_particle_set(x_off, y_off, tilt, p_in)
        x, px, y, py, z, pz = par.x, par.px, par.y, par.py, par.z, par.pz
        
        for i in range(n_step):
            rel_p = 1 + pz  # Particle's relative momentum (P/P0)
            k1 = b1/(l*rel_p)
            
            tx, dzx = quad_mat2_calc(-k1, step_len, rel_p)
            ty, dzy = quad_mat2_calc( k1, step_len, rel_p)
            
            z = ( z
                 + dzx[0] * x**2 + dzx[1] * x * px + dzx[2] * px**2
                 + dzy[0] * y**2 + dzy[1] * y * py + dzy[2] * py**2 )
            
            x_next = tx[0][0] * x + tx[0][1] * px
            px_next = tx[1][0] * x + tx[1][1] * px
            y_next = ty[0][0] * y + ty[0][1] * py
            py_next = ty[1][0] * y + ty[1][1] * py
            
            x, px, y, py = x_next, px_next, y_next, py_next
            
            z = z + low_energy_z_correction(pz, p0c, mc2, step_len)
        
        s = s + l

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
        
        par = offset_particle_unset(x_off, y_off, tilt,
                                    Particle(x, px, y, py, z, pz, s, p0c, mc2))
        
        return par
      
    return track_a_quadrupole_sc