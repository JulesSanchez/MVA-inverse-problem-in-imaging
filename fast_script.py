import numpy as np 
from scipy.special import j0, y0, i0
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import script 
def Go_F(omega,x,y):
    return (1/4)*h0(omega*(np.linalg.norm(x-y,2,axis=-1)))*complex(0,1)

def G_F(omega,x,y,x_ref,rho):
    #Born Approximation of G_F for ponctual rho
    return omega**2*Go_F(omega,x,x_ref)*rho*Go_F(omega,x_ref,y) #+ Go_F(omega,x,y)
    
def h0(s):
    return j0(s) + 1j*y0(s)

def new_Go_F(omega,x,y):
    return (1/4)*h0(omega*np.repeat(cdist(x,y)[None,:,:],len(omega),axis=0))*1j

def new_G_F(omega,x,y,x_ref,rho):
    return omega**2*new_Go_F(omega,x,x_ref)*rho*new_Go_F(omega,x_ref,y)

def exo_2(sigma,N,grid_1,omega_o,transducers,x_ref,rho,reso):
    omega_o = omega_o[:,None,None]
    u_bis = new_G_F(omega_o,transducers,transducers,np.repeat([x_ref],len(transducers),axis=0),rho)
    for k in range(len(u_bis)):
        np.fill_diagonal(u_bis[k], 0)

    W_reel = np.random.normal(0,sigma**2/2,size=u_bis.shape)
    W_cplx = np.random.normal(0,sigma**2/2,size=u_bis.shape)
    u_bis = u_bis + W_reel + 1j*W_cplx

    rho_RT_bis = np.zeros((2*reso+1,2*reso+1), dtype=complex)
    rho_KM_bis = np.zeros((2*reso+1,2*reso+1), dtype=complex)
    for k in range(2*reso+1):
        for l in range(2*reso+1):
            x_local = grid_1[k,l]
            x_local = np.repeat([x_local],len(transducers),axis=0)
            rho_RT_bis[k,l] = np.sum(new_Go_F(omega_o,x_local,transducers)*omega_o**2*new_Go_F(omega_o,transducers,x_local)*np.conj(u_bis))
            rho_KM_bis[k,l] = np.sum(np.exp(1j*omega_o*(cdist(transducers,x_local)+cdist(x_local,transducers)))*np.conj(u_bis))
    rho_RT_bis *= (0.5/np.pi)        
    rho_KM_bis /= N**2

    rho_RT_bis = np.abs(rho_RT_bis)
    rho_KM_bis = np.abs(rho_KM_bis)
    return rho_RT_bis, rho_KM_bis