import sys
sys.path.insert(0, './stlcg/src')

import numpy as np
import scipy.sparse as sp
from pltvid import save_i, dir2vid
import matplotlib.pyplot as plt
import torch
import stlcg
from tqdm import tqdm

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

from solve_fluid import *

# dissipation to use
nu = 0.01
# number of basis vectors
N = 256
# learning rate
lr = 0.001

rng = np.random.default_rng(1)

def basis11_ev_gt_a(xts, a, T):
    # xts: (N,T) array in sequential order (REVERSE BEFORE STLCG)
    
    ax = stlcg.Expression('ax', torch.tensor((1, T, 1)))
    
    return stlcg.Eventually(subformula=(ax>a), interval=[0,T//2]).robustness(torch.flip(xts[intpair_to_idx(1,1), :], (0,))[None, :, None])

def energy_always_below(xts, a, T):
    bx = stlcg.Expression('bx', torch.tensor((1, T+1, 1)))
    phi2 = stlcg.Always(subformula=(bx < a), interval=[T//10,T+1])
    return phi2.robustness((torch.flip(torch.sum(xts**2, axis=0), (0,))[None, :, None]))

def fine_detail_above(xts, lams, a, T):
    cx = stlcg.Expression('cx', torch.tensor((1, T+1, 1)))
    
    phi2 = stlcg.Always(subformula=(cx>a), interval=[T//5,T+1])
    
    return phi2.robustness((torch.flip(torch.sum(xts[lams>-10, :]**2, axis=0), (0,))[None, :, None]))

def torch_exeulerproject(x, u, Cks, dt):
    # differentiable version of the one from the other file
    
    wnorm = torch.norm(x) # save previous energy
    wnext = x + dt*torch.vstack([(x.T@Ck@x) for Ck in Cks])
    wnext = torch.nn.functional.normalize(wnext) * wnorm # reproject energy
    
    return wnext*torch.exp(nu*torch.from_numpy(lam_ks(N))*dt) + u # decay by eigenvalue/viscosity, and add ext. forces

def dynamics_loss(xts, uts, Cks, dt):
    # given xts (N,T+1) and uts (N,T), force them to obey the dynamics
    # assume xts[:,0:1] is x0
    
    T = uts.shape[1]
    
    losses = []
    
    for t in range(T):
        xt = xts[:, t:t+1]
        ut = uts[:, t:t+1]
        
        dynamics_xtp1 = torch_exeulerproject(xt, ut, Cks, dt)
        xtp1 = xts[:, t+1:t+2]
        
        losses += [torch.mean(torch.square(xtp1-dynamics_xtp1))]
        
    return sum(losses)/float(T)

def control_loss(uts):
    return torch.mean(uts**2)

def smoothness_loss(uts):
    return torch.mean((2*uts[:, 1:-1]-uts[:, :-2]-uts[:, 2:])**2)

def opt_xts(x0, T, dt, iterations):
    losses = []
    Cks = construct_square_Ck(N)
    Cks = [torch.from_numpy(Ck.todense()).to(torch.float) for Ck in Cks]
    
    lams = lam_ks(N)[:,0]
    
    # init w true dynamics
    wts = simfluid(w0=x0.copy(), uts=np.zeros((N,T)), dt=dt, T=T, method="exeul")
    xts = torch.from_numpy(np.hstack(wts)).to(torch.float)
    
    uts = torch.zeros((N,T))
    
    xts.requires_grad = True
    uts.requires_grad = True
    
    xts.data[:,0:1] = torch.from_numpy(x0)
    
    # optimizer = torch.optim.SGD([xts, uts], lr=lr)
    optimizer = torch.optim.Adam([xts, uts], lr=lr)
    
    for it in tqdm(range(iterations)):
        optimizer.zero_grad()
        
        # single coeff
        # loss = dynamics_loss(xts, uts, Cks, dt) + \
        #     control_loss(uts) + \
        #         torch.nn.functional.relu(-basis11_ev_gt_a(xts, 0.02, T)[0,0,0]) + \
        #             smoothness_loss(uts)
        
        # total energy
        # loss = dynamics_loss(xts, uts, Cks, dt) + \
        #     control_loss(uts) + \
        #         torch.nn.functional.relu(-energy_always_below(xts, 0.3, T)[0,0,0]) + \
        #             smoothness_loss(uts)
        
        # 
        loss = 100.0*dynamics_loss(xts, uts, Cks, dt) + \
            control_loss(uts) + \
                100.0*torch.nn.functional.relu(-fine_detail_above(xts, lams, 0.03, T)[0,0,0]) + \
                    smoothness_loss(uts)
 
        
        losses += [float(loss)]
        print("iteration", it)
        print("    loss:", float(loss))
        print("    (basis11) robustness:", float(basis11_ev_gt_a(xts, 0.02, T)[0,0,0]))
        print("    (energy) robustness:", float(energy_always_below(xts, 0.3, T)[0,0,0]))
        print("    (fine detail) robustness:", float(fine_detail_above(xts, lams, 0.03, T)[0,0,0]))
        print("    max |ut|^2:", float(np.amax(np.sum(uts.detach().numpy()**2, axis=0))))
        print("    min energy:", np.amin(torch.sum((xts)**2,axis=0).detach().numpy()))
        print("    max energy:", np.amax(torch.sum((xts[:, T//5:T+1])**2,axis=0).detach().numpy()))
        print("    coarse detail energy:", np.amin(torch.sum((xts[lams>-10, T//5:T+1])**2,axis=0).detach().numpy()))
        
        loss.backward()
        
        xts.grad.data[:, 0] = 0
        
        optimizer.step()
    
    return xts, uts, losses
        

if __name__=="__main__":
    
    # x0 = np.zeros((N, 1))
    x0 = (rng.random((N, 1))*2.0-1.0).astype(np.float32)*0.125
    T = 100
    dt = 0.1
    iterations = 1500
    
    st = time.time()
    xts, uts, losses = opt_xts(x0, T, dt, iterations)
    et = time.time()
    print("took", et-st)
    
    plt.plot(list(range(len(losses))), losses)
    plt.savefig("./loss_"+str(float(losses[-1]))+".png", dpi=300)
    plt.clf()
    
    print("!"*100)
    print(np.amin(uts.detach().numpy()))
    print(np.amax(uts.detach().numpy()))
    
    wts = simfluid(x0, uts.detach().numpy(), dt, T, "exeul")
    
    print("!"*100)
    print(np.amin(np.array(wts)))
    print(np.amax(np.array(wts)))
    
    pts = simdens_particle(1000, wts, dt=dt, T=T)
    
    # square_fluid_video("torch_fluid_rob"+str(float(basis11_ev_gt_a(xts, 0.02, T))), wts, 100, pts, background="particle", uts=uts.detach().numpy())
    square_fluid_video("nu"+str(nu)+"_torch_fluid_energy"+str(float(np.amax(torch.sum((xts)**2,axis=0).detach().numpy()))), wts, 100, pts, background="particle", uts=uts.detach().numpy())