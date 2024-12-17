import numpy as np
import scipy.sparse as sp
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from stlpy import systems, solvers, STL
from pltvid import save_i, dir2vid
import time
from tqdm import tqdm
# tqdm = lambda x: x # to turn off tqdm

# dissipation to use
nu = 0.01
# number of basis vectors
N = 256
# N = 64
# N = 16
# mode = "drakesmooth"
mode = "scipygrad"

# ~~~ INDEXING OPERATIONS ~~~
Nn = int(np.rint(np.sqrt(N)))

def intpair_to_idx(m, n):
    m_pos = m + Nn//2
    n_pos = n + Nn//2
    return n_pos*Nn + m_pos

def idx_to_intpair(z):
    m_pos = z % Nn
    n_pos = z // Nn
    return m_pos-Nn//2, n_pos-Nn//2

# ~~~ BASIS OPERATIONS ~~~
square_velbasis = lambda k1, k2: lambda x, y: ((  1.0/(k1**2+k2**2) if (k1,k2)!=(0,0) else 1)*k2*np.sin(k1*x)*np.cos(k2*y), 
                                                -(1.0/(k1**2+k2**2) if (k1,k2)!=(0,0) else 1)*k1*np.cos(k1*x)*np.sin(k2*y))
square_vortbasis = lambda k1, k2: lambda x, y: np.sin(k1*x)*np.sin(k2*y)
square_eigv = lambda k1, k2: -(k1**2+k2**2)

def draw_basis(k1,k2,res=100):
    xs, ys = np.meshgrid(np.linspace(0,np.pi,res), np.linspace(0,np.pi,res))
    u, v = square_velbasis(k1,k2)(xs, ys)
    w = square_vortbasis(k1,k2)(xs,ys)
    uvnorm = np.sqrt(u**2 + v**2)
    
    plt.quiver(xs,ys,u/uvnorm,v/uvnorm)
    plt.imshow(w, extent=(0,1,0,1))

def construct_square_Ck(N):
    # construct the matrices Ck for N basis vectors
    datas = [[] for _ in range(N)]
    rows = [[] for _ in range(N)]
    cols = [[] for _ in range(N)]
    
    n = int(np.rint(np.sqrt(N)))
    
    for i1 in range(-n//2,n//2):
        for i2 in range(-n//2,n//2):
            for j1 in range(-n//2,n//2):
                for j2 in range(-n//2,n//2):
                    if (i1,i2) != (0,0):
                        # safe = intpair_to_idx(i1,i2)<N and intpair_to_idx(j1,j2)<N
                        # C(i1+j1, i2+j2)
                        z1 = intpair_to_idx(i1+j1, i2+j2)
                        if z1 >= 0 and z1 < N:
                            datas[z1] += [-(i1*j2-i2*j1)/(4*(i1**2+i2**2))]
                            rows[z1] += [intpair_to_idx(i1,i2)]
                            cols[z1] += [intpair_to_idx(j1,j2)]
                        # C(i1+j1, i2-j2)
                        z2 = intpair_to_idx(i1+j1, i2-j2)
                        if z2 >= 0 and z2 < N:
                            datas[z2] += [(i1*j2+i2*j1)/(4*(i1**2+i2**2))]
                            rows[z2] += [intpair_to_idx(i1,i2)]
                            cols[z2] += [intpair_to_idx(j1,j2)]
                        # C(i1-j1, i2+j2)
                        z3 = intpair_to_idx(i1-j1, i2+j2)
                        if z3 >= 0 and z3 < N:
                            datas[z3] += [-(i1*j2+i2*j1)/(4*(i1**2+i2**2))]
                            rows[z3] += [intpair_to_idx(i1,i2)]
                            cols[z3] += [intpair_to_idx(j1,j2)]
                        # C(i1-j1, i2-j2)
                        z4 = intpair_to_idx(i1-j1, i2-j2)
                        if z4 >= 0 and z4 < N:
                            datas[z4] += [(i1*j2-i2*j1)/(4*(i1**2+i2**2))]
                            rows[z4] += [intpair_to_idx(i1,i2)]
                            cols[z4] += [intpair_to_idx(j1,j2)]
    
    # construct each sparse array                
    Cks = [sp.csr_array((datas[i],(rows[i], cols[i])), shape=(N,N)) for i in range(len(datas))]
    
    return Cks

def f_square(w, u, Cks):
    crossterm = np.array([(w.T@Ck@w)[0,0] for Ck in Cks])[:, None]
    lam_ks = np.array([-(idx_to_intpair(k)[0]**2 + idx_to_intpair(k)[1]**2) for k in range(w.shape[0])])[:, None]
    return crossterm + nu*lam_ks*w + u

def f_crossterm(w, Cks):
    # return np.array([(w.T@Ck.todense()@w) for Ck in Cks])[:, None]
    return np.vstack([(w.T@Ck@w) for Ck in Cks])

def lam_ks(N):
    return np.array([-(idx_to_intpair(k)[0]**2 + idx_to_intpair(k)[1]**2) for k in range(N)])[:, None]

def f_rk4energyproject(w, u, Cks, dt):
    # slower but slightly more accurate
    
    wnorm = np.linalg.norm(w) # save previous energy
    
    # rk4 step on the crossterm only
    k1 = f_crossterm(w.copy(), Cks)
    k2 = f_crossterm(w.copy()+0.5*dt*k1, Cks)
    k3 = f_crossterm(w.copy()+0.5*dt*k2, Cks)
    k4 = f_crossterm(w.copy()+dt*k3, Cks)
    wnext = w + (dt/6.0)*(k1+2*k2+2*k3+k4)
    
    wnext = wnext * (wnorm/np.linalg.norm(wnext)) # reproject energy
    
    return wnext*np.exp(nu*lam_ks(N)*dt) + u # decay by eigenvalue/viscosity, and add ext. forces

def f_exeulenergyproject(w, u,Cks,dt):
    # faster but marginally less accurate
    
    wnorm = np.linalg.norm(w) # save previous energy
    wnext = w.copy() + dt*f_crossterm(w.copy(),Cks) # expeuler step on the crossterm only
    wnext = wnext * (wnorm/np.maximum(np.linalg.norm(wnext),1e-6)) # reproject energy
    
    return wnext*np.exp(nu*lam_ks(N)*dt) + u # decay by eigenvalue/viscosity, and add ext. forces

Cks = [Ck.todense() for Ck in construct_square_Ck(N)]
def f_stl(x, u):
    return f_exeulenergyproject(x, u, Cks, dt)

def f_stlscipy(x,u):
    return f_exeulenergyproject(x[:,None], u, Cks, dt)[:,0]

def g_square(w, u):
    return w

def project_field_to_basis(v, mode='velocity'):
    # assume v is on [-1,1]^2, in shape [res,res,2].  Then project it onto the (orthogonal) laplacian eigenbasis, and return the coefficients
    res = v.shape[0]
    xs, ys = np.meshgrid(np.linspace(0,np.pi,res), np.linspace(0,np.pi,res))
    coeffs = [0]*N
    for k in range(N):
        k1, k2 = idx_to_intpair(k)
        if mode == 'velocity':
            uk, vk = square_velbasis(k1,k2)(xs,ys)
            similarity = np.sum((uk*v[:,:,0] + vk*v[:,:,0]))/(1.0/(res))
            similarity = 0 if np.isnan(similarity) else similarity
            coeffs[k] = similarity
        elif mode == 'vorticity':
            wk = square_vortbasis(k1,k2)(xs,ys)
            similarity = np.sum(wk*v)/(2.0/(res))
            similarity = 0 if np.isnan(similarity) else similarity
            coeffs[k] = similarity
    return np.array(coeffs)[:,None]

def draw_square_fluid(wt, res, pt=None, background=None, ut=None):
    xs, ys = np.meshgrid(np.linspace(0,np.pi,res), np.linspace(0,np.pi,res))
    u, v = np.zeros_like(xs), np.zeros_like(xs)
    w = np.zeros_like(xs)
    
    if not (ut is None):
        uu, vu = np.zeros_like(xs), np.zeros_like(xs)
        wu = np.zeros_like(xs)
    
    for i in range(wt.shape[0]):
        k1, k2 = idx_to_intpair(i)
        uk, vk = square_velbasis(k1,k2)(xs,ys)
        wk = square_vortbasis(k1,k2)(xs,ys)
        u += wt[i]*uk; v += wt[i]*vk; w += wt[i]*wk
        if not (ut is None):
            uu += ut[i]*uk; vu += ut[i]*vk; wu += ut[i]*wk
        
    plt.axis('off')
    plt.gca().set_aspect('equal', 'box')
    plt.quiver(xs,ys,u,v, zorder=-1, scale_units='x', color='black')
    if not (ut is None):
        plt.quiver(xs,ys,uu,vu, zorder=-0.5, scale_units='x', color='green')
    if background == "vorticity":
        plt.imshow(w[:,::-1], vmin=-1, vmax=1, extent=(0,np.pi,0,np.pi))
    elif background == "density":
        plt.imshow(pt[:,::-1], vmin=-1, vmax=1, extent=(0,np.pi,0,np.pi))
    elif background == "particle":
        pts = np.stack(pt, axis=-1)
        for j in range(pts.shape[0]):
            plt.plot(pts[j, 0, :],pts[j, 1, :],color='tab:cyan',lw=0.75,zorder=0)
        plt.scatter(pts[:, 0, -1],pts[:, 1, -1],color='tab:blue',s=2,zorder=1)
        
def square_fluid_video(vidname, wts, res, dts=None, background="vorticity", uts=None):
    print("Generating video...")
    for t in tqdm(range(len(wts))):
        ut = None
        if not (uts is None):
            ut = uts[:, t:t+1]
            if uts[:, t:t+1].shape == (N,0):
                ut = None
        if background == "vorticity":
            draw_square_fluid(wts[t],res, background, ut=ut)
        elif background == "density":
            draw_square_fluid(wts[t], res, dts[t], background, ut=ut)
        elif background == "particle":
            taillen = 9
            draw_square_fluid(wts[t], res, dts[max(t-taillen,0):t+1], background, ut=ut)
        ax = plt.gca()
        ax.set_xlim([0,np.pi])
        ax.set_ylim([0,np.pi])
        save_i("./frames", t)
        plt.cla()
    dir2vid("./frames", vidname)

# density evolution: based on Stam 2003
# TODO: fix
def diffuse(res,x0,diff,dt):
    safe_idx = lambda i, j: x0[i, j] if (i>=0 and j>=0 and i<x0.shape[0] and j<x0.shape[1]) else 0
    x = np.zeros_like(x0).astype(np.float32)
    a=float(dt*diff*res*res)
    for k in range(20):
        for i in range(res):
            for j in range(res):
                x[i, j] = (x0[i,j] + a*(safe_idx(i-1,j)+safe_idx(i+1,j)
                                       +safe_idx(i,j-1)+safe_idx(i,j+1)))/(1+4*a)
    return x
def advect(res,u,v,d0,dt):
    d = np.zeros((res,res)).astype(np.float32)
    for i in range(res):
        for j in range(res):
            x=i-dt*u[i,j]; y=j-dt*v[i,j]
            if x < 0:
                x = 0
            if x > res-2:
                x = res-2
            if y < 0:
                y = 0
            if y > res-2:
                y = res-2
            i0=int(x); i1=int(x)+1
            j0=int(y); j1=int(y)+1
            s1=x-i0; s0=1-s1; t1=y-j0; t0=1-t1
            
            d[i,j] = s0*(t0*d0[i0,j0] + t1*d0[i0,j1]) + s1*(t0*d0[i1,j0] + t1*d0[i1,j1])
    return d
def evolve_dens(res,u,v,d0,src,dt,diff=1.0):
    x = d0 + dt*src # add source
    x = diffuse(res,x.copy(),diff,dt) # diffuse
    x = advect(res,u,v,x.copy(),dt) # advect
    return x

def simdens_expliciteuler(d0, src, wts, diff, dt, T):
    res = d0.shape[0]
    
    xs, ys = np.meshgrid(np.linspace(0,np.pi,res), np.linspace(0,np.pi,res))
    d = d0.copy()
    
    dts = [d0.copy()]
    
    for t, wt in enumerate(wts):
        u, v = np.zeros_like(xs), np.zeros_like(xs)
        for i in range(wt.shape[0]):
            k1, k2 = idx_to_intpair(i)
            uk, vk = square_velbasis(k1,k2)(xs,ys)
            u += wt[i]*uk; v += wt[i]*vk
            
        d = evolve_dens(res, u.copy(), v.copy(), d.copy(), src, dt, diff)
        
        dts += [d.copy()]
    return dts

def simdens_particle(ptcount, wts, dt, T):
    rng = np.random.default_rng(1)
    
    x = rng.random((ptcount,2))*np.pi
    
    xs = []
    
    print("Flowing particles...")
    for wt in tqdm(wts):
        uv=np.zeros_like(x)
        for i in range(wt.shape[0]):
            k1, k2 = idx_to_intpair(i)
            uk, vk = square_velbasis(k1,k2)(x[:,0:1],x[:,1:2])
            uv += np.hstack((wt[i]*uk, wt[i]*vk))
        
        x = np.where(np.logical_and(x+dt*uv>=0,x+dt*uv<=np.pi), x+dt*uv, x)
        
        xs += [x.copy()]
        
    return xs

def simfluid(w0, uts, dt, T, method="exeul"):
    wts = [w0]
    w = w0
    
    Cks = construct_square_Ck(N)
    
    print("Simulating fluid...")
    for t in tqdm(range(T)):
        # rk4 w/ projection onto sphere of equal energy before viscosity and external forces
        if method == "rk4":
            w = f_rk4energyproject(w.copy(), uts[:, t:t+1], Cks, dt)
        elif method == "exeul":
            w = f_exeulenergyproject(w.copy(), uts[:, t:t+1], Cks, dt)
        else:
            raise NotImplementedError("Undefined integration method")
        
        wts += [w.copy()]
    
    return wts

def solve_fluid(spec, sys, x0, T, solver="drakesmooth"):
    if solver == "drakesmooth":
        solver = solvers.DrakeSmoothSolver(spec, sys, x0, T)
    elif solver == "scipygrad":
        x0 = x0[:,0]
        solver = solvers.ScipyGradientSolver(spec, sys, x0, T, method='trust-constr')
    solver.AddQuadraticCost(Q=np.eye(N), R=np.eye(N))
    x,u,_,_ = solver.Solve()
    return x, u

def draw_fluid_basis():
    xs, ys = np.meshgrid(np.linspace(0,np.pi,res), np.linspace(0,np.pi,res))
    for k in range(N):
        k1, k2 = idx_to_intpair(k)
        uk, vk = square_velbasis(k1,k2)(xs,ys)
        wk = square_vortbasis(k1,k2)(xs,ys)
        plt.gca().set_aspect('equal', 'box')
        plt.imshow(wk[:,::-1], vmin=-1, vmax=1, extent=(0,np.pi,0,np.pi))
        plt.quiver(xs,ys,uk,vk)
        plt.savefig("./fluid_basis/basis_"+str(k1)+"_"+str(k2)+".png", dpi=300)
        plt.clf()

if __name__ == "__main__":
    
    dt = 0.1
    T = 100
    
    res = 100
    seed = 1
    
    rng = np.random.default_rng(seed)
    
    # save basis to pngs; takes a while
    # draw_fluid_basis()
    
    # random
    expname = "random_seed"+str(seed)+"_nu"+str(nu)
    w0 = (rng.random((N, 1))*2.0-1.0).astype(np.float32)*0.125
    
    # proj x coord
    # xs, ys = np.meshgrid(np.linspace(0,np.pi,res), np.linspace(0,np.pi,res))
    # v = np.stack([np.ones_like(xs),np.zeros_like(xs)],axis=-1)
    # print(np.mean(np.linalg.norm(v,axis=2)))
    # w0 = project_field_to_basis(v)
    # print(np.amin(w0), np.amax(w0))
    
    # proj delta
    # xs, ys = np.meshgrid(np.linspace(0,np.pi,res), np.linspace(0,np.pi,res))
    # v = np.stack([np.zeros_like(xs),np.zeros_like(xs)],axis=-1)
    # v[25,25,:] = np.array([0.005, 0.005])
    # v = gaussian_filter(v,(5,5,0))
    # w0 = project_field_to_basis(v)
    
    st = time.time()
    wts = simfluid(w0=w0.copy(), uts=np.repeat(np.zeros_like(w0), T, axis=1), dt=dt, T=T, method="rk4")
    pts = simdens_particle(1000, wts, dt=dt, T=T)
    square_fluid_video(expname, wts, res, pts, background="particle")
    et = time.time()
    
    print("took",et-st)