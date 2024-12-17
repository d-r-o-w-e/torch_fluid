import numpy as np
import matplotlib.pyplot as plt
import os, subprocess, glob, time

# ~~ funny string tricks ~~

it_str = lambda s: "\x1B[3m" + s + "\x1B[0m"
bl_str = lambda s: "\x1B[5m" + s + "\x1B[0m"

_dens = np.array([b'\xe2\x96\xa1', 
                  b'\xe2\x96\xa4', 
                  b'\xe2\x96\xa8', 
                  b'\xe2\x96\xa6',
                  b'\xe2\x96\xa9', 
                  b'\xe2\x96\xa3'])
_dens0 = lambda a: " ".join([b.decode() for b in list(_dens[np.rint((a/(np.amax(a)-np.amin(a)+1e-8))*5).astype(int)])])
_dens1 = lambda a: os.linesep.join([" ".join(c.decode() for c in list(b)) for b in list(_dens[np.rint((a/(np.amax(a)-np.amin(a)+1e-8))*5).astype(int)])])
dens_str = lambda a: _dens0(np.nan_to_num(a, nan=0, posinf=0, neginf=0)) if len(a.shape)==1 else _dens1(np.nan_to_num(a, nan=0, posinf=0, neginf=0))

clrterm = lambda: os.system('cls' if os.name == 'nt' else 'clear')


# ~~ video saving stuff ~~

def save_i(dir, i, dpi=300):
    plt.savefig(dir + "/frame%04d.png" % int(i), dpi=dpi)

def dir2vid(dir, vidname):
    prev_cwd = os.getcwd()
    os.chdir(dir)
    subprocess.call(['ffmpeg',
                     '-i', 'frame%04d.png', 
                     '-c:v', 'libx264',
                     '-crf', '23',
                     '-profile:v', 'baseline', 
                     '-level', '3.0',
                     '-c:a', 'aac', 
                     '-ac', '2', 
                     '-b:a', '128k',
                     '-movflags', 'faststart',
                     '-pix_fmt', 'yuv420p', 
                     vidname + ".mp4"])
    time.sleep(1)
    for file_name in glob.glob("*.png"):
        os.remove(file_name)
    os.chdir(prev_cwd)