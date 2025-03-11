import numpy as np
import matplotlib.pyplot as plt
import time as time

def poisson_solve(rho_hat, kSq_inv):
    """ Solve Poisson equation in Fourier space """
    return -rho_hat * kSq_inv

def diffusion_solve(v_hat, dt, nu, kSq):
    """ Solve diffusion equation in Fourier space """
    return v_hat / (1.0 + dt * nu * kSq)

def grad(v_hat, kx, ky):
    """ Compute gradient in Fourier space """
    dvx_hat = 1j * kx * v_hat
    dvy_hat = 1j * ky * v_hat
    return dvx_hat, dvy_hat

def div(vx_hat, vy_hat, kx, ky):
    """ Compute divergence in Fourier space """
    return 1j * (kx * vx_hat + ky * vy_hat)

def curl(vx_hat, vy_hat, kx, ky):
    """ Compute curl in Fourier space """
    return 1j * (kx * vy_hat - ky * vx_hat)

def apply_dealias(f_hat, dealias):
    """ Apply 2/3 rule dealiasing """
    return dealias * f_hat

def main():
    """ Optimized Navier-Stokes Simulation """
    print("Setting parameters and initializing domain...")
    
    # Simulation parameters
    N = 400
    t = 0
    tEnd = 1
    dt = 0.001
    tOut = 0.01
    nu = 0.001
    plotRealTime = False
    
    # Domain setup
    L = 1    
    xlin = np.linspace(0, L, num=N+1)[:-1]  # Remove duplicate point
    xx, yy = np.meshgrid(xlin, xlin)
    
    # Initial condition (vortex)
    vx = -np.sin(2 * np.pi * yy)
    vy = np.sin(2 * np.pi * xx * 2)
    
    # Fourier space variables
    klin = 2.0 * np.pi / L * np.fft.fftfreq(N, d=1/N)
    kx, ky = np.meshgrid(klin, klin)
    kSq = kx**2 + ky**2
    kSq_inv = np.zeros_like(kSq)
    kSq_inv[kSq > 0] = 1.0 / kSq[kSq > 0]
    
    dealias = (np.abs(kx) < (2./3.)*np.max(klin)) & (np.abs(ky) < (2./3.)*np.max(klin))
    
    # Transform initial condition to Fourier space
    vx_hat = np.fft.fftn(vx)
    vy_hat = np.fft.fftn(vy)
    
    Nt = int(np.ceil(tEnd / dt))
    fig = plt.figure(figsize=(4, 4), dpi=80)
    outputCount = 1
    
    print("Starting simulation...")
    t1 = time.time()
    
    for i in range(Nt):
        # Compute gradients in Fourier space
        dvx_x_hat, dvx_y_hat = grad(vx_hat, kx, ky)
        dvy_x_hat, dvy_y_hat = grad(vy_hat, kx, ky)
        
        # Compute RHS of momentum equation in Fourier space
        ifft_vx = np.real(np.fft.ifftn(vx_hat))
        ifft_vy = np.real(np.fft.ifftn(vy_hat))

        rhs_x_hat = -np.fft.fftn(ifft_vx * np.real(np.fft.ifftn(dvx_x_hat)) +
                                 ifft_vy * np.real(np.fft.ifftn(dvx_y_hat)))
        
        rhs_y_hat = -np.fft.fftn(ifft_vx * np.real(np.fft.ifftn(dvy_x_hat)) +
                                 ifft_vy * np.real(np.fft.ifftn(dvy_y_hat)))
        
        rhs_x_hat = apply_dealias(rhs_x_hat, dealias)
        rhs_y_hat = apply_dealias(rhs_y_hat, dealias)
        
        vx_hat += dt * rhs_x_hat
        vy_hat += dt * rhs_y_hat
        
        # Poisson solve for pressure
        div_rhs_hat = div(rhs_x_hat, rhs_y_hat, kx, ky)
        P_hat = poisson_solve(div_rhs_hat, kSq_inv)
        dPx_hat, dPy_hat = grad(P_hat, kx, ky)
        
        # Correction step
        vx_hat -= dt * dPx_hat
        vy_hat -= dt * dPy_hat
        
        # Diffusion step
        vx_hat = diffusion_solve(vx_hat, dt, nu, kSq)
        vy_hat = diffusion_solve(vy_hat, dt, nu, kSq)
        
        # Compute vorticity for plotting
        if plotRealTime or (i == Nt - 1):
            wz_hat = curl(vx_hat, vy_hat, kx, ky)
            wz = np.real(np.fft.ifftn(wz_hat))
        
        t += dt
        if (i % (Nt // 10) == 0):
            print(f"Simulating... {100 * i / Nt}%")
        
        # Real-time plotting
        plotThisTurn = t + dt > outputCount * tOut
        if (plotRealTime and plotThisTurn) or (i == Nt - 1):
            plt.cla()
            plt.imshow(wz, cmap='RdBu')
            plt.clim(-20, 20)
            ax = plt.gca()
            ax.invert_yaxis()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            ax.set_aspect('equal')
            plt.pause(0.001)
            outputCount += 1
    
    t2 = time.time()
    print("Simulation finished")
    print(f"Time taken: {t2 - t1} seconds")
    
    plt.savefig('navier-stokes-spectral.png', dpi=240)
    return 0

if __name__ == "__main__":
    main()
