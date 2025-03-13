import numpy as np
import matplotlib.pyplot as plt
import time as time

def poisson_solve(P_hat, rho_hat, kSq_inv):
    """ Solve Poisson equation in Fourier space in-place """
    P_hat[...] = -rho_hat * kSq_inv

# The function diffusion_solve is only used for a division: we directly make it with /= in main()

def grad(dvx_hat, dvy_hat, v_hat, kx, ky):
    """ Compute gradient in Fourier space in-place """
    dvx_hat[...] = 1j * kx * v_hat      # to modify directly the argument dvx_hat
    dvy_hat[...] = 1j * ky * v_hat      # same with dvy_hat

def div(div_hat, vx_hat, vy_hat, kx, ky):
    """ Compute divergence in Fourier space in-place """
    div_hat[...] = 1j * (kx * vx_hat + ky * vy_hat)

def curl(w_hat, vx_hat, vy_hat, kx, ky):
    """ Compute curl in Fourier space in-place """
    w_hat[...] = 1j * (kx * vy_hat - ky * vx_hat)

# The function apply_dealias is only used for a multiplication: we directly make it with *= in main()

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
    plotRealTime = True
    
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
    
    # Initialization of the variables for in-place operations
    dvx_x_hat = np.zeros_like(vx_hat)
    dvx_y_hat = np.zeros_like(vx_hat)
    dvy_x_hat = np.zeros_like(vx_hat)
    dvy_y_hat = np.zeros_like(vx_hat)
    div_rhs_hat = np.zeros_like(vx_hat)
    P_hat = np.zeros_like(vx_hat)
    dPx_hat = np.zeros_like(vx_hat)
    dPy_hat = np.zeros_like(vx_hat)
    wz_hat = np.zeros_like(vx_hat)

    print("Starting simulation...")
    t1 = time.time()
    
    for i in range(Nt):
        # Compute gradients in Fourier space
        grad(dvx_x_hat, dvx_y_hat, vx_hat, kx, ky)
        grad(dvy_x_hat, dvy_y_hat, vy_hat, kx, ky)
        
        # Compute RHS of momentum equation in Fourier space
        ifft_vx = np.real(np.fft.ifftn(vx_hat))
        ifft_vy = np.real(np.fft.ifftn(vy_hat))

        rhs_x_hat = -np.fft.fftn(ifft_vx * np.real(np.fft.ifftn(dvx_x_hat)) +
                                 ifft_vy * np.real(np.fft.ifftn(dvx_y_hat)))
        
        rhs_y_hat = -np.fft.fftn(ifft_vx * np.real(np.fft.ifftn(dvy_x_hat)) +
                                 ifft_vy * np.real(np.fft.ifftn(dvy_y_hat)))
        
        # In-place de-aliasing 
        rhs_x_hat *= dealias
        rhs_y_hat *= dealias
        
        vx_hat += dt * rhs_x_hat
        vy_hat += dt * rhs_y_hat
        
        # Poisson solve for pressure
        div(div_rhs_hat, rhs_x_hat, rhs_y_hat, kx, ky)
        poisson_solve(P_hat, div_rhs_hat, kSq_inv)
        grad(dPx_hat, dPy_hat, P_hat, kx, ky)
        
        # Correction step
        vx_hat -= dt * dPx_hat
        vy_hat -= dt * dPy_hat
        
        # In-place diffusion step
        vx_hat /= (1.0 + dt * nu * kSq)
        vy_hat /= (1.0 + dt * nu * kSq)
        
        # Compute vorticity for plotting
        curl(wz_hat, vx_hat, vy_hat, kx, ky)
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
