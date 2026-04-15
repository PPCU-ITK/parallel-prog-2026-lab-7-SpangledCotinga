#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <algorithm>
#include <sstream>
#include <chrono>

using namespace std;

 std::ofstream file("Output_cpu.txt");  // creates file


// ------------------------------------------------------------
// Global parameters
// ------------------------------------------------------------
const double gamma_val = 1.4;   // Ratio of specific heats
const double CFL = 0.5;         // CFL number

// ------------------------------------------------------------
// Compute pressure from the conservative variables
// ------------------------------------------------------------
double pressure(double rho, double rhou, double rhov, double E) {
    double u = rhou / rho;
    double v = rhov / rho;
    double kinetic = 0.5 * rho * (u * u + v * v);
    return (gamma_val - 1.0) * (E - kinetic);
}

// ------------------------------------------------------------
// Compute flux in the x-direction
// ------------------------------------------------------------
void fluxX(double rho, double rhou, double rhov, double E, 
           double& frho, double& frhou, double& frhov, double& fE) {
    double u = rhou / rho;
    double p = pressure(rho, rhou, rhov, E);
    frho = rhou;
    frhou = rhou * u + p;
    frhov = rhov * u;
    fE = (E + p) * u;
}

// ------------------------------------------------------------
// Compute flux in the y-direction
// ------------------------------------------------------------
void fluxY(double rho, double rhou, double rhov, double E,
           double& frho, double& frhou, double& frhov, double& fE) {
    double v = rhov / rho;
    double p = pressure(rho, rhou, rhov, E);
    frho = rhov;
    frhou = rhou * v;
    frhov = rhov * v + p;
    fE = (E + p) * v;
}

// ------------------------------------------------------------
// Main simulation routine
// ------------------------------------------------------------
int main(){
    // ----- Grid and domain parameters -----
    const int Nx = 200 * 16;         // Number of cells in x (excluding ghost cells)
    const int Ny = 100 * 16;         // Number of cells in y
    const double Lx = 2.0;      // Domain length in x
    const double Ly = 1.0;      // Domain length in y
    const double dx = Lx / Nx;
    const double dy = Ly / Ny;

    // Create flat arrays (with ghost cells)
    const int total_size = (Nx + 2) * (Ny + 2);
    
    vector<double> rho(total_size);
    vector<double> rhou(total_size);
    vector<double> rhov(total_size);
    vector<double> E(total_size);
    
    vector<double> rho_new(total_size);
    vector<double> rhou_new(total_size);
    vector<double> rhov_new(total_size);
    vector<double> E_new(total_size);
    
    // A mask to mark solid cells (inside the cylinder)
    vector<bool> solid(total_size, false);

    // ----- Obstacle (cylinder) parameters -----
    const double cx = 0.5;      // Cylinder center x
    const double cy = 0.5;      // Cylinder center y
    const double radius = 0.1;  // Cylinder radius

    // ----- Free-stream initial conditions (inflow) -----
    const double rho0 = 1.0;
    const double u0 = 1.0;
    const double v0 = 0.0;
    const double p0 = 1.0;
    const double E0 = p0/(gamma_val - 1.0) + 0.5*rho0*(u0*u0 + v0*v0);

    // ----- Initialize grid and obstacle mask -----
    for (int i = 0; i < Nx+2; i++){
        for (int j = 0; j < Ny+2; j++){
            // Compute cell center coordinates
            double x = (i - 0.5) * dx;
            double y = (j - 0.5) * dy;
            // Mark cell as solid if inside the cylinder
            if ((x - cx)*(x - cx) + (y - cy)*(y - cy) <= radius * radius) {
                solid[i*(Ny+2)+j] = true;
                // For a wall, we set zero velocity
                rho[i*(Ny+2)+j] = rho0;
                rhou[i*(Ny+2)+j] = 0.0;
                rhov[i*(Ny+2)+j] = 0.0;
                E[i*(Ny+2)+j] = p0/(gamma_val - 1.0);
            } else {
                solid[i*(Ny+2)+j] = false;
                rho[i*(Ny+2)+j] = rho0;
                rhou[i*(Ny+2)+j] = rho0 * u0;
                rhov[i*(Ny+2)+j] = rho0 * v0;
                E[i*(Ny+2)+j] = E0;
            }
        }
    }

    // ----- Determine time step from CFL condition -----
    double c0 = sqrt(gamma_val * p0 / rho0);
    double dt = CFL * min(dx, dy) / (fabs(u0) + c0)/2.0;

    // ----- Time stepping parameters -----
    const int nSteps = 2000;
    double time1 = 0.0;
    double time2 = 0.0;
    double time3 = 0.0;
    double time4 = 0.0;
    double time5 = 0.0;
    double time6 = 0.0;
    double time7 = 0.0;

    int loop1=0;
    int loop2=0;
    int loop3=0;
    int loop4=0;
    int loop5_solid=0;
    int loop5_liquid=0;
    int loop6=0;
    int loop7=0;

    // ----- Main time-stepping loop -----
    auto t1 = std::chrono::high_resolution_clock::now();
    for (int n = 0; n < nSteps; n++){
        // --- Apply boundary conditions on ghost cells ---
        // Left boundary (inflow): fixed free-stream state
        
        auto start1 = std::chrono::high_resolution_clock::now();
        // 4 writes
        #pragma omp parallel for
        for (int j = 0; j < Ny+2; j++){
            rho[0*(Ny+2)+j] = rho0;
            rhou[0*(Ny+2)+j] = rho0*u0;
            rhov[0*(Ny+2)+j] = rho0*v0;
            E[0*(Ny+2)+j] = E0;
            loop1++;
        }
        auto end1 = std::chrono::high_resolution_clock::now();
        time1 += std::chrono::duration<double, std::milli>(end1 - start1).count();
        

        // Right boundary (outflow): copy from the interior
        auto start2 = std::chrono::high_resolution_clock::now();
        // 4 reads and writes
        #pragma omp parallel for
        for (int j = 0; j < Ny+2; j++){
            rho[(Nx+1)*(Ny+2)+j] = rho[Nx*(Ny+2)+j];
            rhou[(Nx+1)*(Ny+2)+j] = rhou[Nx*(Ny+2)+j];
            rhov[(Nx+1)*(Ny+2)+j] = rhov[Nx*(Ny+2)+j];
            E[(Nx+1)*(Ny+2)+j] = E[Nx*(Ny+2)+j];
            loop2++;
        }
        auto end2 = std::chrono::high_resolution_clock::now();
        time2 += std::chrono::duration<double, std::milli>(end2 - start2).count();
        

        // Bottom boundary: reflective
        auto start3 = std::chrono::high_resolution_clock::now();
        // 4 reads and writes
        #pragma omp parallel for
        for (int i = 0; i < Nx+2; i++){
            rho[i*(Ny+2)+0] = rho[i*(Ny+2)+1];
            rhou[i*(Ny+2)+0] = rhou[i*(Ny+2)+1];
            rhov[i*(Ny+2)+0] = -rhov[i*(Ny+2)+1];
            E[i*(Ny+2)+0] = E[i*(Ny+2)+1];
            loop3++;
        }
        auto end3 = std::chrono::high_resolution_clock::now();
        time3 += std::chrono::duration<double, std::milli>(end3 - start3).count();
        

        // Top boundary: reflective
        auto start4 = std::chrono::high_resolution_clock::now();
        // 4 reads and writes
        #pragma omp parallel for
        for (int i = 0; i < Nx+2; i++){
            rho[i*(Ny+2)+(Ny+1)] = rho[i*(Ny+2)+Ny];
            rhou[i*(Ny+2)+(Ny+1)] = rhou[i*(Ny+2)+Ny];
            rhov[i*(Ny+2)+(Ny+1)] = -rhov[i*(Ny+2)+Ny];
            E[i*(Ny+2)+(Ny+1)] = E[i*(Ny+2)+Ny];
            loop4++;
        }
        auto end4 = std::chrono::high_resolution_clock::now();
        time4 += std::chrono::duration<double, std::milli>(end4 - start4).count();
        
        
        // --- Update interior cells using a Lax-Friedrichs scheme ---
        auto start5 = std::chrono::high_resolution_clock::now();
        #pragma omp parallel for collapse(2)
        for (int i = 1; i <= Nx; i++){
            for (int j = 1; j <= Ny; j++){
                // If the cell is inside the solid obstacle, do not update it
                // if-one read
                if (solid[i*(Ny+2)+j]) {
                    // 4 reads and writes
                    rho_new[i*(Ny+2)+j] = rho[i*(Ny+2)+j];
                    rhou_new[i*(Ny+2)+j] = rhou[i*(Ny+2)+j];
                    rhov_new[i*(Ny+2)+j] = rhov[i*(Ny+2)+j];
                    E_new[i*(Ny+2)+j] = E[i*(Ny+2)+j];
                    loop5_solid++;
                    continue;
                }

                // Compsute a Lax averaging of the four neighboring cells
                // 4 reads and writes
                rho_new[i*(Ny+2)+j] = 0.25 * (rho[(i+1)*(Ny+2)+j] + rho[(i-1)*(Ny+2)+j] + 
                                             rho[i*(Ny+2)+(j+1)] + rho[i*(Ny+2)+(j-1)]);
                rhou_new[i*(Ny+2)+j] = 0.25 * (rhou[(i+1)*(Ny+2)+j] + rhou[(i-1)*(Ny+2)+j] + 
                                              rhou[i*(Ny+2)+(j+1)] + rhou[i*(Ny+2)+(j-1)]);
                rhov_new[i*(Ny+2)+j] = 0.25 * (rhov[(i+1)*(Ny+2)+j] + rhov[(i-1)*(Ny+2)+j] + 
                                              rhov[i*(Ny+2)+(j+1)] + rhov[i*(Ny+2)+(j-1)]);
                E_new[i*(Ny+2)+j] = 0.25 * (E[(i+1)*(Ny+2)+j] + E[(i-1)*(Ny+2)+j] + 
                                           E[i*(Ny+2)+(j+1)] + E[i*(Ny+2)+(j-1)]);

                // Compute fluxes
                double fx_rho1, fx_rhou1, fx_rhov1, fx_E1;
                double fx_rho2, fx_rhou2, fx_rhov2, fx_E2;
                double fy_rho1, fy_rhou1, fy_rhov1, fy_E1;
                double fy_rho2, fy_rhou2, fy_rhov2, fy_E2;
                
                // 4*4 reads
                fluxX(rho[(i+1)*(Ny+2)+j], rhou[(i+1)*(Ny+2)+j], rhov[(i+1)*(Ny+2)+j], E[(i+1)*(Ny+2)+j],
                      fx_rho1, fx_rhou1, fx_rhov1, fx_E1);
                fluxX(rho[(i-1)*(Ny+2)+j], rhou[(i-1)*(Ny+2)+j], rhov[(i-1)*(Ny+2)+j], E[(i-1)*(Ny+2)+j],
                      fx_rho2, fx_rhou2, fx_rhov2, fx_E2);
                fluxY(rho[i*(Ny+2)+(j+1)], rhou[i*(Ny+2)+(j+1)], rhov[i*(Ny+2)+(j+1)], E[i*(Ny+2)+(j+1)],
                      fy_rho1, fy_rhou1, fy_rhov1, fy_E1);
                fluxY(rho[i*(Ny+2)+(j-1)], rhou[i*(Ny+2)+(j-1)], rhov[i*(Ny+2)+(j-1)], E[i*(Ny+2)+(j-1)],
                      fy_rho2, fy_rhou2, fy_rhov2, fy_E2);

                // Apply flux differences
                double dtdx = dt / (2 * dx);
                double dtdy = dt / (2 * dy);
                
                // 4 reads and writes
                rho_new[i*(Ny+2)+j] -= dtdx * (fx_rho1 - fx_rho2) + dtdy * (fy_rho1 - fy_rho2);
                rhou_new[i*(Ny+2)+j] -= dtdx * (fx_rhou1 - fx_rhou2) + dtdy * (fy_rhou1 - fy_rhou2);
                rhov_new[i*(Ny+2)+j] -= dtdx * (fx_rhov1 - fx_rhov2) + dtdy * (fy_rhov1 - fy_rhov2);
                E_new[i*(Ny+2)+j] -= dtdx * (fx_E1 - fx_E2) + dtdy * (fy_E1 - fy_E2);
                loop5_liquid++;
            }
        }
        auto end5 = std::chrono::high_resolution_clock::now();
        time5 += std::chrono::duration<double, std::milli>(end5 - start5).count();
        

        // Copy updated values back
        // 4 reads and writes
        auto start6 = std::chrono::high_resolution_clock::now();
        #pragma omp parallel for collapse(2)
        for (int i = 1; i <= Nx; i++){
            for (int j = 1; j <= Ny; j++){
                rho[i*(Ny+2)+j] = rho_new[i*(Ny+2)+j];
                rhou[i*(Ny+2)+j] = rhou_new[i*(Ny+2)+j];
                rhov[i*(Ny+2)+j] = rhov_new[i*(Ny+2)+j];
                E[i*(Ny+2)+j] = E_new[i*(Ny+2)+j];
                loop6++;
            }
        }
        auto end6 = std::chrono::high_resolution_clock::now();
        time6 += std::chrono::duration<double, std::milli>(end6 - start6).count();
        

        // Calculate total kinetic energy
        double total_kinetic = 0.0;
        // 5 reads
        auto start7 = std::chrono::high_resolution_clock::now();
        #pragma omp parallel for reduction(+:total_kinetic) collapse(2)
        for (int i = 1; i <= Nx; i++) {
            for (int j = 1; j <= Ny; j++) {
                double u = rhou[i*(Ny+2)+j] / rho[i*(Ny+2)+j];
                double v = rhov[i*(Ny+2)+j] / rho[i*(Ny+2)+j];
                total_kinetic += 0.5 * rho[i*(Ny+2)+j] * (u * u + v * v);
                loop7++;
            }
        }
        auto end7 = std::chrono::high_resolution_clock::now();
        time7 += std::chrono::duration<double, std::milli>(end7 - start7).count();
        

       /* if (n % 50 == 0) {
            cout << "Step " << n << " completed, total kinetic energy: " << total_kinetic << endl;
        }*/
    }
    auto t2 = std::chrono::high_resolution_clock::now();

   /* // Gb/s=bytes_moves/milliseconds/1e6
    double loop1_bw = ((double)loop1 * 8.0 * 4.0) / time1 / 1e6; // GB/s
    double loop2_bw = ((double)loop2 * (8.0+8.0) * 4.0) / time2 / 1e6; // GB/s
    double loop3_bw = ((double)loop3 * (8.0+8.0) * 4.0) / time3 / 1e6; // GB/s
    double loop4_bw = ((double)loop4 * (8.0+8.0) * 4.0) / time4 / 1e6; // GB/s
    double loop5_bw = (
        ((double)(loop5_solid + loop5_liquid) * 8.0) +
        ((double)loop5_solid * (8.0+8.0) * 4.0) +
        ((double)loop5_liquid * (8.0+8.0) * 4.0) +
        ((double)loop5_liquid * 8.0 * 4.0 * 4.0) +
        ((double)loop5_liquid * (8.0+8.0) * 4.0)) / time5 / 1e6; // GB/s
    double loop6_bw = ((double)loop6 * (8.0+8.0) * 4.0) / time6 / 1e6; // GB/s
    double loop7_bw = ((double)loop7 * 8.0 * 5.0) / time7 / 1e6; // GB/s

    double loop_main_bw = (loop1_bw + loop2_bw + loop3_bw + loop4_bw + loop5_bw + loop6_bw + loop7_bw) * (double)nSteps;*/


    file << "Runtime:\t"  << "\t" << std::chrono::duration<double, std::milli>(t2-t1).count() << "ms" << "\t" << "Nx, Ny times x: 16" << "\t" << std::endl;
    cout << std::chrono::duration<double, std::milli>(t2-t1).count() << std::endl;

    file.close();

      
    return 0;
}

