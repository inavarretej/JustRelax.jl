const isCUDA = false
#const isCUDA = true

@static if isCUDA
    using CUDA
end

using JustRelax, JustRelax.JustRelax2D, JustRelax.DataIO
import JustRelax.@cell

const backend = @static if isCUDA
    CUDABackend # Options: CPUBackend, CUDABackend, AMDGPUBackend
else
    JustRelax.CPUBackend # Options: CPUBackend, CUDABackend, AMDGPUBackend
end

using ParallelStencil, ParallelStencil.FiniteDifferences2D

@static if isCUDA
    @init_parallel_stencil(CUDA, Float64, 2)
else
    @init_parallel_stencil(Threads, Float64, 2)
end

using JustPIC, JustPIC._2D
# Threads is the default backend,
# to run on a CUDA GPU load CUDA.jl (i.e. "using CUDA") at the beginning of the script,
# and to run on an AMD GPU load AMDGPU.jl (i.e. "using AMDGPU") at the beginning of the script.
const backend_JP = @static if isCUDA
    CUDABackend # Options: CPUBackend, CUDABackend, AMDGPUBackend
else
    JustPIC.CPUBackend # Options: CPUBackend, CUDABackend, AMDGPUBackend
end

# Load script dependencies
using GeoParams
using GLMakie
# Load file with all the rheology configurations
include("RiftSetup.jl")
include("RiftRheology.jl")

## SET OF HELPER FUNCTIONS PARTICULAR FOR THIS SCRIPT --------------------------------

import ParallelStencil.INDICES
const idx_k = INDICES[2]
macro all_k(A)
    esc(:($A[$idx_k]))
end

function copyinn_x!(A, B)
    @parallel function f_x(A, B)
        @all(A) = @inn_x(B)
        return nothing
    end

    @parallel f_x(A, B)
end

# Initial pressure profile - not accurate
@parallel function init_P!(P, ρg, z)
    @all(P) = abs(@all(ρg) * @all_k(z)) * <(@all_k(z), 0.0)
    return nothing
end

function BC_topography_displ(Ux, Uy, εbg, xvi, lx, ly, dt)
    xv, yv = xvi

    @parallel_indices (i, j) function pure_shear_x!(Ux)
        xi = xv[i]
        if i == 1 || i == size(Ux, 1)
            Ux[i, j + 1] = εbg * (xi - lx * 0.5) * lx * dt / 2
        end
        return nothing
    end

    @parallel_indices (i, j) function pure_shear_y!(Uy)
        yi = yv[j]
        if j == 1 || j == size(Uy, 1)
            Uy[i + 1, j] = abs(yi) * εbg * ly * dt / 2
        end
        return nothing
    end

    nx, ny = size(Ux)
    @parallel (1:nx, 1:(ny - 2)) pure_shear_x!(Ux)
    nx, ny = size(Uy)
    @parallel (1:(nx - 2), 1:ny) pure_shear_y!(Uy)

    return nothing
end

## Functins borrowed from Utils.jl

#include("../src/common.jl")
#include("../src/Utils.jl")
# Initialize
n        = 200
nx, ny   = n, n ÷ 2
# li, origin, phases_GMG, T_GMG = Setup_Topo(nx+1, ny+1)
li, origin, phases_GMG, T_GMG = flat_setup(nx+1, ny+1)
do_vtk   = true # set to true to generate VTK files for ParaView
figdir   = "output/Rift2D_strain_increment"

nx, ny = size(T_GMG).-1

igg      = if !(JustRelax.MPI.Initialized()) # initialize (or not) MPI grid
    IGG(init_global_grid(nx, ny, 1; init_MPI= true)...)
else
    igg
end


# Physical domain ------------------------------------
ni                  = nx, ny           # number of cells
di                  = @. li / ni       # grid steps
grid                = Geometry(ni, li; origin = origin)
(; xci, xvi)        = grid # nodes at the center and vertices of the cells
# ----------------------------------------------------

# Physical properties using GeoParams ----------------
rheology            = init_rheologies()
dt                  =  5e2 * 3600 * 24 * 365 # diffusive CFL timestep limiter
dtmax               = 10e3 * 3600 * 24 * 365 # diffusive CFL timestep limiter
# ----------------------------------------------------

# Initialize particles -------------------------------
nxcell              = 40
max_xcell           = 60
min_xcell           = 20
particles           = init_particles(
    backend_JP, nxcell, max_xcell, min_xcell, xvi, di, ni
)
subgrid_arrays      = SubgridDiffusionCellArrays(particles)

# velocity grids
grid_vxi            = velocity_grids(xci, xvi, di)

# material phase & temperature  
pPhases, pT         = init_cell_arrays(particles, Val(2))
particle_args       = (pT, pPhases)

# Assign particles phases anomaly
phases_device    = PTArray(backend)(phases_GMG)
phase_ratios     = PhaseRatios(backend_JP, length(rheology),ni)
init_phases!(pPhases, phases_device, particles, xvi)
update_phase_ratios!(phase_ratios, particles, xci,xvi, pPhases)
# ----------------------------------------------------

# STOKES ---------------------------------------------
# Allocate arrays needed for every Stokes problem
stokes           = StokesArrays(backend, ni)
pt_stokes        = PTStokesCoeffs(li, di; ϵ=1e-4, Re=3π, r=1e0, CFL = 1 / √2.1) # Re=3π, r=0.7
# ----------------------------------------------------

# TEMPERATURE PROFILE --------------------------------
Ttop             = minimum(T_GMG)
Tbot             = maximum(T_GMG)
thermal          = ThermalArrays(backend, ni)
@views thermal.T[2:end-1, :] .= PTArray(backend)(T_GMG)
thermal_bc       = TemperatureBoundaryConditions(;
    no_flux      = (left = true, right = true, top = false, bot = false),
)
thermal_bcs!(thermal, thermal_bc)
@views thermal.T[:, end] .= Ttop
@views thermal.T[:, 1]   .= Tbot
temperature2center!(thermal)
# ----------------------------------------------------

# Buoyancy forces
ρg               = ntuple(_ -> @zeros(ni...), Val(2))

θ_ref =(@zeros((ni)...),@zeros((ni)...),@zeros((ni)...),@zeros((ni.+1)...))

compute_ρg!(ρg[2], phase_ratios, rheology, (T=thermal.Tc, P=stokes.P))
stokes.P        .= PTArray(backend)(reverse(cumsum(reverse((ρg[2]).* di[2], dims=2), dims=2), dims=2))

# Rheology
args0            = (T=thermal.Tc, P=stokes.P, dt = Inf)
viscosity_cutoff = (1e17, 1e23)
compute_viscosity!(stokes, phase_ratios, args0, rheology, viscosity_cutoff)

# PT coefficients for thermal diffusion
pt_thermal       = PTThermalCoeffs(
    backend, rheology, phase_ratios, args0, dt, ni, di, li; ϵ=1e-5, CFL=0.98 / √2.1
)

# Boundary conditions
# flow_bcs         = VelocityBoundaryConditions(;
#     free_slip    = (left = true , right = true , top = true , bot = true),
#     free_surface = false,
# )

flow_bcs         = DisplacementBoundaryConditions(;
    free_slip    = (left = true , right = true , top = true , bot = true),
    free_surface = false,
)

εbg = 1e-14 # background strain rate
stokes.U.Ux[:, 2:(end - 1)] .= PTArray(backend)([ εbg * x * dt for x in xvi[1], y in xci[2]])
stokes.U.Uy[2:(end - 1), :] .= PTArray(backend)([-εbg * y * dt for x in xci[1], y in xvi[2]])

# BC_topography_displ(stokes.U.Ux, stokes.U.Uy, εbg, xvi, li..., dt)
flow_bcs!(stokes, flow_bcs) # apply boundary conditions
displacement2velocity!(stokes, dt)
update_halo!(@velocity(stokes)...)

    # IO -------------------------------------------------
    # if it does not exist, make folder where figures are stored
if do_vtk
    vtk_dir      = joinpath(figdir, "vtk")
    take(vtk_dir)
end
take(figdir)
    # ----------------------------------------------------

local Vx_v, Vy_v
if do_vtk
    Vx_v = @zeros(ni.+1...)
    Vy_v = @zeros(ni.+1...)
    Ux_v = @zeros(ni.+1...)
    Uy_v = @zeros(ni.+1...)
    strain = PTArray(backend)(@zeros(ni.+1...))
    strain_rate = PTArray(backend)(@zeros(ni.+1...))
end

T_buffer    = @zeros(ni.+1)
Told_buffer = similar(T_buffer)
dt₀         = similar(stokes.P)
for (dst, src) in zip((T_buffer, Told_buffer), (thermal.T, thermal.Told))
    copyinn_x!(dst, src)
end
grid2particle!(pT, xvi, T_buffer, particles)

# Time loop
t, it = 0.0, 0

# uncomment for random cohesion damage
cohesion_damage = @rand(ni...) .* 0.05 # 5% random cohesion damage

 # iterations over time , uncomment next line
while it < 5000  && t < 1.5768e+14 # run only for 5 Myrs

    # BC_topography_displ(stokes.U.Ux, stokes.U.Uy, εbg, xvi, li..., dt)
    # flow_bcs!(stokes, flow_bcs) # apply boundary conditions
    # update_halo!(@velocity(stokes)...)

    # interpolate fields from particle to grid vertices
    particle2grid!(T_buffer, pT, xvi, particles)
    @views T_buffer[:, end]      .= Ttop
    @views T_buffer[:, 1]        .= Tbot
    @views thermal.T[2:end-1, :] .= T_buffer
    thermal_bcs!(thermal, thermal_bc)
    temperature2center!(thermal)

    args = (; T = thermal.Tc, P = stokes.P,  dt=Inf, cohesion_C = cohesion_damage)
    strain_increment=true
    # Stokes solver ----------------
    t_stokes = @elapsed begin
        out = solve!(
            stokes,
            pt_stokes,
            di,
            flow_bcs,
            ρg,
            phase_ratios,
            rheology,
            args,
            dt,
            strain_increment,
            igg,
            ;
            kwargs = (
                iterMax              = 10e5,
                nout                 = 1e3,
                viscosity_cutoff     = viscosity_cutoff,
                free_surface         = false,
                viscosity_relaxation = 1e-3
            )
        );
    end


    dt   = min(dtmax, compute_dt(stokes, di) * 0.95)
    println("Stokes solver time             ")
    println("   Total time:      $t_stokes s")
    println("   Time/iteration:  $(t_stokes / out.iter) s")
    # ------------------------------

    compute_shear_heating!(
        thermal,
        stokes,
        phase_ratios,
        rheology, # needs to be a tuple
        dt,
    )

    # Thermal solver ---------------
    heatdiffusion_PT!(
        thermal,
        pt_thermal,
        thermal_bc,
        rheology,
        args,
        dt,
        di;
        kwargs = (
            igg     = nothing,
            phase   = phase_ratios,
            iterMax = 10e3,
            nout    = 1e3,
            verbose = true,
        )
    )
    @show extrema(thermal.T)

    subgrid_characteristic_time!(
        subgrid_arrays, particles, dt₀, phase_ratios, rheology, thermal, stokes, xci, di
    )
    centroid2particle!(subgrid_arrays.dt₀, xci, dt₀, particles)
    subgrid_diffusion!(
        pT, thermal.T, thermal.ΔT, subgrid_arrays, particles, xvi,  di, dt
    )
    # ------------------------------

    # Advection --------------------
    # advect particles in space
    advection_LinP!(particles, RungeKutta2(), @velocity(stokes), grid_vxi, dt)
    # advect particles in memory
    move_particles!(particles, xvi, particle_args)

    # check if we need to inject particles
    inject_particles_phase!(particles, pPhases, (pT, ), (T_buffer, ), xvi)
    # update phase ratios
    update_phase_ratios!(phase_ratios, particles, xci, xvi, pPhases)

    @show it += 1
    t        += dt
    println("Simulation time: $(t/3.154e+7) years")
    # Data I/O and plotting ---------------------
    if it == 1 || rem(it, 1) == 0
        # checkpointing(figdir, stokes, thermal.T, η, t)


        (; η_vep, η) = stokes.viscosity
        if do_vtk

            velocity2vertex!(Vx_v, Vy_v, @velocity(stokes)...)
            velocity2vertex!(Ux_v, Uy_v, @displacement(stokes)...)
            data_v = (;
                T   = Array(T_buffer),        
                Vx  = Array(Vx_v),
                Vy  = Array(Vy_v),
                Ux = Array(Ux_v),
                Uy = Array(Uy_v),
                    )
            data_c = (;
                Pdyno= Array(stokes.P) .- Array(reverse(cumsum(reverse((ρg[2]).* di[2], dims=2), dims=2), dims=2)),
                P   = Array(stokes.P),
                η_v = Array(η),
                η   = Array(η_vep),
                τII = Array(stokes.τ.II),
                τxx = Array(stokes.τ.xx),
                τyy = Array(stokes.τ.yy),
                εII = Array(stokes.ε.II),
                εII_pl = Array(stokes.ε_pl.II),
                strain_rate = Array(strain_rate),
            )
            velocity_v = (
                Array(Vx_v),
                Array(Vy_v),
            )
            save_vtk(
                joinpath(vtk_dir, "vtk_" * lpad("$it", 6, "0")),
                xvi,
                xci,
                data_v,
                data_c,
                velocity_v;
                t=t
            )
        end
    


            # Make particles plottable
            tensor_invariant!(stokes.ε)
            tensor_invariant!(stokes.ε_pl)
            tensor_invariant!(stokes.τ)

            p        = particles.coords
            ppx, ppy = p
            pxv      = ppx.data[:]./1e3
            pyv      = ppy.data[:]./1e3
            clr      = pPhases.data[:]
            # clr      = pT.data[:]
            idxv     = particles.index.data[:];
            x1 = 1e3.*(1:length(out.norm_Rx)) 
            # Make Makie figure
            ar  = 3
            fig = Figure(size = (1500, 600), title = "t = $t")
            ax1 = Axis(fig[1,1], aspect = ar, title = "log10(τII)")
            ax2 = Axis(fig[2,1], aspect = ar, title = "log10(ε_pl.II)")
            ax3 = Axis(fig[1,3], aspect = ar, title = "log10(εII)")
            ax4 = Axis(fig[2,3], aspect = ar, title = "log10(η)")
            ax5 = Axis(fig[1, 5], title="Log10 Rx", xlabel="Iteration", ylabel="Error")
            ax6 = Axis(fig[2, 5], title="Log10 Ry", xlabel="Iteration", ylabel="Error")
            # Plot temperature
            h1  = GLMakie.heatmap!(ax1, xci[1].*1e-3, xci[2].*1e-3,Array(log10.(stokes.τ.II)) , colormap=:batlow)
            # Plot particles phase
            # h2  = GLMakie.scatter!(ax2, Array(pxv[idxv]), Array(pyv[idxv]), color=Array(clr[idxv]), markersize = 1)
            h2 = GLMakie.heatmap!(ax2, xci[1].*1e-3, xci[2].*1e-3, Array(log10.(stokes.ε_pl.II)) , colormap=:batlow)
            # Plot 2nd invariant of strain rate
            h3  = GLMakie.heatmap!(ax3, xci[1].*1e-3, xci[2].*1e-3, Array(log10.(stokes.ε.II)) , colormap=:batlow)
            # Plot effective viscosity
            h4  = GLMakie.heatmap!(ax4, xci[1].*1e-3, xci[2].*1e-3, Array(log10.(η_vep)) , colormap=:lipari)
            h5 = GLMakie.plot!(ax5, x1, Array(log10.(out.norm_Rx)), color=:blue)
            h6 = GLMakie.plot!(ax6, x1, Array(log10.(out.norm_Ry)), color=:blue)
            hidexdecorations!(ax1)
            hidexdecorations!(ax2)
            hidexdecorations!(ax3)
            h = 200
            Colorbar(fig[1,2], h1, height = h)
            Colorbar(fig[2,2], h2, height = h)
            Colorbar(fig[1,4], h3, height = h)
            Colorbar(fig[2,4], h4, height = h)
            linkaxes!(ax1, ax2, ax3, ax4)
            linkaxes!(ax5, ax6)
            fig
            save(joinpath(figdir, "$(it).png"), fig)

        end
    end

    # ------------------------------






