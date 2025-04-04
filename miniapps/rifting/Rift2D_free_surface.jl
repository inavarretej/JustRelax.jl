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
#using  CairoMakie

# Load file with all the rheology configurations
include("RiftSetup_FaultInclusion.jl")
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

@parallel_indices (i, j) function update_Dirichlet_mask!(mask, phase_ratio_vertex, air_phase)
    @inbounds mask[i + 1, j] = @index(phase_ratio_vertex[air_phase, i, j]) == 1
    nothing
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

## END OF HELPER FUNCTION ------------------------------------------------------------


## END OF MAIN SCRIPT ----------------------------------------------------------------
do_vtk   = true # set to true to generate VTK files for ParaView
figdir   = "output/Rift2D_reset_plastic2"
n        = 256
nx, ny   = n, n ÷ 2
# li, origin, phases_GMG, T_GMG = Setup_Topo(nx+1, ny+1)
li, origin, phases_GMG, T_GMG = flat_setup(nx+1, ny+1)
# heatmap(T_GMG)
# heatmap(phases_GMG)

nx, ny = size(T_GMG).-1

igg      = if !(JustRelax.MPI.Initialized()) # initialize (or not) MPI grid
    IGG(init_global_grid(nx, ny, 1; init_MPI= true)...)
else
    igg
end


## BEGIN OF MAIN SCRIPT --------------------------------------------------------------
function main(li, origin, phases_GMG, igg; nx=16, ny=16, figdir="figs2D", do_vtk =false)

    # Physical domain ------------------------------------
    ni                  = nx, ny           # number of cells
    di                  = @. li / ni       # grid steps
    grid                = Geometry(ni, li; origin = origin)
    (; xci, xvi)        = grid # nodes at the center and vertices of the cells
    # ----------------------------------------------------

    # Physical properties using GeoParams ----------------
    rheology            = init_rheologies()
    dt                  = 1e3 * 3600 * 24 * 365 # diffusive CFL timestep limiter
    dtmax               = 25e3 * 3600 * 24 * 365 # diffusive CFL timestep limiter
    # ----------------------------------------------------

    # Initialize particles -------------------------------
    nxcell    = 100
    max_xcell = 125
    min_xcell = 75
    particles = init_particles(
        backend_JP, nxcell, max_xcell, min_xcell, xvi, di, ni
    )
    # velocity grids
    grid_vxi = velocity_grids(xci, xvi, di)
    # material phase & temperature
    subgrid_arrays = SubgridDiffusionCellArrays(particles)
    pPhases, pT    = init_cell_arrays(particles, Val(2))
    particle_args  = (pT, pPhases)
    # Assign particles phases anomaly
    phases_device = PTArray(backend)(phases_GMG)
    phase_ratios  = phase_ratios = PhaseRatios(backend_JP, length(rheology), ni)
    init_phases!(pPhases, phases_device, particles, xvi)
    update_phase_ratios!(phase_ratios, particles, xci, xvi, pPhases)
    # ----------------------------------------------------

    # RockRatios
    air_phase = 4
    ϕ_R = RockRatio(backend, ni)
    update_rock_ratio!(ϕ_R, phase_ratios, air_phase)
 
    # marker chain
    nxcell, min_xcell, max_xcell = 100, 75, 125
    initial_elevation = 0e3
    chain = init_markerchain(backend_JP, nxcell, min_xcell, max_xcell, xvi[1], initial_elevation)
    update_phases_given_markerchain!(pPhases, chain, particles, origin, di, air_phase)
    update_phase_ratios!(phase_ratios, particles, xci, xvi, pPhases)
    update_rock_ratio!(ϕ_R, phase_ratios, air_phase)
  
    # STOKES ---------------------------------------------
    # Allocate arrays needed for every Stokes problem
    stokes           = StokesArrays(backend, ni)
    # pt_stokes        = PTStokesCoeffs(li, di; ϵ=1e-4, Re=3π, r=1e0, CFL = 1 / √2.1) # Re=3π, r=0.7
    pt_stokes        = PTStokesCoeffs(li, di; ϵ=1e-4, Re=3π, r=0.7, CFL = 0.85 / √2.1) # Re=3π, r=0.7
    # ----------------------------------------------------

    # TEMPERATURE PROFILE --------------------------------
    Ttop             = minimum(T_GMG)
    Tbot             = maximum(T_GMG)
    thermal          = ThermalArrays(backend, ni)
    @views thermal.T[2:end-1, :] .= PTArray(backend)(T_GMG)

    # Add thermal anomaly BC's
    T_air = 273.0e0
    Ω_T = @zeros(size(thermal.T)...)
    
    thermal_bc = TemperatureBoundaryConditions(;
        no_flux     = (; left = true, right = true, top = false, bot = false),
        dirichlet   = (; constant = T_air, mask = Ω_T)
    )
    thermal_bcs!(thermal, thermal_bc)
    @views thermal.T[:, end] .= Ttop
    @views thermal.T[:, 1]   .= Tbot
    temperature2center!(thermal)
    # ----------------------------------------------------

    # Buoyancy forces
    ρg               = ntuple(_ -> @zeros(ni...), Val(2))
    compute_ρg!(ρg[2], phase_ratios, rheology, (T=thermal.Tc, P=stokes.P))
    stokes.P        .= PTArray(backend)(reverse(cumsum(reverse((ρg[2]).* di[2], dims=2), dims=2), dims=2))

    # Rheology
    args0            = (T=thermal.Tc, P=stokes.P, dt = Inf)
    viscosity_cutoff = (1e17, 1e23)
    compute_viscosity!(stokes, phase_ratios, args0, rheology, viscosity_cutoff; air_phase = air_phase)

    # PT coefficients for thermal diffusion
    pt_thermal       = PTThermalCoeffs(
        backend, rheology, phase_ratios, args0, dt, ni, di, li; ϵ=1e-5, CFL=0.98 / √2.1
    )

    # # Boundary conditions
    # flow_bcs         = VelocityBoundaryConditions(;
    #     free_slip    = (left = true , right = true , top = true , bot = true),
    #     free_surface = false,
    # )

    flow_bcs         = DisplacementBoundaryConditions(;
        free_slip    = (left = true , right = true , top = true , bot = true),
        free_surface = false,
    )

    εbg = +6.34e-16 # background strain rate
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
    # cohesion_damage = @rand(ni...) .* 0.05 # 5% random cohesion damage
    strain_increment = true
    while it < 500 # run only for 5 Myrs

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

        # args = (; T = thermal.Tc, P = stokes.P,  dt=Inf, cohesion_C = cohesion_damage)
        args = (; T = thermal.Tc, P = stokes.P,  dt=Inf)

        # Stokes solver ----------------
        t_stokes = @elapsed begin
            solve_VariationalStokes!(
                stokes,
                pt_stokes,
                di,
                flow_bcs,
                ρg,
                phase_ratios,
                ϕ_R,
                rheology,
                args,
                dt,
                strain_increment,
                igg;
                kwargs = (;
                    iterMax = it > 0 ? 100.0e4 : 250.0e4,
                    free_surface = true,
                    nout = 1e3,
                    viscosity_cutoff = viscosity_cutoff,
                )
            )
        end
        # dtmax = (it < 10 ? 2e3 : 3e3) * 3600 * 24 * 365 # diffusive CFL timestep limiter
        dt    = compute_dt(stokes, di, dtmax)
        println("Stokes solver time             ")
        println("   Total time:      $t_stokes s")
        println("           Δt:      $(dt / (3600 * 24 * 365)) kyrs")
        # println("   Time/iteration:  $(t_stokes / out.iter) s")
        # ------------------------------

        compute_shear_heating!(
            thermal,
            stokes,
            phase_ratios,
            rheology, # needs to be a tuple
            dt,
        )

        # Thermal solver ---------------
        # update mask of Dirichlet BC
        @parallel (@idx ni .+ 1) update_Dirichlet_mask!(thermal_bc.dirichlet.mask, phase_ratios.vertex, air_phase)
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
                iterMax = 50e4,
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
        advection_MQS!(particles, RungeKutta2(), @velocity(stokes), grid_vxi, dt)
        # advect particles in memory
        move_particles!(particles, xvi, particle_args)
        # check if we need to inject particles
        inject_particles_phase!(particles, pPhases, (pT, ), (T_buffer, ), xvi)

        # advect marker chain
        advect_markerchain!(chain, RungeKutta2(), @velocity(stokes), grid_vxi, dt)
        update_phases_given_markerchain!(pPhases, chain, particles, origin, di, air_phase)

        # update phase ratios
        update_phase_ratios!(phase_ratios, particles, xci, xvi, pPhases)
        update_rock_ratio!(ϕ_R, phase_ratios, air_phase)
  

        @show it += 1
        t        += dt

        # Data I/O and plotting ---------------------
        if it == 1 || rem(it, 1) == 0
            # checkpointing(figdir, stokes, thermal.T, η, t)
            (; η_vep, η) = stokes.viscosity
            if do_vtk
                
                velocity2vertex!(Vx_v, Vy_v, @velocity(stokes)...)
                data_v = (;
                    T   = Array(T_buffer),        
                    Vx  = Array(Vx_v),
                    Vy  = Array(Vy_v),
                )
                data_c = (;
                    Pdyno= Array(stokes.P) .- Array(reverse(cumsum(reverse((ρg[2]).* di[2], dims=2), dims=2), dims=2)),
                    P   = Array(stokes.P),
                    η   = Array(η_vep),
                    τII = Array(stokes.τ.II),
                    τxx = Array(stokes.τ.xx),
                    τyy = Array(stokes.τ.yy),
                    εII = Array(stokes.ε.II),
                    εII_pl = Array(stokes.ε_pl.II),
                )
                velocity_v = (
                    Array(Vx_v),
                    Array(Vy_v),
                )
                save_marker_chain(
                    joinpath(vtk_dir, "topo_" * lpad("$it", 6, "0")),
                    xvi[1], 
                    Array(chain.h_vertices),
                )
                save_vtk(
                    joinpath(vtk_dir, "vtk_" * lpad("$it", 6, "0")),
                    xvi,
                    xci,
                    data_v,
                    data_c,
                    velocity_v
                )
            end

            # Make particles plottable
            tensor_invariant!(stokes.ε)
            tensor_invariant!(stokes.ε_pl)

            # p        = particles.coords
            # ppx, ppy = p
            # pxv      = ppx.data[:]./1e3
            # pyv      = ppy.data[:]./1e3
            # chain_x, chain_y = chain.coords
            # clr      = pPhases.data[:]
            # # clr      = pT.data[:]
            # idxv     = particles.index.data[:];
            # # Make Makie figure
            # ar  = 3
            # fig = Figure(size = (1200, 600), title = "t = $t")
            # ax1 = Axis(fig[1,1], aspect = ar, title = "T [K]  (t=$(t/(1e6 * 3600 * 24 *365.25)) Myrs)")
            # ax2 = Axis(fig[2,1], aspect = ar, title = "Phase")
            # ax3 = Axis(fig[1,3], aspect = ar, title = "log10(εII)")
            # ax4 = Axis(fig[2,3], aspect = ar, title = "log10(η)")
            # # Plot temperature
            # h1  = CairoMakie.heatmap!(ax1, xvi[1].*1e-3, xvi[2].*1e-3, Array(thermal.T[2:end-1,:]) , colormap=:batlow)
            # # Plot particles phase
            # # h2  = CairoMakie.scatter!(ax2, Array(pxv[idxv]), Array(pyv[idxv]), color=Array(clr[idxv]), markersize = 1)
            # h2 = CairoMakie.heatmap!(ax2, xci[1].*1e-3, xci[2].*1e-3, Array(log10.(stokes.ε_pl.II)) , colormap=:batlow)
            # # Plot 2nd invariant of strain rate
            # h3  = CairoMakie.heatmap!(ax3, xci[1].*1e-3, xci[2].*1e-3, Array(log10.(stokes.ε.II)) , colormap=:batlow)
            # # Plot effective viscosity
            # h4  = CairoMakie.heatmap!(ax4, xci[1].*1e-3, xci[2].*1e-3, Array(log10.(η_vep)) , colormap=:lipari)
            # hidexdecorations!(ax1)
            # hidexdecorations!(ax2)
            # hidexdecorations!(ax3)
            # h = 200
            # # Colorbar(fig[1,2], h1, height = h)
            # # Colorbar(fig[2,2], h2, height = h)
            # # Colorbar(fig[1,4], h3, height = h)
            # # Colorbar(fig[2,4], h4, height = h)
            # linkaxes!(ax1, ax2, ax3, ax4)
            # fig # |> display
            # save(joinpath(figdir, "$(it).png"), fig)
        end
        # ------------------------------

    end

    stokes.ε_pl.xx .= 0.0
    stokes.ε_pl.yy .= 0.0
    stokes.ε_pl.xy .= 0.0
    stokes.ε_pl.xy_c .= 0.0
    stokes.ε_pl.II .= 0.0
    stokes.EII_pl .= 0.0

    dt_2 = 3600 * 24 * 30
    factor = dt_2/dt
    CFL = 0.85 * factor / √2.1
    pt_stokes        = PTStokesCoeffs(li, di; ϵ=1e-4, Re=3π, r=0.7, CFL = CFL) # Re=3π, r=0.7

    dt .= dt_2

    while it < 1700 # run only for 5 Myrs

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

        # args = (; T = thermal.Tc, P = stokes.P,  dt=Inf, cohesion_C = cohesion_damage)
        args = (; T = thermal.Tc, P = stokes.P,  dt=Inf)

        # Stokes solver ----------------
        t_stokes = @elapsed begin
            solve_VariationalStokes!(
                stokes,
                pt_stokes,
                di,
                flow_bcs,
                ρg,
                phase_ratios,
                ϕ_R,
                rheology,
                args,
                dt,
                strain_increment,
                igg;
                kwargs = (;
                iterMax = it > 0 ? 100.0e4 : 250.0e4,
                    free_surface = true,
                    nout = 1e3,
                    viscosity_cutoff = viscosity_cutoff,
                )
            )
        end
        # dtmax = (it < 10 ? 2e3 : 3e3) * 3600 * 24 * 365 # diffusive CFL timestep limiter
        #dt    = compute_dt(stokes, di, dtmax)
        println("Stokes solver time             ")
        println("   Total time:      $t_stokes s")
        println("           Δt:      $(dt / (3600 * 24 * 365)) kyrs")
        # println("   Time/iteration:  $(t_stokes / out.iter) s")
        # ------------------------------

        compute_shear_heating!(
            thermal,
            stokes,
            phase_ratios,
            rheology, # needs to be a tuple
            dt,
        )

        # Thermal solver ---------------
        # update mask of Dirichlet BC
        @parallel (@idx ni .+ 1) update_Dirichlet_mask!(thermal_bc.dirichlet.mask, phase_ratios.vertex, air_phase)
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
                iterMax = 50e4,
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
        advection_MQS!(particles, RungeKutta2(), @velocity(stokes), grid_vxi, dt)
        # advect particles in memory
        move_particles!(particles, xvi, particle_args)
        # check if we need to inject particles
        inject_particles_phase!(particles, pPhases, (pT, ), (T_buffer, ), xvi)

        # advect marker chain
        advect_markerchain!(chain, RungeKutta2(), @velocity(stokes), grid_vxi, dt)
        update_phases_given_markerchain!(pPhases, chain, particles, origin, di, air_phase)

        # update phase ratios
        update_phase_ratios!(phase_ratios, particles, xci, xvi, pPhases)
        update_rock_ratio!(ϕ_R, phase_ratios, air_phase)
  

        @show it += 1
        t        += dt

        # Data I/O and plotting ---------------------
        if it == 1 || rem(it, 1) == 0
            # checkpointing(figdir, stokes, thermal.T, η, t)
            (; η_vep, η) = stokes.viscosity
            if do_vtk
                
                velocity2vertex!(Vx_v, Vy_v, @velocity(stokes)...)
                data_v = (;
                    T   = Array(T_buffer),        
                    Vx  = Array(Vx_v),
                    Vy  = Array(Vy_v),
                )
                data_c = (;
                    Pdyno= Array(stokes.P) .- Array(reverse(cumsum(reverse((ρg[2]).* di[2], dims=2), dims=2), dims=2)),
                    P   = Array(stokes.P),
                    η   = Array(η_vep),
                    τII = Array(stokes.τ.II),
                    τxx = Array(stokes.τ.xx),
                    τyy = Array(stokes.τ.yy),
                    εII = Array(stokes.ε.II),
                    εII_pl = Array(stokes.ε_pl.II),
                )
                velocity_v = (
                    Array(Vx_v),
                    Array(Vy_v),
                )
                save_marker_chain(
                    joinpath(vtk_dir, "topo_" * lpad("$it", 6, "0")),
                    xvi[1], 
                    Array(chain.h_vertices),
                )
                save_vtk(
                    joinpath(vtk_dir, "vtk_" * lpad("$it", 6, "0")),
                    xvi,
                    xci,
                    data_v,
                    data_c,
                    velocity_v
                )
            end

            # Make particles plottable
            tensor_invariant!(stokes.ε)
            tensor_invariant!(stokes.ε_pl)

            # p        = particles.coords
            # ppx, ppy = p
            # pxv      = ppx.data[:]./1e3
            # pyv      = ppy.data[:]./1e3
            # chain_x, chain_y = chain.coords
            # clr      = pPhases.data[:]
            # # clr      = pT.data[:]
            # idxv     = particles.index.data[:];
            # # Make Makie figure
            # ar  = 3
            # fig = Figure(size = (1200, 600), title = "t = $t")
            # ax1 = Axis(fig[1,1], aspect = ar, title = "T [K]  (t=$(t/(1e6 * 3600 * 24 *365.25)) Myrs)")
            # ax2 = Axis(fig[2,1], aspect = ar, title = "Phase")
            # ax3 = Axis(fig[1,3], aspect = ar, title = "log10(εII)")
            # ax4 = Axis(fig[2,3], aspect = ar, title = "log10(η)")
            # # Plot temperature
            # h1  = CairoMakie.heatmap!(ax1, xvi[1].*1e-3, xvi[2].*1e-3, Array(thermal.T[2:end-1,:]) , colormap=:batlow)
            # # Plot particles phase
            # # h2  = CairoMakie.scatter!(ax2, Array(pxv[idxv]), Array(pyv[idxv]), color=Array(clr[idxv]), markersize = 1)
            # h2 =  CairoMakie.heatmap!(ax2, xci[1].*1e-3, xci[2].*1e-3, Array(log10.(stokes.ε_pl.II)) , colormap=:batlow)
            # # Plot 2nd invariant of strain rate
            # h3  = CairoMakie.heatmap!(ax3, xci[1].*1e-3, xci[2].*1e-3, Array(log10.(stokes.ε.II)) , colormap=:batlow)
            # # Plot effective viscosity
            # h4  = CairoMakie.heatmap!(ax4, xci[1].*1e-3, xci[2].*1e-3, Array(log10.(η_vep)) , colormap=:lipari)
            # hidexdecorations!(ax1)
            # hidexdecorations!(ax2)
            # hidexdecorations!(ax3)
            # h = 200
            # # Colorbar(fig[1,2], h1, height = h)
            # # Colorbar(fig[2,2], h2, height = h)
            # # Colorbar(fig[1,4], h3, height = h)
            # # Colorbar(fig[2,4], h4, height = h)
            # linkaxes!(ax1, ax2, ax3, ax4)
            # fig # |> display
            # save(joinpath(figdir, "$(it).png"), fig)
        end
        # ------------------------------

    end

    return nothing
end


main(li, origin, phases_GMG, igg; figdir = figdir, nx = nx, ny = ny, do_vtk = do_vtk);



# p        = particles.coords
# ppx, ppy = p
# pxv      = ppx.data[:]./1e3
# pyv      = ppy.data[:]./1e3
# chain_x, chain_y = chain.coords
# x = filter(!isnan, Array(chain_x.data[:]))
# y = filter(!isnan, Array(chain_y.data[:]))
# scatter(x, y, markersize = 30)

# scatter(Array(chain.h_vertices))
# scatter!(Array(chain.h_vertices0))