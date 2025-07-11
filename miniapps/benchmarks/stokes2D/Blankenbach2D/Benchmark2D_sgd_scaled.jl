using JustRelax, JustRelax.JustRelax2D, JustRelax.DataIO
const backend_JR = CPUBackend

using ParallelStencil, ParallelStencil.FiniteDifferences2D
@init_parallel_stencil(Threads, Float64, 2) #or (CUDA, Float64, 2) or (AMDGPU, Float64, 2)

using JustPIC
using JustPIC._2D
# Threads is the default backend,
# to run on a CUDA GPU load CUDA.jl (i.e. "using CUDA") at the beginning of the script,
# and to run on an AMD GPU load AMDGPU.jl (i.e. "using AMDGPU") at the beginning of the script.
const backend = JustPIC.CPUBackend # Options: CPUBackend, CUDABackend, AMDGPUBackend

# Load script dependencies
using Printf, LinearAlgebra, GeoParams, GLMakie

# Load file with all the rheology configurations
include("Blankenbach_Rheology_scaled.jl")

## SET OF HELPER FUNCTIONS PARTICULAR FOR THIS SCRIPT --------------------------------
function copyinn_x!(A, B)
    @parallel function f_x(A, B)
        @all(A) = @inn_x(B)
        return nothing
    end
    return @parallel f_x(A, B)
end

# Initial thermal profile
@parallel_indices (i, j) function init_T!(T, y)
    T[i, j] = 1 - y[j]
    return nothing
end

# Thermal rectangular perturbation
function rectangular_perturbation!(T, xc, yc, r, xvi)
    @parallel_indices (i, j) function _rectangular_perturbation!(T, xc, yc, r, x, y)
        if ((x[i] - xc)^2 ≤ r^2) && ((y[j] - yc)^2 ≤ r^2)
            T[i + 1, j] += 0.2
        end
        return nothing
    end
    nx, ny = size(T)
    @parallel (1:(nx - 2), 1:ny) _rectangular_perturbation!(T, xc, yc, r, xvi...)
    return nothing
end
## END OF HELPER FUNCTION ------------------------------------------------------------

## BEGIN OF MAIN SCRIPT --------------------------------------------------------------
function main2D(igg; ar = 1, nx = 32, ny = 32, nit = 1.0e1, figdir = "figs2D", do_vtk = false)

    # Physical domain ------------------------------------
    ly = 1.0                  # domain length in y
    lx = ly                   # domain length in x
    ni = nx, ny               # number of cells
    li = lx, ly               # domain length in x- and y-
    di = @. li / ni           # grid step in x- and -y
    origin = 0.0, 0.0             # origin coordinates
    grid = Geometry(ni, li; origin = origin)
    (; xci, xvi) = grid # nodes at the center and vertices of the cells
    # ----------------------------------------------------

    # Physical properties using GeoParams ----------------
    rheology = init_rheologies()
    dt = dt_diff = 0.9 * min(di...)^2 / 4.0 # diffusive CFL timestep limiter
    # ----------------------------------------------------

    # Initialize particles -------------------------------
    nxcell, max_xcell, min_xcell = 24, 36, 12
    particles = init_particles(
        backend, nxcell, max_xcell, min_xcell, xvi, di, ni
    )
    subgrid_arrays = SubgridDiffusionCellArrays(particles)
    # velocity grids
    grid_vx, grid_vy = velocity_grids(xci, xvi, di)
    # temperature
    pT, pT0, pPhases = init_cell_arrays(particles, Val(3))
    particle_args = (pT, pT0, pPhases)
    phase_ratios = PhaseRatios(backend, length(rheology), ni)
    init_phases!(pPhases, particles)
    update_phase_ratios!(phase_ratios, particles, xci, xvi, pPhases)
    # ----------------------------------------------------

    # STOKES ---------------------------------------------
    # Allocate arrays needed for every Stokes problem
    stokes = StokesArrays(backend_JR, ni)
    pt_stokes = PTStokesCoeffs(li, di; ϵ_abs = 1.0e-4, ϵ_rel = 1.0e-4, CFL = 1 / √2.1)
    # ----------------------------------------------------

    # TEMPERATURE PROFILE --------------------------------
    thermal = ThermalArrays(backend_JR, ni)
    thermal_bc = TemperatureBoundaryConditions(;
        no_flux = (left = true, right = true, top = false, bot = false),
    )
    # initialize thermal profile
    @parallel (@idx size(thermal.T)) init_T!(thermal.T, xvi[2])
    # Elliptical temperature anomaly
    xc_anomaly = 0.0    # origin of thermal anomaly
    yc_anomaly = 1 / 3  # origin of thermal anomaly
    r_anomaly = 0.1 / 2    # radius of perturbation
    rectangular_perturbation!(thermal.T, xc_anomaly, yc_anomaly, r_anomaly, xvi)
    thermal_bcs!(thermal, thermal_bc)
    thermal.Told .= thermal.T
    temperature2center!(thermal)
    # ----------------------------------------------------

    # Rayleigh number ------------------------------------
    Ra = rheology[1].Gravity[1].g
    println("Ra = $Ra")

    args = (; T = thermal.Tc, P = stokes.P, dt = Inf)

    # Buoyancy forces  & viscosity ----------------------
    ρg = @zeros(ni...), @zeros(ni...)
    η = @ones(ni...)
    compute_ρg!(ρg[2], phase_ratios, rheology, args)
    compute_viscosity!(
        stokes, phase_ratios, args, rheology, (-Inf, Inf)
    )

    # PT coefficients for thermal diffusion -------------
    pt_thermal = PTThermalCoeffs(
        backend_JR, rheology, phase_ratios, args, dt, ni, di, li; ϵ = 1.0e-5, CFL = 0.5 / √2.1
    )

    # Boundary conditions -------------------------------
    flow_bcs = VelocityBoundaryConditions(;
        free_slip = (left = true, right = true, top = true, bot = true),
    )
    flow_bcs!(stokes, flow_bcs) # apply boundary conditions
    update_halo!(@velocity(stokes)...)

    # IO ------------------------------------------------
    # if it does not exist, make folder where figures are stored
    if do_vtk
        vtk_dir = joinpath(figdir, "vtk")
        take(vtk_dir)
    end
    take(figdir)
    # ----------------------------------------------------

    # Plot initial T and η profiles-----------------------
    fig = let
        Yv = [y for x in xvi[1], y in xvi[2]][:]
        Y = [y for x in xci[1], y in xci[2]][:]
        fig = Figure(size = (1200, 900))
        ax1 = Axis(fig[1, 1], aspect = 2 / 3, title = "T")
        ax2 = Axis(fig[1, 2], aspect = 2 / 3, title = "log10(η)")
        scatter!(ax1, Array(thermal.T[2:(end - 1), :][:]), (1 .- Yv))
        scatter!(ax2, Array(log10.(η[:])), (1 .- Y))
        ylims!(ax1, maximum(xvi[2]), 0)
        ylims!(ax2, maximum(xvi[2]), 0)
        hideydecorations!(ax2)
        save(joinpath(figdir, "initial_profile.png"), fig)
        fig
    end

    T_buffer = @zeros(ni .+ 1)
    Told_buffer = similar(T_buffer)
    dt₀ = similar(stokes.P)
    for (dst, src) in zip((T_buffer, Told_buffer), (thermal.T, thermal.Told))
        copyinn_x!(dst, src)
    end
    grid2particle!(pT, xvi, T_buffer, particles)
    pT0.data .= pT.data

    local Vx_v, Vy_v
    if do_vtk
        Vx_v = @zeros(ni .+ 1...)
        Vy_v = @zeros(ni .+ 1...)
    end
    # Time loop
    t, it = 0.0, 1
    Urms = Float64[]
    Nu_top = Float64[]
    trms = Float64[]

    # Buffer arrays to compute velocity rms
    Vx_v = @zeros(ni .+ 1...)
    Vy_v = @zeros(ni .+ 1...)

    while it ≤ nit

        # Update buoyancy and viscosity -
        args = (; T = thermal.Tc, P = stokes.P, dt = Inf)
        compute_viscosity!(stokes, phase_ratios, args, rheology, (-Inf, Inf))
        compute_ρg!(ρg[2], phase_ratios, rheology, args)
        # ------------------------------

        # Stokes solver ----------------
        solve!(
            stokes,
            pt_stokes,
            di,
            flow_bcs,
            ρg,
            phase_ratios,
            rheology,
            args,
            Inf,
            igg;
            kwargs = (;
                iterMax = 150.0e3,
                nout = 200,
                viscosity_cutoff = (-Inf, Inf),
                verbose = true,
            )
        )
        dt = compute_dt(stokes, di, dt_diff)
        # ------------------------------

        # Thermal solver ---------------
        heatdiffusion_PT!(
            thermal,
            pt_thermal,
            thermal_bc,
            rheology,
            args,
            dt,
            di;
            kwargs = (;
                igg = igg,
                phase = phase_ratios,
                iterMax = 10.0e3,
                nout = 1.0e2,
                verbose = true,
            )
        )
        # subgrid diffusion
        for (dst, src) in zip((T_buffer, Told_buffer), (thermal.T, thermal.Told))
            copyinn_x!(dst, src)
        end
        subgrid_characteristic_time!(
            subgrid_arrays, particles, dt₀, phase_ratios, rheology, thermal, stokes, xci, di
        )
        centroid2particle!(subgrid_arrays.dt₀, xci, dt₀, particles)
        subgrid_diffusion!(
            pT, T_buffer, thermal.ΔT[2:(end - 1), :], subgrid_arrays, particles, xvi, di, dt
        )
        # ------------------------------

        # Advection --------------------
        # advect particles in space
        advection!(particles, RungeKutta2(), @velocity(stokes), (grid_vx, grid_vy), dt)
        # advect particles in memory
        move_particles!(particles, xvi, particle_args)
        # check if we need to inject particles
        inject_particles_phase!(particles, pPhases, (pT,), (T_buffer,), xvi)
        # update phase ratios
        update_phase_ratios!(phase_ratios, particles, xci, xvi, pPhases)

        # Nusselt number, Nu = ∫ ∂T/∂z dx ----
        Nu_it = sum(((abs.(thermal.T[2:(end - 1), end] - thermal.T[2:(end - 1), end - 1])) ./ di[2]) .* di[1])
        push!(Nu_top, Nu_it)
        # -------------------------------------------

        # Compute U rms -----------------------------
        # U₍ᵣₘₛ₎ = √ ∫∫ (vx²+vz²) dx dz
        Urms_it = let
            velocity2vertex!(Vx_v, Vy_v, stokes.V.Vx, stokes.V.Vy)
            @. Vx_v .= hypot.(Vx_v, Vy_v) # we reuse Vx_v to store the velocity magnitude
            sqrt(sum(Vx_v .^ 2 .* prod(di)))
        end
        push!(Urms, Urms_it)
        push!(trms, t)
        # -------------------------------------------

        # interpolate fields from particle to grid vertices
        particle2grid!(T_buffer, pT, xvi, particles)
        @views T_buffer[:, end] .= 0.0
        @views T_buffer[:, 1] .= 1.0
        @views thermal.T[2:(end - 1), :] .= T_buffer
        flow_bcs!(stokes, flow_bcs) # apply boundary conditions
        temperature2center!(thermal)
        @show extrema(thermal.T)
        any(isnan.(thermal.T)) && break

        # Data I/O and plotting ---------------------
        if it == 1 || rem(it, 200) == 0 || it == nit || any(isnan.(thermal.T))

            if do_vtk
                velocity2vertex!(Vx_v, Vy_v, @velocity(stokes)...)
                data_v = (;
                    T = Array(ustrip.(dimensionalize(thermal.T[2:(end - 1), :], C, CharDim))),
                    τxy = Array(ustrip.(dimensionalize(stokes.τ.xy, s^-1, CharDim))),
                    εxy = Array(ustrip.(dimensionalize(stokes.ε.xy, s^-1, CharDim))),
                    Vx = Array(ustrip.(dimensionalize(Vx_v, cm / yr, CharDim))),
                    Vy = Array(ustrip.(dimensionalize(Vy_v, cm / yr, CharDim))),
                )
                data_c = (;
                    P = Array(ustrip.(dimensionalize(stokes.P, MPa, CharDim))),
                    τxx = Array(ustrip.(dimensionalize(stokes.τ.xx, MPa, CharDim))),
                    τyy = Array(ustrip.(dimensionalize(stokes.τ.yy, MPa, CharDim))),
                    τII = Array(ustrip.(dimensionalize(stokes.τ.II, MPa, CharDim))),
                    εxx = Array(ustrip.(dimensionalize(stokes.ε.xx, s^-1, CharDim))),
                    εyy = Array(ustrip.(dimensionalize(stokes.ε.yy, s^-1, CharDim))),
                    εII = Array(ustrip.(dimensionalize(stokes.ε.II, s^-1, CharDim))),
                    η = Array(ustrip.(dimensionalize(stokes.viscosity.η, Pa * s, CharDim))),
                )
                velocity_v = (
                    Array(ustrip.(dimensionalize(Vx_v, cm / yr, CharDim))),
                    Array(ustrip.(dimensionalize(Vy_v, cm / yr, CharDim))),
                )
                save_vtk(
                    joinpath(vtk_dir, "vtk_" * lpad("$it", 6, "0")),
                    xvi,
                    xci,
                    data_v,
                    data_c,
                    velocity_v,
                    t = t
                )
            end

            # Make particles plottable
            p = particles.coords
            ppx, ppy = p
            pxv = ppx.data[:]
            pyv = ppy.data[:]
            clr = pT.data[:] #pPhases.data[:]
            idxv = particles.index.data[:]

            # Make Makie figure
            fig = Figure(size = (900, 900), title = "t = $t")
            ax1 = Axis(fig[1, 1], aspect = ar, title = "T [K]  (t=$(t / (1.0e6 * 3600 * 24 * 365.25)) Myrs)")
            ax2 = Axis(fig[2, 1], aspect = ar, title = "Vy [m/s]")
            ax3 = Axis(fig[1, 3], aspect = ar, title = "Vx [m/s]")
            ax4 = Axis(fig[2, 3], aspect = ar, title = "T [K]")
            #
            h1 = heatmap!(ax1, xvi[1], xvi[2], Array(thermal.T[2:(end - 1), :]), colormap = :lajolla, colorrange = (0, 1))
            #
            h2 = heatmap!(ax2, xvi[1], xvi[2], Array(stokes.V.Vy), colormap = :batlow)
            #
            h3 = heatmap!(ax3, xvi[1], xvi[2], Array(stokes.V.Vx), colormap = :batlow)
            #
            # h4  = scatter!(ax4, Array(pxv[idxv]), Array(pyv[idxv]), markersize=3)
            h4 = scatter!(ax4, Array(pxv[idxv]), Array(pyv[idxv]), color = Array(clr[idxv]), colormap = :lajolla, colorrange = (0, 1), markersize = 3)
            hidexdecorations!(ax1)
            hidexdecorations!(ax2)
            hidexdecorations!(ax3)
            Colorbar(fig[1, 2], h1)
            Colorbar(fig[2, 2], h2)
            Colorbar(fig[1, 4], h3)
            Colorbar(fig[2, 4], h4)
            linkaxes!(ax1, ax2, ax3, ax4)
            save(joinpath(figdir, "$(it).png"), fig)
            fig

            fig2 = Figure(size = (900, 1200), title = "Time Series")
            ax21 = Axis(fig2[1, 1], aspect = 3, title = L"V_{RMS}")
            ax22 = Axis(fig2[2, 1], aspect = 3, title = L"Nu_{top}")
            l1 = lines!(ax21, trms, (Urms))
            l2 = lines!(ax22, trms, (Nu_top))
            save(joinpath(figdir, "Time_Series_V_Nu.png"), fig2)

            cmap = ([:white, :white, :white, :white])
            fig3 = Figure(size = (900, 900), title = "t = $t")
            ax = Axis(fig3[1, 1], aspect = ar, title = "T [K]  (t=$(t / (1.0e6 * 3600 * 24 * 365.25)) Myrs)")
            h1 = heatmap!(ax, xvi..., thermal.T[2:(end - 1), :], colormap = :lipari, colorrange = (0, 1))
            contour!(ax, xvi..., thermal.T[2:(end - 1), :], linewidth = 5, levels = 0.2:0.2:0.8, colormap = cmap)
            Colorbar(fig3[1, 2], h1)
            save(joinpath(figdir, "Temp.png"), fig3)
            fig3
        end
        it += 1
        t += dt
        # ------------------------------
    end

    # Horizontally averaged depth profile
    Tmean = @zeros(ny + 1)
    Emean = @zeros(ny)

    let
        for j in 1:(ny + 1)
            Tmean[j] = sum(thermal.T[2:(end - 1), j]) / (nx + 1)
        end
        for j in 1:ny
            Emean[j] = sum(η[:, j]) / nx
        end
        Y = [y for x in xci[1], y in xci[2]][:]
        fig = Figure(size = (1200, 900))
        ax1 = Axis(fig[1, 1], aspect = 2 / 3, title = "⟨T⟩")
        ax2 = Axis(fig[1, 2], aspect = 2 / 3, title = "⟨log10(η)⟩")
        lines!(ax1, Tmean, (1 .- xvi[2]))
        lines!(ax2, log10.(Emean), (1 .- xci[2]))
        ylims!(ax1, maximum(xvi[2]), 0)
        ylims!(ax2, maximum(xvi[2]), 0)
        hideydecorations!(ax2)
        save(joinpath(figdir, "Mean_profiles_$(it).png"), fig)
        fig
    end

    @show Urms[Int64(nit)] Nu_top[Int64(nit)]

    return nothing
end
## END OF MAIN SCRIPT ----------------------------------------------------------------

# (Path)/folder where output data and figures are stored
figdir = "Blankenbach_subgrid_scaled"
do_vtk = false # set to true to generate VTK files for ParaView
ar = 1 # aspect ratio
n = 64
nx = n
ny = n
nit = 2.0e3 #6e3
igg = if !(JustRelax.MPI.Initialized()) # initialize (or not) MPI grid
    IGG(init_global_grid(nx, ny, 1; init_MPI = true)...)
else
    igg
end

# run main script
main2D(igg; figdir = figdir, ar = ar, nx = nx, ny = ny, nit = nit, do_vtk = do_vtk);
