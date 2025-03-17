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

function compute_maxloc!(B, A; window = (1, 1, 1))
    ni = size(A)

    @parallel_indices (I...) function _maxloc!(B, A, window)
        B[I...] = _maxloc_window_clamped(A, I..., window...)
        return nothing
    end

    @parallel (@idx ni) _maxloc!(B, A, window)
    return nothing
end

@inline function _maxloc_window_clamped(A, I, J, width_x, width_y)
    nx, ny = size(A)
    I_range = (I - width_x):(I + width_x)
    J_range = (J - width_y):(J + width_y)
    x = -Inf

    for j in J_range
        jj = clamp(j, 1, ny) # handle boundary cells
        for i in I_range
            ii = clamp(i, 1, nx) # handle boundary cells
            Aij = A[ii, jj]
            if Aij > x
                x = Aij
            end
        end
    end
    return x
end


#include("../src/common.jl")
#include("../src/Utils.jl")
# Initialize
n        = 200
nx, ny   = n, n ÷ 2
# li, origin, phases_GMG, T_GMG = Setup_Topo(nx+1, ny+1)
li, origin, phases_GMG, T_GMG = flat_setup(nx+1, ny+1)
do_vtk   = true # set to true to generate VTK files for ParaView
figdir   = "Rift2D_displacement"

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
dt                  =  1e3 * 3600 * 24 * 365 # diffusive CFL timestep limiter
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

εbg = +1e-14 # background strain rate
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

# while it < 5000  && t < 1.5768e+14 # run only for 5 Myrs

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


    # Stokes solver ----------------
    # Arguments
    iterMax              = 50e3;
    nout                 = 1e3;
    viscosity_cutoff     = viscosity_cutoff;
    free_surface         = false;
    viscosity_relaxation = 1e-3;

    # _solve!
    # unpack
    _di = inv.(di); #size_steps
    _dt = inv(dt); # 1/dt
    (; ϵ, r, θ_dτ, ηdτ) = pt_stokes; # Pseudo transient parameters
    (; η, η_vep) = stokes.viscosity # Viscosity withouth plasticity and with plasticity 
    ni = size(stokes.P) # grid size

    # ~preconditioner
    ητ = deepcopy(η) # Viscosity withouth plasticity

    compute_maxloc!(ητ, η; window = (1, 1)) # Compute maxloc of viscosity inside the window
    update_halo!(ητ) # WHAT IS THIS?
    # end

    # errors
    error = 2 * ϵ # error
    iter = 0 # iteration counter
    err_evo1 = Float64[] # error evolution
    err_evo2 = Float64[] 
    norm_Rx = Float64[] # norm of the residual
    norm_Ry = Float64[]
    norm_∇V = Float64[]
    
    sizehint!(norm_Rx, Int(iterMax)) # sizehint! is a function that pre-allocates memory for the array
    sizehint!(norm_Ry, Int(iterMax))
    sizehint!(norm_∇V, Int(iterMax))
    sizehint!(err_evo1, Int(iterMax))
    sizehint!(err_evo2, Int(iterMax))

    # solver loop
    @copy stokes.P0 stokes.P # copy initial pressure to pressure
    wtime0 = 0.0 # start time
    relλ = 0.2 # relaxation factor
    θ = deepcopy(stokes.P) # pressure
    λ = @zeros(ni...) # plastic multiplier
    λv = @zeros(ni .+ 1...) # plastic multiplier ? 
    η0 = deepcopy(η) # viscosity
    do_visc = true # do viscosity

    for Aij in @tensor_center(stokes.ε_pl)
        Aij .= 0.0 # plastic strain rate to 0
    end
    Vx_on_Vy = @zeros(size(stokes.V.Vy)) # velocity x on velocity y ? 

    # compute buoyancy forces and viscosity
    compute_ρg!(ρg, phase_ratios, rheology, args) # compute buoyancy forces
    compute_viscosity!(stokes, phase_ratios, args, rheology, viscosity_cutoff) # compute viscosity
    displacement2velocity!(stokes, dt, flow_bcs) # displacement to velocity

    # Pseudo iterations

    compute_maxloc!(ητ, η; window = (1, 1))
    update_halo!(ητ)

    ## MiniKernels Functions 

    ## 2D mini kernels
    const T2 = AbstractArray{T, 2} where {T} # 2D array

    # finite differences
    @inline _d_xa(A::T, i, j, _dx) where {T <: T2} = (-A[i, j] + A[i + 1, j]) * _dx # finite differences in x
    @inline _d_ya(A::T, i, j, _dy) where {T <: T2} = (-A[i, j] + A[i, j + 1]) * _dy # finite differences in y
    @inline _d_xi(A::T, i, j, _dx) where {T <: T2} = (-A[i, j + 1] + A[i + 1, j + 1]) * _dx # finite differences in x
    @inline _d_yi(A::T, i, j, _dy) where {T <: T2} = (-A[i + 1, j] + A[i + 1, j + 1]) * _dy # finite differences in y
    

    ## Velocity and Displacement Gradients 
    include("../src/stokes/VelocityKernels.jl")
    @parallel (@idx ni) compute_∇V!(stokes.∇V, @velocity(stokes)..., _di...) # compute velocity gradient


    include("../src/MiniKernels.jl") 

    ## Pressure Solver

    include("../src/stokes/PressureKernels.jl")

    get_bulk_modulus = get_bulkmodulus # get bulk modulus

    compute_P!(
                θ,
                stokes.P0,
                stokes.R.RP,
                stokes.∇V,
                stokes.Q,
                ητ,
                rheology,
                phase_ratios,
                dt,
                r,
                θ_dτ,
                args,
            ) # compute pressure

    ## Buoyancy Forces
    
    #include("../src/rheology/BuoyancyForces.jl")

    import JustRelax.JustRelax2D: compute_ρg!, update_ρg!
    update_ρg!(ρg, phase_ratios, rheology, args) # update buoyancy forces


    #include("../src/Utils.jl")
    @parallel (@idx ni .+ 1) compute_strain_rate!(
        @strain(stokes)..., stokes.∇V, @velocity(stokes)..., _di...
            )
        @parallel (@idx ni .+ 1) compute_strain_rate!(
            @strain_increment(stokes)..., stokes.∇U, @displacement(stokes)..., _di...
                )

    ## Input for update_stresses_center_vertex_ps!

    function clamped_indices(ni::NTuple{2, Integer}, i, j)
        nx, ny = ni
        i0 = clamp(i - 1, 1, nx)
        ic = clamp(i, 1, nx)
        j0 = clamp(j - 1, 1, ny)
        jc = clamp(j, 1, ny)
        return i0, j0, ic, jc
    end # clamped indices
    
    function av_clamped(A, i0, j0, ic, jc)
        return 0.25 * (A[i0, j0] + A[ic, jc] + A[i0, jc] + A[ic, j0])
    end # average clamped


    ε = @strain(stokes); # strain rate tensor εxx and εyy at center, εxy at vertex
    ε_pl = @tensor_center(stokes.ε_pl); # plastic strain rate at center
    EII = stokes.EII_pl; # second invariant of the strain rate (plastic?) at center
    τ = @tensor_center(stokes.τ);  # stress tensor τxx, τyy and τxy at center
    τshear_v = (stokes.τ.xy,); # shear stress tensor τxy at vertices
    τ_o = @tensor_center(stokes.τ_o); # old stress tensor τxx, τyy and τxy at center
    τshear_ov = (stokes.τ_o.xy,); # old shear stress tensor τxy at vertices
    Pr = θ; # pressure at center
    Pr_c = stokes.P; # pressure at center (difference with the previous one?)
    η = stokes.viscosity.η; # viscosity at center
    λ; # plastic multiplier at center
    λv; # plastic multiplier at vertices
    τII = stokes.τ.II; # second invariant of the stress tensor at center
    η_vep = stokes.viscosity.η_vep # viscosity with plasticity at center
    relλ; # relaxation factor
    dt; # time step
    θ_dτ; # pseudo transient parameter
    rheology;
    phace_center = phase_ratios.center;
    phase_vertex = phase_ratios.vertex;
    phase_xy = phase_ratios.xy;
    phase_yx = phase_ratios.yz;
    phase_xz = phase_ratios.xz;

    I = [3,3]

    τxyv = τshear_v[1]
    τxyv_old = τshear_ov[1]
    ni = size(Pr)
    Ic = clamped_indices(ni, I...)

    # interpolate to ith vertex (i,j) as average of the surrounding cell centers
    Pv_ij = av_clamped(Pr, Ic...) # pressure at vertex
    εxxv_ij = av_clamped(ε[1], Ic...) # strain rate tensor εxx at vertex
    εyyv_ij = av_clamped(ε[2], Ic...) # strain rate tensor εyy at vertex
    τxxv_ij = av_clamped(τ[1], Ic...) # stress tensor τxx at vertex
    τyyv_ij = av_clamped(τ[2], Ic...) # stress tensor τyy at vertex
    τxxv_old_ij = av_clamped(τ_o[1], Ic...) # old stress tensor τxx at vertex
    τyyv_old_ij = av_clamped(τ_o[2], Ic...) # old stress tensor τyy at vertex
    EIIv_ij = av_clamped(EII, Ic...) # second invariant of the strain rate at vertex

    include("../src/rheology/StressUpdate.jl")
    get_shear_modulus = get_shearmodulus
    
    phase = @inbounds phase_vertex[I...] # phase at vertex
    is_pl, Cv, sinϕv, cosϕv, sinψv, η_regv = plastic_params_phase(rheology, EIIv_ij, phase) # plastic parameters
    _Gv = inv(fn_ratio(get_shear_modulus, rheology, phase) ) # inverse of the shear modulus times the time step
    Kv = fn_ratio(get_bulk_modulus, rheology, phase) # bulk modulus
    volumev = isinf(Kv) ? 0.0 : Kv * dt * sinϕv * sinψv # plastic volumetric change K * dt * sinϕ * sinψ
    ηv_ij = av_clamped(η, Ic...) # viscosity at vertex
    dτ_rv = inv(θ_dτ + ηv_ij * _Gvdt + dt) 

    function compute_stress_increment(τij::Real, τij_o::Real, ηij, Δεij::Real, _G, dτ_r)
        dτij = dτ_r * fma(2.0 * ηij, Δεij, fma(-(τij - τij_o) * ηij, _G, -τij*dt))
        return dτij
    end
    

    # stress increments @ vertex
    dτxxv = compute_stress_increment(τxxv_ij, τxxv_old_ij, ηv_ij, Δεxxv_ij, _Gv, dτ_rv)
    dτyyv = compute_stress_increment(τyyv_ij, τyyv_old_ij, ηv_ij, Δεyyv_ij, _Gv, dτ_rv)
    dτxyv = compute_stress_increment(
        τxyv[I...], τxyv_old[I...], ηv_ij, Δε[3][I...], _Gvdt, dτ_rv
    )
    τIIv_ij = √(0.5 * ((τxxv_ij + dτxxv)^2 + (τyyv_ij + dτyyv)^2) + (τxyv[I...] + dτxyv)^2)
    

    # yield function @ center
    Fv = τIIv_ij - Cv * cosϕv - max(Pv_ij, 0.0) * sinϕv
    if is_pl && !iszero(τIIv_ij) && Fv > 0
        # stress correction @ vertex
        λv[I...] =
            (1.0 - relλ) * λv[I...] +
            relλ * (max(Fv, 0.0) / (ηv_ij * dτ_rv + η_regv + volumev))
        dQdτxy = 0.5 * (τxyv[I...] + dτxyv) / τIIv_ij 
        τxyv[I...] += dτxyv - 2.0 * ηv_ij * 0.5 * λv[I...] * dQdτxy * dτ_rv
    else
        # stress correction @ vertex
        τxyv[I...] += dτxyv
    end

    #if all(I .≤ ni)
        # Material properties
        phase = @inbounds phase_center[I...]
        _Gdt = inv(fn_ratio(get_shear_modulus, rheology, phase) * dt)
        is_pl, C, sinϕ, cosϕ, sinψ, η_reg = plastic_params_phase(rheology, EII[I...], phase)
        K = fn_ratio(get_bulk_modulus, rheology, phase)
        volume = isinf(K) ? 0.0 : K * dt * sinϕ * sinψ # plastic volumetric change K * dt * sinϕ * sinψ
        ηij = η[I...]
        dτ_r = 1.0 / (θ_dτ + ηij * _Gdt + 1.0)

        # cache strain rates for center calculations
        τij, τij_o, εij = cache_tensors(τ, τ_o, ε, I...)

        # visco-elastic strain rates @ center
        εij_ve = @. εij + 0.5 * τij_o * _Gdt
        εII_ve = GeoParams.second_invariant(εij_ve)
        # stress increments @ center
        # dτij = @. (-(τij - τij_o) * ηij * _Gdt - τij .+ 2.0 * ηij * εij) * dτ_r
        dτij = compute_stress_increment(τij, τij_o, ηij, εij, _Gdt, dτ_r)
        τII_ij = GeoParams.second_invariant(dτij .+ τij)
        # yield function @ center
        F = τII_ij - C * cosϕ - max(Pr[I...], 0.0) * sinϕ

        if is_pl && !iszero(τII_ij) && F > 0
            # stress correction @ center
            λ[I...] =
                (1.0 - relλ) * λ[I...] +
                relλ * (max(F, 0.0) / (η[I...] * dτ_r + η_reg + volume))
            dQdτij = @. 0.5 * (τij + dτij) / τII_ij
            # dτij        = @. (-(τij - τij_o) * ηij * _Gdt - τij .+ 2.0 * ηij * (εij  - λ[I...] *dQdτij )) * dτ_r
            εij_pl = λ[I...] .* dQdτij
            dτij = @. dτij - 2.0 * ηij * εij_pl * dτ_r
            τij = dτij .+ τij
            setindex!.(τ, τij, I...)
            setindex!.(ε_pl, εij_pl, I...)
            τII[I...] = GeoParams.second_invariant(τij)
            # Pr_c[I...] = Pr[I...] + K * dt * λ[I...] * sinψ
            η_vep[I...] = 0.5 * τII_ij / εII_ve
        else
            # stress correction @ center
            setindex!.(τ, dτij .+ τij, I...)
            η_vep[I...] = ηij
            τII[I...] = τII_ij
        end

        Pr_c[I...] = Pr[I...] + (isinf(K) ? 0.0 : K * dt * λ[I...] * sinψ)
    end











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
            igg;
            kwargs = (
                iterMax              = 50e3,
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

        if it > 1
            strain_rate = (strain - Array(stokes.ε.II))./dt
        end
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
        
        strain = Array(stokes.ε.II)
        # p        = particles.coords
        # ppx, ppy = p
        # pxv      = ppx.data[:]./1e3
        # pyv      = ppy.data[:]./1e3
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
        # h1  = GLMakie.heatmap!(ax1, xvi[1].*1e-3, xvi[2].*1e-3, Array(thermal.T[2:end-1,:]) , colormap=:batlow)
        # # Plot particles phase
        # # h2  = GLMakie.scatter!(ax2, Array(pxv[idxv]), Array(pyv[idxv]), color=Array(clr[idxv]), markersize = 1)
        # h2 = GLMakie.heatmap!(ax2, xci[1].*1e-3, xci[2].*1e-3, Array(log10.(stokes.ε_pl.II)) , colormap=:batlow)
        # # Plot 2nd invariant of strain rate
        # h3  = GLMakie.heatmap!(ax3, xci[1].*1e-3, xci[2].*1e-3, Array(log10.(stokes.ε.II)) , colormap=:batlow)
        # # Plot effective viscosity
        # h4  = GLMakie.heatmap!(ax4, xci[1].*1e-3, xci[2].*1e-3, Array(log10.(η_vep)) , colormap=:lipari)
        # hidexdecorations!(ax1)
        # hidexdecorations!(ax2)
        # hidexdecorations!(ax3)
        # h = 200
        # Colorbar(fig[1,2], h1, height = h)
        # Colorbar(fig[2,2], h2, height = h)
        # Colorbar(fig[1,4], h3, height = h)
        # Colorbar(fig[2,4], h4, height = h)
        # linkaxes!(ax1, ax2, ax3, ax4)
        # fig
        # save(joinpath(figdir, "$(it).png"), fig)
    end
    # ------------------------------

end

#return nothing


## END OF MAIN SCRIPT ----------------------------------------------------------------
do_vtk   = true # set to true to generate VTK files for ParaView
figdir   = "Rift2D_displacement"


main(li, origin, phases_GMG, igg; figdir = figdir, nx = nx, ny = ny, do_vtk = do_vtk);
