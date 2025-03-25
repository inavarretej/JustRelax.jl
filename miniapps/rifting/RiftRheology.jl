# from "Fingerprinting secondary mantle plumes", Cloetingh et al. 2022

function init_rheologies()

    # Dislocation and Diffusion creep
 
    disl_lithospheric_mantle    = DislocationCreep(A=2.51e-17, n=3.5, E=530e3, V=6e-6,  r=0.0, R=8.3145)
    diff_lithospheric_mantle    = DiffusionCreep(A=2.51e-17, n=1.0, E=0, V=0,  p=0, r=0.0, R=8.3145)

    # rheologies from the GeoParams database
    disl_upper_crust = GeoParams.Dislocation.SetDislocationCreep(
        # GeoParams.Dislocation.wet_quartzite_Ueda_2008
        GeoParams.Dislocation.strong_diabase_Mackwell_1998
    )
    disl_lower_crust = GeoParams.Dislocation.SetDislocationCreep(
        GeoParams.Dislocation.wet_anorthite_Rybacki_2006
    )

    # Elasticity
    el_upper_crust              = SetConstantElasticity(; G=10e9, ν=0.45)
    el_lower_crust              = SetConstantElasticity(; G=10e9, ν=0.45)
    el_lithospheric_mantle      = SetConstantElasticity(; G=10e9, ν=0.45)
    el_seed                     = SetConstantElasticity(; G=5e9, ν=0.45)
    # el_sublithospheric_mantle   = SetConstantElasticity(; G=67e9, ν=0.5)
    β_upper_crust               = inv(get_Kb(el_upper_crust))
    β_lower_crust               = inv(get_Kb(el_lower_crust))
    β_lithospheric_mantle       = inv(get_Kb(el_lithospheric_mantle))
    # β_sublithospheric_mantle    = inv(get_Kb(el_sublithospheric_mantle))

    # Physical properties using GeoParams ----------------
    η_reg         = 1e17 # regularized viscosity
    cohesion      = 30e6 ## 100e6
    friction      = 30
    friction_seed = 1
    softening_C   = LinearSoftening((cohesion/2, cohesion), (0e0, 2e0))
    pl            = DruckerPrager_regularised(; C = cohesion, softening_C = softening_C, ϕ=friction, η_vp=η_reg, Ψ=0.0) # regularized plasticity
    pl_seed       = DruckerPrager_regularised(; C = 1e6, ϕ=friction_seed, η_vp=η_reg, Ψ=0.0, ) # regularized plasticity
    
    # crust
    # K_crust = TP_Conductivity(;
    #     a = 0.6,
    #     b = 807e00,
    #     c = 0.77,
    #     d = 0.00004,
    # )
    # K_mantle = TP_Conductivity(;
    #     a = 0.73,
    #     b = 1293e00,
    #     c = 0.77,
    #     d = 0.00004,
    # )

    # Define rheolgy struct
    rheology = (
        # Name              = "Brittle",
        SetMaterialParams(;
            Phase             = 1,
            Density           = PT_Density(; ρ0=2.7e3, β=β_upper_crust, T0=0.0, α = 2.4e-5),
            HeatCapacity      = ConstantHeatCapacity(; Cp=1e3),
            Conductivity      = ConstantConductivity(; k=2.1),
            # Conductivity      = K_crust,
            CompositeRheology = CompositeRheology((disl_upper_crust,el_upper_crust,pl)),
            Elasticity = el_upper_crust,
            Plasticity = pl,
            RadioactiveHeat   = ConstantRadioactiveHeat(1.3),
            ShearHeat         = ConstantShearheating(1.0NoUnits),
            Gravity           = ConstantGravity(; g=9.81),
        ),  
        # Name              = "Weak Viscosity Brittle",
        SetMaterialParams(;
            Phase             = 2,
            Density           = PT_Density(; ρ0=3.3e3, β=β_upper_crust, T0=0.0, α = 3.0e-5),
            HeatCapacity      = ConstantHeatCapacity(; Cp=1e3),
            Conductivity      = ConstantConductivity(; k=2.1),
            # Conductivity      = K_crust,
            Elasticity = el_upper_crust,
            Plasticity = pl,
            CompositeRheology = CompositeRheology((disl_lithospheric_mantle,el_upper_crust,pl)),
            RadioactiveHeat   = ConstantRadioactiveHeat(1.3),
            ShearHeat         = ConstantShearheating(1.0NoUnits),
            Gravity           = ConstantGravity(; g=9.81),
        ),
        
        # Name              = "Weak Brittle Inclusion",
        SetMaterialParams(;
            Phase             = 3,
            Density           = PT_Density(; ρ0=3.3e3, β=β_upper_crust, T0=0.0, α = 2.4e-5),
            HeatCapacity      = ConstantHeatCapacity(; Cp=1e3),
            Conductivity      = ConstantConductivity(; k=2.1),
            Elasticity        = el_seed,
            Plasticity        = pl_seed,
            # Conductivity      = K_crust,
            CompositeRheology = CompositeRheology((disl_upper_crust,el_seed,pl_seed)),
            RadioactiveHeat   = ConstantRadioactiveHeat(1.3),
            ShearHeat         = ConstantShearheating(1.0NoUnits),
            Gravity           = ConstantGravity(; g=9.81),
        ),

        # Name              = "StickyAir",
        SetMaterialParams(;
            Phase             = 4,
            Density           = ConstantDensity(; ρ=1e0),
            HeatCapacity      = ConstantHeatCapacity(; Cp=7.5e2),
            RadioactiveHeat   = ConstantRadioactiveHeat(0.0),
            Conductivity      = ConstantConductivity(; k=0.5),
            CompositeRheology = CompositeRheology((LinearViscous(; η=1e18),)),
        ),
    )
end

function init_phases!(phases, phase_grid, particles, xvi)
    ni = size(phases)
    @parallel (@idx ni) _init_phases!(phases, phase_grid, particles.coords, particles.index, xvi)
end

@parallel_indices (I...) function _init_phases!(phases, phase_grid, pcoords::NTuple{N, T}, index, xvi) where {N,T}

    ni = size(phases)

    for ip in cellaxes(phases)
        # quick escape
        @index(index[ip, I...]) == 0 && continue

        pᵢ = ntuple(Val(N)) do i
            @index pcoords[i][ip, I...]
        end

        d = Inf # distance to the nearest particle
        particle_phase = -1
        for offi in 0:1, offj in 0:1
            ii = I[1] + offi
            jj = I[2] + offj

            !(ii ≤ ni[1]) && continue
            !(jj ≤ ni[2]) && continue

            xvᵢ = (
                xvi[1][ii],
                xvi[2][jj],
            )
            d_ijk = √(sum((pᵢ[i] - xvᵢ[i])^2 for i in 1:N))
            if d_ijk < d
                d = d_ijk
                particle_phase = phase_grid[ii, jj]
            end
        end
        @index phases[ip, I...] = Float64(particle_phase)
    end

    return nothing
end
