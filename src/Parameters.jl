# Parameter.jl
# Retrieves and defines parameters for the model, including utility function, 
# production function, and grid specifications.

# -------------------------------------------------------------------
# Utility function and derivatives
# -------------------------------------------------------------------
@inline function ufnc(c, Param)
    σ = Param.σ
    return (σ == 1.0) ? log(c) : (c^(1.0 - σ) - 1.0) / (1.0 - σ)
end

@inline function d1ufnc(c, Param)
    return c^(-Param.σ)
end

@inline function d2ufnc(c, Param)
    σ = Param.σ
    return -σ * c^(-σ - 1.0)
end

# -------------------------------------------------------------------
# Capital adjustment cost function and derivatives
# -------------------------------------------------------------------
@inline function kadjfnc(k′, k, Param)
    ψ_k = Param.ψ_k
    δ   = Param.δ
    return ψ_k / 2.0 * ((k′ / k) - (1.0 - δ))^2 * k
end

@inline function dk′_kadjfnc(k′, k, Param)
    ψ_k = Param.ψ_k
    δ   = Param.δ
    return ψ_k * ((k′ / k) - (1.0 - δ))
end

@inline function dk_kadjfnc(k′, k, Param)
    ψ_k = Param.ψ_k
    δ   = Param.δ
    return ψ_k / 2.0 * ((k′ / k) - (1.0 - δ)) * (-(k′ / k) - (1.0 - δ))
end

# -------------------------------------------------------------------
# Create parameter structure
# -------------------------------------------------------------------
struct Struct_Param{T,GK,DK,GD,DD,GX,GW,GZ}
    # Preference
    β::T                      # Discount factor
    σ::T                      # Coefficient of relative risk aversion
    ξ::T                      # Weight on consumption political composite

    # Production
    α::T                      # Capital share in production
    δ::T                      # Depreciation rate
    ψ_k::T                    # Capital adjustment cost parameter
    χ::T                      # Death probability
    ϕ::T                      # Earnings-based borrowing constraint parameter

    # Capital grid
    ObjGrid_K::GK             # Capital grid object
    ObjDGrid_K::DK            # Dense capital grid object

    # Debt grid
    ObjGrid_D::GD             # Debt grid object
    ObjDGrid_D::DD            # Dense debt grid object

    # Political capital grid
    δ_x::T                    # Probability of political capital depreciation
    c_x::T                    # Cost of political engagement
    ObjGrid_X::GX             # Political capital grid object

    # Idiosyncratic shocks
    ObjGrid_ω::GW             # Taste shifter grid object
    ObjGrid_Z::GZ             # Productivity shock grid object
end

function Get_Params(;
    # Preference
    β     = 0.98,               # Discount factor
    σ     = 1.0,                # Coefficient of relative risk aversion
    ξ     = 0.5,                # Weight on consumption political composite

    # Production
    α      = 0.25,              # Capital share in production
    δ      = 0.07,              # Depreciation rate
    ψ_k    = 5.0,               # Capital adjustment cost parameter
    χ      = 0.08,              # Death probability
    ϕ      = 0.5,               # Earnings-based borrowing constraint parameter

    # Capital grid
    Nk    = 20,                 # Number of capital grid points
    Nkg   = 100,                # Number of capital grid points for policy function interpolation
    k_min = 0.01,               # Minimum capital level
    k_max = 10.0,               # Maximum capital level
    ObjGrid_K  = make_grid(:log,    lb=k_min, ub=k_max, n=Nk,  as_object=true, name="Capital Grid"),
    ObjDGrid_K = make_grid(:linear, lb=k_min, ub=k_max, n=Nkg, as_object=true, name="Dense Capital Grid"),

    # Debt grid
    Nd    = 10,                 # Number of debt grid points
    Ndg   = 100,                # Number of debt grid points for policy function interpolation
    d_min =-10.0,               # Minimum debt level (borrowing limit)
    d_mid = 0.0,                # Intermediate debt level (zero debt)
    d_max = 10.0,               # Maximum debt level
    ObjGrid_D  = make_grid(:doublelog_eq, lb=d_min, mid=d_mid, ub=d_max, n=Nd,  as_object=true, name="Debt Grid"),
    ObjDGrid_D = make_grid(:doublelog_eq, lb=d_min, mid=d_mid, ub=d_max, n=Ndg, as_object=true, name="Dense Debt Grid"),

    # Political capital grid
    δ_x    = 0.5,               # Probability of political capital depreciation
    c_x    = 0.5,               # Cost of political engagement
    Nx     = 2,                 # Number of points in the political capital grid (0 and 1)
    x_min  = 0.0,               # Minimum political capital level
    x_max  = 1.0,               # Maximum political capital level
    Grid_X = [0, 1],            # Political capital grid (0 and 1, integers)
    Prob_X = [δ_x, 1.0 - δ_x],  # Transition probabilities for political capital (1 to 0 and 1 to 1)
    ObjGrid_X = (; Values = Grid_X, Prob = Prob_X, N = Nx, Name = "Political Capital Grid"),

    # Idiosyncratic shocks
    # Taste shifter
    Nω  = 5,                    # Number of points in the idiosyncratic taste shifter grid
    μ_ω = 0.0,                  # Mean of the log idiosyncratic taste shifter
    ρ_ω = 0.8,                  # Autoregressive parameter for ω (idiosyncratic taste shifter)
    σ_ω = 0.05,                 # Std dev of innovation to ω (idiosyncratic taste shifter)
    ObjGrid_ω = make_stochproc(:tauchen, μ=μ_ω, σ=σ_ω, ρ=ρ_ω, N=Nω, transf=:exp, name="Idiosyncratic Taste Shifter Grid", width=3.0),

    # Productivity shock
    Nz  = 5,                    # Number of points in the idiosyncratic productivity grid
    μ_z = 0.0,                  # Mean of the log idiosyncratic productivity
    ρ_z = 0.8,                  # Autoregressive parameter for z (idiosyncratic productivity)
    σ_z = 0.03,                 # Std dev of innovation to z (idiosyncratic productivity)
    ObjGrid_Z = make_stochproc(:tauchen, μ=μ_z, σ=σ_z, ρ=ρ_z, N=Nz, transf=:exp, name="Idiosyncratic Productivity Grid", width=3.0),
    )

    return Struct_Param(
        β, σ, ξ,
        α, δ, ψ_k, χ, ϕ,
        ObjGrid_K, ObjDGrid_K,
        ObjGrid_D, ObjDGrid_D,
        δ_x, c_x, ObjGrid_X,
        ObjGrid_ω, ObjGrid_Z
    )
end