import Pkg
Pkg.activate(dirname(@__DIR__))

using Revise, UnPack
using SparseArrays, Roots, LinearAlgebra
using Plots
using Statistics
using Lib_Julia_Econ

# Helpers
# include(joinpath(@__DIR__, "Parameters.jl"))
include("Parameters.jl")
Param = Get_Params()


# ## Create parameter structure
# function Get_Params(;
#     # Preference 
#     β     = 0.98,               # Discount factor
#     σ     = 1.0,                # Coefficient of relative risk aversion
#     ξ     = 0.5,                # Weight on consumption political composite
#     # Utility function (CRRA with σ=1 as log)
#     ufnc   = c -> (σ == 1.0 ? log(c) : (c.^(1.0 - σ) .- 1.0) ./ (1.0 - σ)),  
#     d1ufnc = c -> c.^(-σ),
#     d2ufnc = c -> -σ * c.^(-σ - 1),
    
#     # Production
#     α      = 0.25,              # Capital share in production
#     δ      = 0.07,              # Depreciation rate
#     ψ_k    = 5.0,               # Capital adjustment cost parameter
#     χ      = 0.08,              # Death probability
#     ϕ      = 0.5,               # Earnings-based borrowing constraint parameter
#     # Capital adjustment cost function
#     kadjfnc     = (k′, k) -> ψ_k/2.0 * ((k′/k) - (1.0 - δ))^2 * k,  
#     dk′_kadjfnc = (k′, k) -> ψ_k * ((k′/k) - (1.0 - δ)),
#     dk_kadjfnc  = (k′, k) -> ψ_k/2.0 * ((k′/k) - (1.0 - δ))*(-(k′/k) - (1.0 - δ)),
    
#     # Capital grid
#     Nk    = 20,                 # Number of capital grid points
#     Nkg   = 100,                # Number of capital grid points for policy function interpolation
#     k_min = 0.01,               # Minimum capital level
#     k_max = 10.0 ,              # Maximum capital level
#     ObjGrid_K  = make_grid(:log,    lb=k_min, ub=k_max, n=Nk,  as_object=true, name="Capital Grid"),
#     ObjDGrid_K = make_grid(:linear, lb=k_min, ub=k_max, n=Nkg, as_object=true, name="Dense Capital Grid"),
    
#     # Debt grid
#     Nd    = 10,                 # Number of debt grid points
#     Ndg   = 100,                # Number of debt grid points for policy function interpolation
#     d_min =-10.0,               # Minimum debt level (borrowing limit)
#     d_mid = 0.0,                # Intermediate debt level (zero debt)
#     d_max = 10.0,               # Maximum debt level
#     ObjGrid_D  = make_grid(:doublelog_eq, lb=d_min, mid=d_mid, ub=d_max, n=Nd,  as_object=true, name="Debt Grid"),
#     ObjDGrid_D = make_grid(:doublelog_eq, lb=d_min, mid=d_mid, ub=d_max, n=Ndg, as_object=true, name="Dense Debt Grid"),
    
#     # Political capital grid
#     δ_x    = 0.5,               # Probability of political capital depreciation
#     c_x    = 0.5,               # Cost of political engagement
#     Nx     = 2,                 # Number of points in the political capital grid (0 and 1)
#     x_min  = 0.0,               # Minimum political capital level
#     x_max  = 1.0,               # Maximum political capital level
#     Grid_X = [0, 1],            # Political capital grid (0 and 1, integers)
#     Prob_X = [δ_x, 1.0 - δ_x],  # Transition probabilities for political capital (1 to 0 and 1 to 1)
#     ObjGrid_X = (; Values = Grid_X, Prob = Prob_X, N = Nx, Name = "Political Capital Grid"),
    
#     # Idiosyncratic shocks
#     # Taste shifter
#     Nω  = 5,                    # Number of points in the idiosyncratic taste shifter grid
#     μ_ω = 0.0,                  # Mean of the log idiosyncratic taste shifter
#     ρ_ω = 0.8,                  # Autoregressive parameter for ω (idiosyncratic taste shifter)
#     σ_ω = 0.05,                 # Std dev of innovation to ω (idiosyncratic taste shifter)
#     ObjGrid_ω = make_stochproc(:tauchen, μ=μ_ω, σ=σ_ω, ρ=ρ_ω, N=Nω, transf=:exp, name="Idiosyncratic Taste Shifter Grid", width=3.0),
#     # Productivity shock
#     Nz  = 5,                    # Number of points in the idiosyncratic productivity grid
#     μ_z = 0.0,                  # Mean of the log idiosyncratic productivity
#     ρ_z = 0.8,                  # Autoregressive parameter for z (idiosyncratic productivity)
#     σ_z = 0.03,                 # Std dev of innovation to z (idiosyncratic productivity)
#     ObjGrid_Z = make_stochproc(:tauchen, μ=μ_z, σ=σ_z, ρ=ρ_z, N=Nz, transf=:exp, name="Idiosyncratic Productivity Grid", width=3.0), 
#     )
    
#     return (β = β, σ = σ, ξ = ξ, ufnc = ufnc, d1ufnc = d1ufnc, d2ufnc = d2ufnc,
#     α = α, δ = δ, ψ_k = ψ_k, χ = χ, ϕ = ϕ, kadjfnc = kadjfnc, dk_kadjfnc = dk_kadjfnc, dk′_kadjfnc = dk′_kadjfnc,
#     ObjGrid_K = ObjGrid_K, ObjDGrid_K = ObjDGrid_K, ObjGrid_D = ObjGrid_D, ObjDGrid_D = ObjDGrid_D,
#     ObjGrid_X = ObjGrid_X, δ_x = δ_x, c_x = c_x,
#     ObjGrid_ω = ObjGrid_ω, ObjGrid_Z = ObjGrid_Z) 
    
# end


## Getting default parameters
# if !(@isdefined Param)
#     const Param = Get_Params()
# end
Param.ObjGrid_X


## Helper functions

# Unit output price
punit = (pp, x, ω) -> x == 0.0 ? 1.0 : ω * pp

# Flow profits
function profit(pp, x, ω, z, k, e; Param)
    @unpack (α, c_x) = Param
    prof = punit(pp, x, ω) * z * k^α - c_x * e
    return prof
end

# Net worth
function networth(pp, x, ω, z, k, e, d; Param)
    @unpack (α, δ, c_x) = Param
    nw = profit(pp, x, ω, z, k, e; Param) + (1.0 - δ) * k - d
    return nw
end



## Solve problem of unconstrained firm

# Determine efficient scale of operation
# We do policy function iteration. Define function G(k′)
function Gkstar(k′, ik, ix, iz, iω, K_Unconstrained, E_Unconstrained, Price_tf, Param) 
    @unpack (α, δ, χ, ObjGrid_ω, ObjGrid_Z, ObjGrid_X, ObjGrid_K) = Param
    Grid_ω = ObjGrid_ω.Values
    Grid_Z = ObjGrid_Z.Values
    Grid_X = ObjGrid_X.Values
    Grid_K = ObjGrid_K.Values
    
    # Compute expected marginal return to capital
    Expectation = 0.0
    for (iω′, ω′) in enumerate(Grid_ω)
        for (iz′, z′) in enumerate(Grid_Z)
            prob_zω = ObjGrid_ω.Prob[iω, iω′] * ObjGrid_Z.Prob[iz, iz′]
            if Grid_X[ix] == 0
                # Future x′ depends on engagement choice today
                x′_val  = E_Unconstrained[ik, ix, iz, iω]   # must be 0.0 or 1.0
                ix′     = (x′_val == 0) ? 1 : 2
                prob_x  = 1.0
                x′      = Grid_X[ix′]
                kstar′′ = linterp1(Grid_K, K_Unconstrained[:, ix′, iz′, iω′], k′)
                Expectation += prob_x * prob_zω * (punit(Price_tf.pp, x′, ω′) * z′ * α * (k′)^(α - 1.0) + (1.0 - δ) - (1.0 - χ) * dk_kadjfnc(kstar′′, k′, Param))
            else
                # No engagement choice today, so future x′ depends on depreciation probability
                for ix′ in 1:ObjGrid_X.N
                    x′      = Grid_X[ix′]
                    prob_x  = ObjGrid_X.Prob[ix′]
                    kstar′′ = linterp1(Grid_K, K_Unconstrained[:, ix′, iz′, iω′], k′)
                    Expectation += prob_x * prob_zω * (punit(Price_tf.pp, x′, ω′) * z′ * α * (k′)^(α - 1.0) + (1.0 - δ) - (1.0 - χ) * dk_kadjfnc(kstar′′, k′, Param))
                end
            end
        end
    end
    
    # Objective for policy function iteration: G(k′) = 0
    kval = Grid_K[ik]
    G = 1.0 + dk′_kadjfnc(k′, kval, Param) - Price_tf.SDF * Expectation
    return G
end

# Policy function iteration
function solve_kstar(E_Unconstrained, Price_t, Param)
    @unpack (ObjGrid_K, ObjGrid_X, ObjGrid_ω, ObjGrid_Z) = Param
    @unpack (δ) = Param
    # Initializing
    K_Unconstrained = zeros(ObjGrid_K.N, ObjGrid_X.N, ObjGrid_Z.N, ObjGrid_ω.N)
    K_Unconstrained[:, :, :, :] .= (1.0 - δ) * ObjGrid_K.Values;
    K_old = copy(K_Unconstrained)
    K_new = copy(K_Unconstrained)
    for iter in 1:1000
        # println("Iteration: ", iter)    
        for ik in 1:ObjGrid_K.N
            for ix in 1:ObjGrid_X.N
                for iz in 1:ObjGrid_Z.N
                    for iω in 1:ObjGrid_ω.N
                        k′_star = find_zero(k′ -> Gkstar(k′, ik, ix, iz, iω, K_old, E_Unconstrained, Price_t, Param), (0.01, 1000.0), Bisection())
                        K_new[ik, ix, iz, iω] = k′_star
                    end
                end
            end
        end
        
        # Check convergence of policy function iteration
        max_diff = maximum(abs.(K_new - K_old))
        # println("Max difference in policy function: ", max_diff)
        if max_diff < 1e-4
            println("Convergence (k*)  achieved after ", iter, " iterations, with max difference: ", max_diff)
            break
        end
        K_old .= K_new
    end
    K_Unconstrained .= K_new
    
    return K_Unconstrained
end

# K_Unconstrained = solve_kstar(E_Unconstrained, Price_t, Param)


## Minimum savings policy
function update_msp(Dmsp, K_Unconstrained, E_Unconstrained, Price_tf, Param)
    @unpack (ObjGrid_K, ObjGrid_X, ObjGrid_ω, ObjGrid_Z) = Param
    @unpack (α, δ, c_x) = Param
    Grid_K = ObjGrid_K.Values
    Grid_X = ObjGrid_X.Values
    Grid_Z = ObjGrid_Z.Values
    Grid_ω = ObjGrid_ω.Values
    Nk, Nx, Nz, Nω = ObjGrid_K.N, ObjGrid_X.N, ObjGrid_Z.N, ObjGrid_ω.N

    
    TDmsp = similar(Dmsp)
    for (ik, k) in enumerate(Grid_K)
        for (ix, x) in enumerate(Grid_X)
            for (iz, z) in enumerate(Grid_Z)
                for (iω, ω) in enumerate(Grid_ω)
                    # Capital at beginning of next period
                    k′   = K_Unconstrained[ik, ix, iz, iω]
                    Dmsp_temp = zeros(ObjGrid_Z.N, ObjGrid_ω.N)
                    for (iz′, z′) in enumerate(Grid_Z)
                        for (iω′, ω′) in enumerate(Grid_ω)
                            # Compute flow profits for every realization of the future state
                            # Note that we are implementing the efficient capital level
                            if Grid_X[ix] == 0
                                # Future x′ depends on engagement choice today
                                x′_val  = E_Unconstrained[ik, ix, iz, iω]   # must be 0 or 1
                                ix′     = (x′_val == 0) ? 1 : 2
                                prob_x  = 1.0
                                x′      = Grid_X[ix′]
                                kstar′′ = linterp1(Grid_K, K_Unconstrained[:, ix′, iz′, iω′], k′)
                                eval′   = 1.0 # Worst case scenario
                                dstar′′ = linterp1(Grid_K, Dmsp[:, ix′, iz′, iω′], k′)
                                πval′   = profit(Price_tf.pp, x′, ω′, z′, k′, eval′; Param)
                                dtilde  = πval′ + (1.0 - δ) * k′ - kstar′′ - kadjfnc(kstar′′, k′, Param) + (1.0/Price_tf.R) * dstar′′
                            else
                                # No engagement choice today, so future x′ depends on depreciation probability
                                # If the state is unreacheable, set dtild(x′) to large value
                                dtildex  = zeros(ObjGrid_X.N)
                                dtildex .= Inf
                                for ix′ in 1:ObjGrid_X.N
                                    x′      = Grid_X[ix′]
                                    prob_x  = ObjGrid_X.Prob[ix′]
                                    prob_x  < 1e-6 && continue  # Skip states with negligible probability to save computation
                                    kstar′′ = linterp1(Grid_K, K_Unconstrained[:, ix′, iz′, iω′], k′)
                                    eval′   = (x′ == 0) ? 1.0 : 0.0    # Worst case scenario
                                    dstar′′ = linterp1(Grid_K, Dmsp[:, ix′, iz′, iω′], k′)
                                    πval′   = profit(Price_tf.pp, x′, ω′, z′, k′, eval′; Param)                                    
                                    dtildex[ix′] = πval′ + (1.0 - δ) * k′ - kstar′′ - kadjfnc(kstar′′, k′, Param) + (1.0/Price_tf.R) * dstar′′
                                end
                                dtilde = minimum(dtildex)
                            end                          
                            # If the state is unreacheable, set Wtild to large value
                            prob_zω = ObjGrid_ω.Prob[iω, iω′] * ObjGrid_Z.Prob[iz, iz′]
                            dtilde  = (prob_zω < 1e-6) ? Inf : dtilde
                            Dmsp_temp[iz′, iω′] = dtilde
                        end
                    end
                    TDmsp[ik, ix, iz, iω] = minimum(Dmsp_temp)
                end
            end
        end
    end
    return TDmsp
end

function solve_msp(K_Unconstrained, E_Unconstrained, Price_tf, Param, Dmsp)
    @unpack (ObjGrid_K, ObjGrid_X, ObjGrid_ω, ObjGrid_Z) = Param
    # Iterate on minimum savings policy until convergence
    for iter in 1:1000
        # println("Iteration: ", iter)    
        # Update minimum savings policy given current policy functions for unconstrained firm
        TDmsp = update_msp(Dmsp, K_Unconstrained, E_Unconstrained, Price_tf, Param)    
        # Check convergence of minimum savings policy
        max_diff = maximum(abs.(Dmsp - TDmsp))
        # println("Max difference in minimum savings policy: ", max_diff)
        if max_diff < 1e-4
            println("Convergence (MSP) achieved after ", iter, " iterations, with max difference: ", max_diff)
            break
        end
        Dmsp .= TDmsp
    end
    return Dmsp
end


# Dmsp = solve_msp(K_Unconstrained, E_Unconstrained, Price_t, Param)


## Update value function given optimal policy for unconstrained firm
# Note that the value for an unconstrained firm is linear/separable in current debt
# We can ignore current debt for these firms: 
# We only need it for updating the engagement policy, so 𝓥_n and 𝓥_p are being affected linearly by the same d
# So we evalue v* at d=0 for simplicity
function value_uncons(Dmsp, V_Unconstrained, K_Unconstrained, E_Unconstrained, Price_t, Price_tf, Param)
    @unpack (ObjGrid_D, ObjGrid_K, ObjGrid_X, ObjGrid_ω, ObjGrid_Z) = Param
    @unpack (α, δ, χ, c_x) = Param
    Grid_D = ObjGrid_D.Values
    Grid_K = ObjGrid_K.Values
    Grid_X = ObjGrid_X.Values
    Grid_Z = ObjGrid_Z.Values
    Grid_ω = ObjGrid_ω.Values
    # Allocating
    TV_Unconstrained = zeros(ObjGrid_K.N, ObjGrid_X.N, ObjGrid_Z.N, ObjGrid_ω.N)
    TE_Unconstrained = copy(TV_Unconstrained)
    for (ik, k) in enumerate(Grid_K)
        for (iz, z) in enumerate(Grid_Z)
            for (iω, ω) in enumerate(Grid_ω)
                for (ix, x) in enumerate(Grid_X)
                    # Engagement choice
                    eval = E_Unconstrained[ik, ix, iz, iω]
                    # Capital choice
                    k′   = K_Unconstrained[ik, ix, iz, iω]
                    # Minimum savings policy
                    d′   = Dmsp[ik, ix, iz, iω]
                    # Net worth, ex-engagement cost (will be considered below when comparing 𝓥_n and 𝓥_p)
                    nw = punit(Price_t.pp, x, ω) * z * k^α + (1.0 - δ) * k
                    # Current value
                    vnow_base = nw - k′ - kadjfnc(k′, k, Param) + (1.0/Price_t.R) * d′
                    # Continuation value
                    if Grid_X[ix] == 0
                        # Counterfactual continuations from same current state (x=0)
                        vcont_n, vcont_p = value_uncons_vcont_x0(ik, iz, iω, k′, d′, Price_t, Price_tf, V_Unconstrained, Param)
                        # Value associated with each engamagement choice
                        𝓥_n = vnow_base + vcont_n 
                        𝓥_p = vnow_base + vcont_p - c_x
                        # Engagement choice 
                        TE_Unconstrained[ik, 1, iz, iω] = (𝓥_p > 𝓥_n) ? 1.0 : 0.0
                        # Use current eval (from previous iteration) to compute the actual value this iteration
                        vnow  = vnow_base - c_x * eval
                        vcont = (eval ≈ 1.0) ? vcont_p : vcont_n

                    else
                        # Your existing continuation code for x=1 can remain as-is
                        vcont = value_uncons_vcont_x1(ik, iz, iω, k′, d′, Price_t, Price_tf, V_Unconstrained, Param)
                        vnow  = vnow_base
                    end
                    # Update value function for unconstrained firm                    
                    TV_Unconstrained[ik, ix, iz, iω] = vnow + vcont
                end
            end
        end
    end
    
    return TV_Unconstrained, TE_Unconstrained
end


# Continuation value given current state (ik,iz,iω,ix) for unconstrained firm with x=0
function value_uncons_vcont_x0(ik, iz, iω, k′, d′, Price_t, Price_tf, V_Unconstrained, Param)
    @unpack (ObjGrid_K, ObjGrid_X, ObjGrid_ω, ObjGrid_Z) = Param
    @unpack (α, δ, χ) = Param
    Grid_K = ObjGrid_K.Values
    Grid_X = ObjGrid_X.Values
    Grid_Z = ObjGrid_Z.Values
    Grid_ω = ObjGrid_ω.Values
    # Initializing
    vcont_n = 0.0
    vcont_p = 0.0
    for (iz′, z′) in enumerate(Grid_Z)
        for (iω′, ω′) in enumerate(Grid_ω)
            # Probability weight wrt to idiosyncratic shocks
            prob_zω = ObjGrid_ω.Prob[iω, iω′] * ObjGrid_Z.Prob[iz, iz′]

            # No engagement, so e=0 and x′=0
            ix′      = 1
            x′       = Grid_X[ix′]
            Enw_n    = punit(Price_tf.pp, x′, ω′) * z′ * k′^α + (1.0 - δ) * k′ - d′
            v_n      = linterp1(Grid_K, V_Unconstrained[:, ix′, iz′, iω′], k′)
            vcont_n += Price_t.SDF * prob_zω * ((1.0 - χ) * v_n + χ * Enw_n)

            # Engagement, so e=1 and x′=1
            ix′      = 2
            x′       = Grid_X[ix′]
            Enw_p    = punit(Price_tf.pp, x′, ω′) * z′ * k′^α + (1.0 - δ) * k′ - d′
            v_p      = linterp1(Grid_K, V_Unconstrained[:, ix′, iz′, iω′], k′)
            term_p   = (1.0 - χ) * v_p + χ * Enw_p
            vcont_p += Price_t.SDF * prob_zω * ((1.0 - χ) * v_p + χ * Enw_p)            
        end
    end

    return vcont_n, vcont_p
end


# Continuation value given current state (ik,iz,iω,ix) for unconstrained firm with x=1
function value_uncons_vcont_x1(ik, iz, iω, k′, d′, Price_t, Price_tf, V_Unconstrained, Param)
    @unpack (ObjGrid_K, ObjGrid_X, ObjGrid_ω, ObjGrid_Z) = Param
    @unpack (α, δ, χ) = Param
    Grid_K = ObjGrid_K.Values
    Grid_X = ObjGrid_X.Values
    Grid_Z = ObjGrid_Z.Values
    Grid_ω = ObjGrid_ω.Values
    # Initializing
    vcont = 0.0
    for (iz′, z′) in enumerate(Grid_Z)
        for (iω′, ω′) in enumerate(Grid_ω)
            # Probability weight wrt to idiosyncratic shocks
            prob_zω = ObjGrid_ω.Prob[iω, iω′] * ObjGrid_Z.Prob[iz, iz′]

            # Expectation over x' given current x=1
            term_x = 0.0
            for ix′ in 1:ObjGrid_X.N
                x′      = Grid_X[ix′]
                prob_x  = ObjGrid_X.Prob[ix′]
                Enw     = punit(Price_tf.pp, x′, ω′) * z′ * k′^α + (1.0 - δ) * k′ - d′
                v′      = linterp1(Grid_K, V_Unconstrained[:, ix′, iz′, iω′], k′)
                term_x += prob_x * ((1.0 - χ) * v′ + χ * Enw)
            end

            prob_zω = ObjGrid_ω.Prob[iω, iω′] * ObjGrid_Z.Prob[iz, iz′]
            vcont  += Price_t.SDF * prob_zω * term_x          
        end
    end

    return vcont
end

# V_Unconstrained, TE_Unconstrained = value_uncons(Dmsp, V_Unconstrained, K_Unconstrained, E_Unconstrained, Price_t, Price_t, Param)


## Get unconstrained threshold
function get_unconstrained_threshold(K_Unconstrained, Dmsp, Price_t, Param)
    @unpack (ObjGrid_K, ObjGrid_X, ObjGrid_ω, ObjGrid_Z) = Param
    @unpack (α, δ, χ, c_x) = Param
    
    nthresh_uncons = zeros(ObjGrid_K.N, ObjGrid_X.N, ObjGrid_Z.N, ObjGrid_ω.N)
    for (ik, k) in enumerate(ObjGrid_K.Values)
        for (ix, x) in enumerate(ObjGrid_X.Values)
            for (iz, z) in enumerate(ObjGrid_Z.Values)
                for (iω, ω) in enumerate(ObjGrid_ω.Values)
                    nthresh_uncons[ik, ix, iz, iω] = K_Unconstrained[ik, ix, iz, iω] + kadjfnc(K_Unconstrained[ik, ix, iz, iω], k, Param) - (1.0/Price_t.R) * Dmsp[ik, ix, iz, iω]
                end
            end
        end
    end
    
    return nthresh_uncons
end


## Iterate to solve the unconstrained firm's problem
function solve_unconstrained_firm(Param; Price_t = (R=1.04, SDF=1.0/1.04, pp=1.25), max_iter = 10, max_viter = 500)
    # Initialize
    V_Unconstrained  = zeros(Param.ObjGrid_K.N, Param.ObjGrid_X.N, Param.ObjGrid_Z.N, Param.ObjGrid_ω.N)
    E_Unconstrained  = copy(V_Unconstrained)
    TE_Unconstrained = copy(E_Unconstrained)
    K_Unconstrained  = similar(V_Unconstrained)
    # Initialize engagement policy to zero for all states
    Dmsp = zeros(Param.ObjGrid_K.N, Param.ObjGrid_X.N, Param.ObjGrid_Z.N, Param.ObjGrid_ω.N)

    for iter in 1:max_iter
        # Solve the efficient capital policy given current engagement policy
        K_Unconstrained = solve_kstar(E_Unconstrained, Price_t, Param)
        # Get the MSP given the efficient capital and engagement policies
        TDmsp = solve_msp(K_Unconstrained, E_Unconstrained, Price_t, Param, Dmsp)
        Dmsp .= TDmsp
        # Update value function given optimal policies for unconstrained firm
        for viter in 1:max_viter
            TV_Unconstrained, TE_Unconstrained = value_uncons(Dmsp, V_Unconstrained, K_Unconstrained, E_Unconstrained, Price_t, Price_t, Param)
            vdiff = maximum(abs.(TV_Unconstrained .- V_Unconstrained))
            V_Unconstrained .= 0.9 .* V_Unconstrained .+ 0.1 .* TV_Unconstrained
            if vdiff < 1e-8
                break
            end
        end
        # Check convergence of engagement policy
        max_diff = maximum(abs.(E_Unconstrained - TE_Unconstrained))
        println("Iteration: ", iter, " - Max difference in engagement policy: ", max_diff, " and Average difference: ", mean(abs.(E_Unconstrained - TE_Unconstrained)))
        println("Average engagement: ", mean(E_Unconstrained))
        if max_diff < 1e-4
            println("Convergence achieved after ", iter, " iterations.")
            break
        end
        # Update engagement policy given updated value function
        E_Unconstrained .= TE_Unconstrained
    end

    # Retrieve the unconstrained threshold
    nthresh_uncons = get_unconstrained_threshold(K_Unconstrained, Dmsp, Price_t, Param)

    return (
        Price_t          = Price_t,
        V_Unconstrained  = V_Unconstrained,
        E_Unconstrained  = E_Unconstrained,
        TE_Unconstrained = TE_Unconstrained,
        K_Unconstrained  = K_Unconstrained,
        Dmsp             = Dmsp,
        nthresh_uncons   = nthresh_uncons
    )
end


## Given some prices and solve unconstrained problem in hard scope
Sol_Unconstrained = solve_unconstrained_firm(Param)
Price_t = Sol_Unconstrained.Price_t
V_Unconstrained  = Sol_Unconstrained.V_Unconstrained
E_Unconstrained  = Sol_Unconstrained.E_Unconstrained
TE_Unconstrained = Sol_Unconstrained.TE_Unconstrained
K_Unconstrained  = Sol_Unconstrained.K_Unconstrained
Dmsp = Sol_Unconstrained.Dmsp


## Analysis
# Plotting efficient capital policy
plot(Param.ObjGrid_K.Values, K_Unconstrained[:, 1,1,1], label="x=0", xlabel="Current Capital (k)", ylabel="Optimal Next Period Capital (k′)", title="Efficient Capital Policy (ik=5, iz=3, iω=3)")
# Add 45 degree line
plot!(Param.ObjGrid_K.Values, Param.ObjGrid_K.Values, label="45 Degree Line", linestyle=:dash)


## Plotting unconstrained threshold
plot(Param.ObjGrid_K.Values, Sol_Unconstrained.nthresh_uncons[:, 1,1,1], label="x=0", 
     xlabel="Current Capital (k)", ylabel="Unconstrained Threshold (n*)", title="Unconstrained Threshold (ik=5, iz=3, iω=3)",
     lw=2)


## Solve firm decision rules
function value_firm(Sol_Unconstrained, Price_t; Param)
    @unpack (ObjGrid_D, ObjGrid_K, ObjGrid_X, ObjGrid_ω, ObjGrid_Z) = Param
    @unpack (α, δ, χ) = Param
    Grid_D = ObjGrid_D.Values
    Grid_K = ObjGrid_K.Values
    Grid_X = ObjGrid_X.Values
    Grid_Z = ObjGrid_Z.Values
    Grid_ω = ObjGrid_ω.Values
    Nd, Nk, Nx, Nz, Nω = ObjGrid_D.N, ObjGrid_K.N, ObjGrid_X.N, ObjGrid_Z.N, ObjGrid_ω.N
    
    # Allocating
    V_Firm = zeros(Nd, Nk, Nx, Nz, Nω)
    Gd = zeros(Nd, Nk, Nx, Nz, Nω)
    Gk = zeros(Nd, Nk, Nx, Nz, Nω)
    Gx = zeros(Nd, Nk, Nx, Nz, Nω)
    Ge = zeros(Nd, Nk, Nx, Nz, Nω)

    # Loop over all states
    for (id, d) in enumerate(Grid_D), (ik, k) in enumerate(Grid_K), (ix, x) in enumerate(Grid_X), (iz, z) in enumerate(Grid_Z), (iω, ω) in enumerate(Grid_ω)
        s̃  = CartesianIndex(ik, ix, iz, iω)
        s  = CartesianIndex(id, ik, ix, iz, iω)                        
        nw = punit(Price_t.pp, x, ω) * z * k^α + (1.0 - δ) * k - d

        # If nw > nthresh_uncons, then the firm is unconstrained
        if nw > Sol_Unconstrained.nthresh_uncons[s̃]
            V_Firm[s] = V_Unconstrained[s̃]
            Gd[s]     = Sol_Unconstrained.Dmsp[s̃]
            Gk[s]     = K_Unconstrained[s̃]
            Ge[s]     = E_Unconstrained[s̃]
        else
            # If the firm is constrained, solve DRs
            πval = profit(Price_t.pp, x, ω, z, k, E_Unconstrained[s̃]; Param)
            V_Firm[s] = πval + (1.0 - δ) * k - Dmsp[s̃]
        end
                    
    end

    return V_Firm
end


## Solve constrained firm problem
function solve_constrained_firm(Price_t; Param)
    @unpack (ObjGrid_D, ObjGrid_K, ObjGrid_X, ObjGrid_ω, ObjGrid_Z) = Param
    @unpack (α, δ, χ) = Param
    Grid_D = ObjGrid_D.Values
    Grid_K = ObjGrid_K.Values
    Grid_X = ObjGrid_X.Values
    Grid_Z = ObjGrid_Z.Values
    Grid_ω = ObjGrid_ω.Values
    Nd, Nk, Nx, Nz, Nω = ObjGrid_D.N, ObjGrid_K.N, ObjGrid_X.N, ObjGrid_Z.N, ObjGrid_ω.N
    # Allocating
    
    # Guesses
    V_Constrained = zeros(Nd, Nk, Nx, Nz, Nω)
    Gk = zeros(Nd, Nk, Nx, Nz, Nω)
    Ge = zeros(Nd, Nk, Nx, Nz, Nω)
    GΛ = zeros(Nd, Nk, Nx, Nz, Nω)
    Gξ = zeros(Nd, Nk, Nx, Nz, Nω)    
    # Loop over all states
    for (id, d) in enumerate(Grid_D), (ik, k) in enumerate(Grid_K), (ix, x) in enumerate(Grid_X), (iz, z) in enumerate(Grid_Z), (iω, ω) in enumerate(Grid_ω)
        s = CartesianIndex(id, ik, ix, iz, iω)





    end
    


end



## First order condition for capital choice
function GkFOC(d′, k′, id, ik, ix, iz, iω, Gk, Ge, Gλ,  Price_tf, Param) 
    @unpack (α, δ, χ, ObjGrid_ω, ObjGrid_Z, ObjGrid_X, ObjGrid_K) = Param
    Grid_D = ObjGrid_D.Values
    Grid_K = ObjGrid_K.Values
    Grid_X = ObjGrid_X.Values
    Grid_Z = ObjGrid_Z.Values
    Grid_ω = ObjGrid_ω.Values
    
    # Compute expected marginal return to capital
    Expectation = 0.0
    for (iω′, ω′) in enumerate(Grid_ω), (iz′, z′) in enumerate(Grid_Z)
        
        prob_zω = ObjGrid_ω.Prob[iω, iω′] * ObjGrid_Z.Prob[iz, iz′]
        if Grid_X[ix] == 0
            # Future x′ depends on engagement choice today
            x′_val  = Ge[id, ik, ix, iz, iω]   # must be 0.0 or 1.0
            ix′     = (x′_val == 0) ? 1 : 2
            prob_x  = 1.0
            x′      = Grid_X[ix′]
            kstar′′ = linterp2(Grid_D, Grid_K, Gk[:, :, ix′, iz′, iω′], d′, k′)
            Gλ_val  = linterp2(Grid_D, Grid_K, Gλ[:, :, ix′, iz′, iω′], d′, k′) 
            adjλ    = 1.0 + (1.0 - χ) * Gλ_val
            Expectation += prob_x * prob_zω * (adjλ * (punit(Price_tf.pp, x′, ω′) * z′ * α * (k′)^(α - 1.0) + (1.0 - δ)) - 
                           (1.0 - χ) * dk_kadjfnc(kstar′′, k′, Param))
        else
            # No engagement choice today, so future x′ depends on depreciation probability
            for ix′ in 1:ObjGrid_X.N
                x′      = Grid_X[ix′]
                prob_x  = ObjGrid_X.Prob[ix′]
                kstar′′ = linterp2(Grid_D, Grid_K, Gk[:, :, ix′, iz′, iω′], d′, k′)
                Gλ_val  = linterp2(Grid_D, Grid_K, Gλ[:, :, ix′, iz′, iω′], d′, k′) 
                adjλ    = 1.0 + (1.0 - χ) * Gλ_val
                Expectation += prob_x * prob_zω * (adjλ * (punit(Price_tf.pp, x′, ω′) * z′ * α * (k′)^(α - 1.0) + (1.0 - δ)) - 
                               (1.0 - χ) * dk_kadjfnc(kstar′′, k′, Param))
            end
        end
    end
    
    # Objective for policy function iteration: G(k′) = 0
    kval = Grid_K[ik]
    G = (1.0 + dk′_kadjfnc(k′, kval, Param)) * (1.0 + Gλ[id, ik, ix, iz, iω]) - Price_tf.SDF * Expectation

    return G
end

# Policy function iteration
function solve_k(Gd, Ge, Gλ, Price_t, Param)
    @unpack (ObjGrid_D, ObjGrid_K, ObjGrid_X, ObjGrid_ω, ObjGrid_Z) = Param
    @unpack (δ) = Param
    Nd, Nk, Nx, Nz, Nω = ObjGrid_D.N, ObjGrid_K.N, ObjGrid_X.N, ObjGrid_Z.N, ObjGrid_ω.N
    # Initializing
    Gk = zeros(Nd, Nk, Nx, Nz, Nω)
    for id in 1:Nd
        for ik in 1:Nk
            for ix in 1:Nx
                for iz in 1:Nz
                    for iω in 1:Nω
                        Gk[id, ik, ix, iz, iω] = (1.0 - δ) * ObjGrid_K.Values[ik]
                    end
                end
            end
        end
    end
    K_old = copy(Gk)
    K_new = copy(Gk)
    for iter in 1:1000
        println("Iteration: ", iter)    
        for id in 1:Nd
            for ik in 1:Nk
                for ix in 1:Nx
                    for iz in 1:Nz
                        for iω in 1:Nω
                            d′ = Gd[id, ik, ix, iz, iω]
                            k′_star = find_zero(k′ -> GkFOC(d′, k′, id, ik, ix, iz, iω, K_old, Ge, Gλ, Price_t, Param), (0.01, 1000.0), Bisection())
                            K_new[id, ik, ix, iz, iω] = k′_star
                        end
                    end
                end
            end
        end
        
        # Check convergence of policy function iteration
        max_diff = maximum(abs.(K_new - K_old))
        println("Max difference in policy function: ", max_diff)
        if max_diff < 1e-4
            println("Convergence (k*)  achieved after ", iter, " iterations, with max difference: ", max_diff)
            break
        end
        K_old .= K_new
    end
    Gk .= K_new
    
    return Gk
end


## Update Gd from the binding dividend constraint
function update_Gd(Gd, Gk, Ge, Gλ, Price_t, Param)
    @unpack (ObjGrid_D, ObjGrid_K, ObjGrid_X, ObjGrid_ω, ObjGrid_Z) = Param
    Grid_D = ObjGrid_D.Values
    Grid_K = ObjGrid_K.Values
    Grid_X = ObjGrid_X.Values
    Grid_Z = ObjGrid_Z.Values
    Grid_ω = ObjGrid_ω.Values

    # Loop over all states
    for (id, d) in enumerate(Grid_D), (ik, k) in enumerate(Grid_K), (ix, x) in enumerate(Grid_X), (iz, z) in enumerate(Grid_Z), (iω, ω) in enumerate(Grid_ω)
        # Update Gd based on the current policy functions and value function derivatives
        eval  = Ge[id, ik, ix, iz, iω]
        k′    = Gk[id, ik, ix, iz, iω]
        π_val = profit(Price_t.pp, x, ω, z, k, eval; Param)
        Gd[id, ik, ix, iz, iω] = (π_val - k′ - (1.0 - δ) * k - kadjfnc(k′, k, Param) - d) * Price_t.R
    end

    return Gd
end


## Update multiplier on dividend constraint
function update_Gλ(Gd, Gk, Ge, Gλ, Price_t, Param)
    @unpack (ObjGrid_D, ObjGrid_K, ObjGrid_X, ObjGrid_ω, ObjGrid_Z) = Param
    Grid_D = ObjGrid_D.Values
    Grid_K = ObjGrid_K.Values
    Grid_X = ObjGrid_X.Values
    Grid_Z = ObjGrid_Z.Values
    Grid_ω = ObjGrid_ω.Values

    # Loop over all states
    for (id, d) in enumerate(Grid_D), (ik, k) in enumerate(Grid_K), (ix, x) in enumerate(Grid_X), (iz, z) in enumerate(Grid_Z), (iω, ω) in enumerate(Grid_ω)
        # Update Gλ based on the current policy functions and value function derivatives
        Expectation = 0.0
        for (iz′, z′) in enumerate(Grid_Z), (iω′, ω′) in enumerate(Grid_ω)
            prob_zω = ObjGrid_ω.Prob[iω, iω′] * ObjGrid_Z.Prob[iz, iz′]
            d′      = Gd[id, ik, ix, iz, iω]
            k′      = Gk[id, ik, ix, iz, iω]
            λ′_val  = linterp2(Grid_D, Grid_K, Gλ[:, :, ix, iz′, iω′], d′, k′)
            temp    = 1.0 + (1.0 - χ) * λ′_val
            Expectation += prob_zω * temp
        end
        Gλ[id, ik, ix, iz, iω] = Price_t.R * Price_t.SDF * Expectation
    end

    return Gλ
end


## Check if borrowing constraint is binding or if it is slacks
function check_dcons_binds(Gd, Gk, Ge, Gλ, Price_t, Param)
    @unpack (ObjGrid_D, ObjGrid_K, ObjGrid_X, ObjGrid_ω, ObjGrid_Z) = Param
    Grid_D = ObjGrid_D.Values
    Grid_K = ObjGrid_K.Values
    Grid_X = ObjGrid_X.Values
    Grid_Z = ObjGrid_Z.Values
    Grid_ω = ObjGrid_ω.Values

    # Loop over all states
    for (id, d) in enumerate(Grid_D), (ik, k) in enumerate(Grid_K), (ix, x) in enumerate(Grid_X), (iz, z) in enumerate(Grid_Z), (iω, ω) in enumerate(Grid_ω)
        # Check if the borrowing constraint is binding or if it is slack
        eval  = Ge[id, ik, ix, iz, iω]
        d′    = Gd[id, ik, ix, iz, iω]
        π_val = profit(Price_t.pp, x, ω, z, k, eval; Param)
        check_dcons = (ϕ * π_val - d′ > 1e-4) ? true : false
    end

    return check_dcons
end


## Use the outcome of check_dcons_binds and retrieve decision rules for currently constrained firms
function objk′_constrained(k′, k, d′, nw, Price_t, Param)
    obj = nw - k′ - kadjfnc(k′, k, Param) + (1.0/Price_t.R) * d′
    return obj
end


function solve_constrained_firm(Gd, Gk, Ge, Gλ, Price_t, Param)
    @unpack (ObjGrid_D, ObjGrid_K, ObjGrid_X, ObjGrid_ω, ObjGrid_Z) = Param
    Grid_D = ObjGrid_D.Values
    Grid_K = ObjGrid_K.Values
    Grid_X = ObjGrid_X.Values
    Grid_Z = ObjGrid_Z.Values
    Grid_ω = ObjGrid_ω.Values

    # Loop over all states
    for (id, d) in enumerate(Grid_D), (ik, k) in enumerate(Grid_K), (ix, x) in enumerate(Grid_X), (iz, z) in enumerate(Grid_Z), (iω, ω) in enumerate(Grid_ω)
        # Check if the borrowing constraint is binding or if it is slack
        eval  = Ge[id, ik, ix, iz, iω]
        d′    = Gd[id, ik, ix, iz, iω]
        π_val = profit(Price_t.pp, x, ω, z, k, eval; Param)
        check_dcons = (ϕ * π_val - d′ > 1e-4) ? true : false
        
        if check_dcons
            # If the borrowing constraint is slack, solve for optimal policies as if unconstrained
            # This would involve solving the FOCs for k and e as if the constraint does not bind
            # You can use a root-finding algorithm to solve the FOCs for these states
            # For example:
            k′_star = find_zero(k′ -> GkFOC(d′, k′, id, ik, ix, iz, iω, Gk, Ge, Gλ, Price_t, Param), (0.01, 1000.0), Bisection())
            Gk[id, ik, ix, iz, iω] = k′_star
            # Similarly solve for engagement choice e using its FOC (not shown here)
        else
            d′   = ϕ * π_val
            eval = Ge[id, ik, ix, iz, iω]
            nw   = networth(Price_t.pp, x, ω, z, k, eval, d; Param)
            # If the borrowing constraint is binding, then the optimal policy is to set d′ equal to ϕ * π_val
            Gd[id, ik, ix, iz, iω] = d′
            # The capital choice is pinned down by the dividend constraint
            # Solve for a root in objk′_constrained
            k′ = find_zero(k′ -> objk′_constrained(k′, k, d′, nw, Price_t, Param), (0.01, 1000.0), Bisection())
            Gk[id, ik, ix, iz, iω] = k′

        end
    end

    return Gd, Gk
end


## Use Gd, Gk to update value at each coordinate (id, ik, ix, iz, iω) and engagement choice
function update_value_cons(id, ik, ix, iz, iω, Gd, Gk, Ge, Gλ, Price_t, Param)
    @unpack (ObjGrid_D, ObjGrid_K, ObjGrid_X, ObjGrid_ω, ObjGrid_Z) = Param
    Grid_D = ObjGrid_D.Values
    Grid_K = ObjGrid_K.Values
    Grid_X = ObjGrid_X.Values
    Grid_Z = ObjGrid_Z.Values
    Grid_ω = ObjGrid_ω.Values

    d′    = Gd[id, ik, ix, iz, iω]
    k′    = Gk[id, ik, ix, iz, iω]
    eval  = Ge[id, ik, ix, iz, iω]
    nw    = networth(Price_t.pp, Grid_X[ix], Grid_ω[iω], Grid_Z[iz], Grid_K[ik], eval, d′; Param)
    
    # Update value function for constrained firm using the optimal policies and the value function derivatives
    flow = nw - k′ - kadjfnc(k′, Grid_K[ik], Param) + (1.0/Price_t.R) * d′
    expectation   = 0.0
    V_constrained = flow + Price_t.SDF * expectation

    return V_constrained
end




## Testing solve_k
@unpack (ObjGrid_D, ObjGrid_K, ObjGrid_X, ObjGrid_ω, ObjGrid_Z) = Param
Gd = zeros(ObjGrid_D.N, ObjGrid_K.N, ObjGrid_X.N, ObjGrid_Z.N, ObjGrid_ω.N)
Ge = zeros(ObjGrid_D.N, ObjGrid_K.N, ObjGrid_X.N, ObjGrid_Z.N, ObjGrid_ω.N)
Gλ = zeros(ObjGrid_D.N, ObjGrid_K.N, ObjGrid_X.N, ObjGrid_Z.N, ObjGrid_ω.N)
Gk = solve_k(Gd, Ge, Gλ, Price_t, Param)
