import Pkg
Pkg.activate(dirname(@__DIR__))

using Revise, UnPack
using SparseArrays, Roots, LinearAlgebra
using Plots
using Statistics
using JLD2
using Lib_Julia_Econ


## Helpers
# If there are problem with path, use: include(joinpath(@__DIR__, "Parameters.jl"))
include("Parameters.jl")
include("Tools_Simple_Fns.jl")
include("Tools_Unconstrained_Firms.jl")
include("Tools_Constrained_Firms.jl")
Param = Get_Params()
Param.ObjGrid_X


## Given some prices and solve unconstrained problem in hard scope
Sol_Unconstrained = solve_unconstrained_firm(Param)
V_Unconstrained   = Sol_Unconstrained.V_Unconstrained
E_Unconstrained   = Sol_Unconstrained.E_Unconstrained
TE_Unconstrained  = Sol_Unconstrained.TE_Unconstrained
K_Unconstrained   = Sol_Unconstrained.K_Unconstrained
Dmsp    = Sol_Unconstrained.Dmsp
Price_t = Sol_Unconstrained.Price_t


## Analysis
# Plotting efficient capital policy (izω=1 corresponds to first joint (z,ω) state)
plot(Param.ObjGrid_K.Values, K_Unconstrained[:, 1, 1], label="x=0", xlabel="Current Capital (k)", ylabel="Optimal Next Period Capital (k′)", title="Efficient Capital Policy (ix=1, izω=1)")
# Add 45 degree line
plot!(Param.ObjGrid_K.Values, Param.ObjGrid_K.Values, label="45 Degree Line", linestyle=:dash)


## Plotting unconstrained threshold
plot(Param.ObjGrid_K.Values, Sol_Unconstrained.nthresh_uncons[:, 1, 1], label="x=0",
     xlabel="Current Capital (k)", ylabel="Unconstrained Threshold (n*)", title="Unconstrained Threshold (ix=1, izω=1)",
     lw=2)

## Relationship of engagement with size
# Empirical evidence: larger firms more engaged
# Want to plot engagement policy (E_Unconstrained) against capital (K_Unconstrained) for different (z,ω) states
plot(Param.ObjGrid_K.Values, E_Unconstrained[:, 1, 1], label="x=0, izω=1", xlabel="Current Capital (k)", ylabel="Engagement Policy (E)", title="Engagement Policy vs Capital (ix=1, izω=1)")
plot!(Param.ObjGrid_K.Values, E_Unconstrained[:, 1, 2], label="x=0, izω=2")



## Solve constrained problem and combine solutions
# Sol_Constrained = solve_constrained_firm(Param; Price_t = Price_t)
# Sol_Firm = value_firm(Sol_Unconstrained, Sol_Constrained, Price_t; Param)


## Solve firm decision rules
# This function combines unconstrained and constrained solutions
# For each state s = (d, k, x, zω), check if firm is unconstrained or constrained
# and return the appropriate value function and policy functions
function value_firm(Sol_Unconstrained, Sol_Constrained, Price_t; Param)
    @unpack (ObjGrid_D, ObjGrid_K, ObjGrid_X, ObjGrid_Zω) = Param
    @unpack (α, δ) = Param
    Grid_D = ObjGrid_D.Values
    Grid_K = ObjGrid_K.Values
    Grid_X = ObjGrid_X.Values
    Grid_Zω = ObjGrid_Zω.Values
    Nd, Nk, Nx, Nzω = ObjGrid_D.N, ObjGrid_K.N, ObjGrid_X.N, ObjGrid_Zω.N

    # Extract unconstrained solutions
    V_Unconstrained = Sol_Unconstrained.V_Unconstrained
    E_Unconstrained = Sol_Unconstrained.E_Unconstrained
    K_Unconstrained = Sol_Unconstrained.K_Unconstrained
    Dmsp            = Sol_Unconstrained.Dmsp
    nthresh_uncons  = Sol_Unconstrained.nthresh_uncons

    # Extract constrained solutions
    V_Constrained = Sol_Constrained.V_Constrained
    E_Constrained = Sol_Constrained.E_Constrained
    K_Constrained = Sol_Constrained.K_Constrained
    D_Constrained = Sol_Constrained.D_Constrained

    # Allocating output arrays
    V_Firm = zeros(Nd, Nk, Nx, Nzω)
    Gd = zeros(Nd, Nk, Nx, Nzω)
    Gk = zeros(Nd, Nk, Nx, Nzω)
    Ge = zeros(Nd, Nk, Nx, Nzω)
    Is_Unconstrained = zeros(Bool, Nd, Nk, Nx, Nzω)  # Indicator for unconstrained firms

    # Loop over all states
    for id in 1:Nd
        d = Grid_D[id]
        for ik in 1:Nk
            k = Grid_K[ik]
            for ix in 1:Nx
                x = Grid_X[ix]
                for izω in 1:Nzω
                    z, ω = Grid_Zω[:, izω]

                    # Compute net worth
                    nw = punit(Price_t.pp, x, ω) * z * k^α + (1.0 - δ) * k - d

                    # Check if firm is unconstrained
                    if nw > nthresh_uncons[ik, ix, izω]
                        # Unconstrained firm: value is linear in debt
                        # V(d,k,x,zω) = V*(k,x,zω) - d (since marginal value of wealth = 1)
                        V_Firm[id, ik, ix, izω] = V_Unconstrained[ik, ix, izω] - d
                        Gd[id, ik, ix, izω]     = Dmsp[ik, ix, izω]
                        Gk[id, ik, ix, izω]     = K_Unconstrained[ik, ix, izω]
                        Ge[id, ik, ix, izω]     = E_Unconstrained[ik, ix, izω]
                        Is_Unconstrained[id, ik, ix, izω] = true
                    else
                        # Constrained firm: use constrained solution
                        V_Firm[id, ik, ix, izω] = V_Constrained[id, ik, ix, izω]
                        Gd[id, ik, ix, izω]     = D_Constrained[id, ik, ix, izω]
                        Gk[id, ik, ix, izω]     = K_Constrained[id, ik, ix, izω]
                        Ge[id, ik, ix, izω]     = E_Constrained[id, ik, ix, izω]
                        Is_Unconstrained[id, ik, ix, izω] = false
                    end
                end
            end
        end
    end

    return (
        V_Firm           = V_Firm,
        Gd               = Gd,
        Gk               = Gk,
        Ge               = Ge,
        Is_Unconstrained = Is_Unconstrained
    )
end



## Solve the firm problem and get combined solution
Sol_Unconstrained = solve_unconstrained_firm(Param)
Sol_Constrained   = solve_constrained_firm(Param; Price_t = Sol_Unconstrained.Price_t)
Sol_Firm = value_firm(Sol_Unconstrained, Sol_Constrained, Sol_Unconstrained.Price_t; Param)


## Save results to jld2 file
# Create results directory if it doesn't exist
results_dir = joinpath(dirname(@__DIR__), "results")
mkpath(results_dir)
# Save results
@save joinpath(results_dir, "Firm_Solutions.jld2") Sol_Unconstrained Sol_Constrained Sol_Firm Param



