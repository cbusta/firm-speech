import Pkg
Pkg.activate(dirname(@__DIR__))

using Revise, UnPack
using SparseArrays, Roots, LinearAlgebra
using Plots, Statistics, Distributions
using JLD2
using Lib_Julia_Econ
using LaTeXStrings


## Helpers
# If there are problem with path, use: include(joinpath(@__DIR__, "Parameters.jl"))
include("Parameters.jl")
include("Tools_Simple_Fns.jl")
include("Tools_Unconstrained_Firms.jl")
include("Tools_Constrained_Firms.jl")
include("Tools_Distribution.jl")
include("Tools_Steady.jl")
Param = Get_Params()
Param.ObjGrid_X


## Solve the firm problem and get combined solution
# Option 1: Solve at fixed prices
Sol_Unconstrained = solve_unconstrained_firm(Param)
Sol_Constrained   = solve_constrained_firm(Param; Price_t = Sol_Unconstrained.Price_t)
Sol_Firm = value_firm(Sol_Unconstrained, Sol_Constrained, Sol_Unconstrained.Price_t; Param)

# Option 2: Solve for steady state equilibrium (uncomment to use)
# Steady = solve_steady(Param; pp_lo=0.5, pp_hi=3.0, R=1.04, verbose=true)
# Sol_Unconstrained = Steady.Sol_Unconstrained
# Sol_Constrained   = Steady.Sol_Constrained
# Sol_Firm          = Steady.Sol_Firm


## Save results to jld2 file
# Create results directory if it doesn't exist
results_dir = joinpath(dirname(@__DIR__), "results")
mkpath(results_dir)
# Save results
@save joinpath(results_dir, "Firm_Solutions.jld2") Sol_Unconstrained Sol_Constrained Sol_Firm Param


## Given some prices and solve unconstrained problem in hard scope
V_Unconstrained   = Sol_Unconstrained.V_Unconstrained
E_Unconstrained   = Sol_Unconstrained.E_Unconstrained
K_Unconstrained   = Sol_Unconstrained.K_Unconstrained
Dmsp    = Sol_Unconstrained.Dmsp
Price_t = Sol_Unconstrained.Price_t


## Analysis
# Plotting efficient capital policy (izω=1 corresponds to first joint (z,ω) state)
p1 = plot(Param.ObjGrid_K.Values, K_Unconstrained[:, 1, 1], lw = 3, label=L"K^*",
          xlabel="Current Capital (k)", ylabel="Optimal Next Period Capital (k′)", title="Efficient Capital Policy (ix=1, izω=1)")
p1 = plot!(Param.ObjGrid_K.Values, Param.ObjGrid_K.Values, label="45 Degree Line", linestyle=:dash)
p2 = plot(Param.ObjGrid_K.Values, Sol_Unconstrained.nthresh_uncons[:, 1, 1], lw = 3, label = L"\bar{n}",
          xlabel="Current Capital (k)", ylabel="Unconstrained Threshold (n*)", title="Unconstrained Threshold (ix=1, izω=1)")
plot(p1, p2, layout=(2, 1), size=(600, 600))

## Relationship of engagement with size
# Empirical evidence: larger firms more engaged
# Want to plot engagement policy (E_Unconstrained) against capital (K_Unconstrained) for different (z,ω) states
plot(Param.ObjGrid_K.Values, E_Unconstrained[:, 1, 1], lw = 3, label="x=0, izω=1", 
     xlabel="Current Capital (k)", ylabel="Engagement Policy (E)", title="Engagement Policy vs Capital (ix=1, izω=1)")
plot!(Param.ObjGrid_K.Values, E_Unconstrained[:, 1, 23], lw = 3, label="x=0, izω=23")



## Solve for stationary distribution
Sol_Dist = solve_stationary_distribution(Sol_Firm, Param)

# Compute and print aggregates
agg = compute_aggregates(Sol_Dist.μ, Sol_Firm, Sol_Unconstrained, Param, Price_t)
print_aggregates(agg)

# Analyze engagement by firm size
size_stats = engagement_by_size(Sol_Dist.μ, Sol_Firm, Param)


## Plot the distribution of entrants over capital
# Use create_entry_distribution (returns distribution on dense grid)
entry_dist = create_entry_distribution(Param)
# Marginalise over d, x, zω to get capital distribution of entrants
entry_k_dist = dropdims(sum(entry_dist, dims=(1, 3, 4)), dims=(1, 3, 4))

# Plot the distribution of entrants over z and ω
# Since entry_dist is (Ndg, Nkg, Nx, Nzω), we need to compute marginals over the joint (z,ω) process
# First, marginalize over d, k, x to get distribution over zω
entry_zω_dist = dropdims(sum(entry_dist, dims=(1, 2, 3)), dims=(1, 2, 3))  # Length Nzω

# Compute marginal over z by summing over ω states
Nz = Param.ObjGrid_Z.N
Nω = Param.ObjGrid_ω.N
entry_z_dist = zeros(Nz)
entry_ω_dist = zeros(Nω)
for izω in 1:Param.ObjGrid_Zω.N
    iz = ((izω - 1) ÷ Nω) + 1
    iω = ((izω - 1) % Nω) + 1
    entry_z_dist[iz] += entry_zω_dist[izω]
    entry_ω_dist[iω] += entry_zω_dist[izω]
end

Grid_Z = Param.ObjGrid_Z.Values
Grid_ω = Param.ObjGrid_ω.Values
p1 = plot(Grid_Z, entry_z_dist, lw = 3, legend=false,
          xlabel="Productivity (z)", ylabel="Mass", title="Distribution of Entrants over z")
p2 = plot(Grid_ω, entry_ω_dist, lw = 3, legend=false,
          xlabel="Taste Shifter (ω)", ylabel="Mass", title="Distribution of Entrants over ω")
p3 = plot(Param.ObjDGrid_K.Values, entry_k_dist, lw = 3, legend=false,
          xlabel="Capital (k)", ylabel="Mass", title="Distribution of Entrants over k")
plot(p1, p2, p3, layout=(2,2), size=(600, 600))


## Plot the size distribution
# Get capital distribution from stationary distribution (on dense grid)
Ndg, Nkg = Param.ObjDGrid_D.N, Param.ObjDGrid_K.N
capital_dist = zeros(Nkg)
debt_dist = zeros(Ndg)
for id in 1:Ndg
    for ik in 1:Nkg
        for ix in 1:Param.ObjGrid_X.N
            for izω in 1:Param.ObjGrid_Zω.N
                capital_dist[ik] += Sol_Dist.μ[id, ik, ix, izω]
                debt_dist[id] += Sol_Dist.μ[id, ik, ix, izω]
            end
        end
    end
end
plot(Param.ObjDGrid_K.Values, capital_dist, lw = 3, label="Capital Distribution",
     xlabel="Capital (k)", ylabel="Mass", title="Stationary Distribution of Capital")


## Compute percentiles of capital distribution (using dense grid)
capital_cdf = cumsum(capital_dist) / sum(capital_dist)
debt_cdf = cumsum(debt_dist) / sum(debt_dist)
# Use searchsortedfirst to avoid nothing from findfirst
capital_percentiles = [Param.ObjDGrid_K.Values[min(searchsortedfirst(capital_cdf, p/100), Nkg)] for p in 1:100]
debt_percentiles = [Param.ObjDGrid_D.Values[min(searchsortedfirst(debt_cdf, p/100), Ndg)] for p in 1:100]


## Plot the distribution of engagement by size (using dense grid for μ, interpolate Ge)
engagement_by_size_dist = zeros(Nkg)
Grid_D_coarse = Param.ObjGrid_D.Values
Grid_K_coarse = Param.ObjGrid_K.Values
for id in 1:Ndg
    d = Param.ObjDGrid_D.Values[id]
    for ik in 1:Nkg
        k = Param.ObjDGrid_K.Values[ik]
        for ix in 1:Param.ObjGrid_X.N
            for izω in 1:Param.ObjGrid_Zω.N
                # Interpolate engagement policy from coarse grid to dense grid point
                e = linterp2(Grid_D_coarse, Grid_K_coarse, Sol_Firm.Ge[:, :, ix, izω], d, k)
                engagement_by_size_dist[ik] += Sol_Dist.μ[id, ik, ix, izω] * e
            end
        end
    end
end
plot(Param.ObjDGrid_K.Values, engagement_by_size_dist, lw = 3, label="Engagement by Size",
     xlabel="Capital (k)", ylabel="Mass * Engagement", title="Distribution of Engagement by Size")


## Plot the distribution of political capital (x) by size (using dense grid)
political_capital_by_size_dist = zeros(Nkg)
for id in 1:Ndg
    for ik in 1:Nkg
        for ix in 1:Param.ObjGrid_X.N
            for izω in 1:Param.ObjGrid_Zω.N
                political_capital_by_size_dist[ik] += Sol_Dist.μ[id, ik, ix, izω] * Param.ObjGrid_X.Values[ix]
            end
        end
    end
end
plot(Param.ObjDGrid_K.Values, political_capital_by_size_dist, lw = 3, label="Political Capital by Size",
     xlabel="Capital (k)", ylabel="Mass * Political Capital", title="Distribution of Political Capital by Size")


## Plot fraction of firms with x=1 at each size percentile (using dense grid)
fraction_x1_by_size = zeros(100)
DGrid_K = Param.ObjDGrid_K.Values
for p in 1:100
    # Find capital threshold for this percentile using searchsortedfirst
    k_idx = min(searchsortedfirst(capital_cdf, p/100), Nkg)
    k_threshold = DGrid_K[k_idx]

    # Compute fraction of firms with x=1 below this capital threshold
    mass_below_threshold = 0.0
    mass_x1_below_threshold = 0.0
    for id in 1:Ndg
        for ik in 1:Nkg
            if DGrid_K[ik] <= k_threshold
                for ix in 1:Param.ObjGrid_X.N
                    for izω in 1:Param.ObjGrid_Zω.N
                        mass_below_threshold += Sol_Dist.μ[id, ik, ix, izω]
                        if Param.ObjGrid_X.Values[ix] == 1
                            mass_x1_below_threshold += Sol_Dist.μ[id, ik, ix, izω]
                        end
                    end
                end
            end
        end
    end

    fraction_x1_by_size[p] = mass_x1_below_threshold / max(mass_below_threshold, 1e-14)
end
plot(1:100, fraction_x1_by_size, lw = 3, label="Fraction with x=1",
     xlabel="Size Percentile", ylabel="Fraction with x=1", title="Fraction of Firms with x=1 by Size Percentile")



## Market clearing for the non-differentiated good
# The non-differentiated good is used for investment, including capital adj costs, and political engagement costs
demand_cNP = agg.agg_investment + agg.agg_adj_cost + agg.agg_engagement_cost
excess_demand_cNP = demand_cNP - agg.agg_cNP
println("Aggregate demand for non-differentiated good: $demand_cNP")
println("Aggregate supply of non-differentiated good:  $(agg.agg_cNP)")
println("Excess demand for non-differentiated good:    $excess_demand_cNP")