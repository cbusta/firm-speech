# Tools_Steady.jl: Functions to compute steady state equilibrium
# Combines unconstrained and constrained firm solutions, computes distribution,
# and solves for equilibrium prices via bisection on excess demand.

using Statistics
using Roots


## Combine unconstrained and constrained solutions
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


## Compute excess demand for non-differentiated good given prices
# Returns excess demand: positive means demand > supply
function excess_cNP(Price_t, Param; verbose=false)
    # Step 1: Solve unconstrained firm problem
    if verbose
        println("\n=== Solving for pp = $(round(Price_t.pp, digits=4)) ===")
    end
    Sol_Unconstrained = solve_unconstrained_firm(Param; Price_t=Price_t, verbose=verbose)

    # Step 2: Solve constrained firm problem
    Sol_Constrained = solve_constrained_firm(Param; Price_t=Price_t, verbose=verbose)

    # Step 3: Combine solutions
    Sol_Firm = value_firm(Sol_Unconstrained, Sol_Constrained, Price_t; Param)

    # Step 4: Compute stationary distribution
    Sol_Dist = solve_stationary_distribution(Sol_Firm, Param; verbose=verbose)

    # Step 5: Compute aggregates
    agg = compute_aggregates(Sol_Dist.μ, Sol_Firm, Sol_Unconstrained, Param, Price_t)

    # Step 6: Compute excess demand for non-differentiated good
    # Demand: investment + adjustment costs + engagement costs
    # Supply: output from non-engaged firms (x=0)
    demand_cNP = agg.agg_investment + agg.agg_adj_cost + agg.agg_engagement_cost
    supply_cNP = agg.agg_cNP
    excess = demand_cNP - supply_cNP

    if verbose
        println("\n  Market clearing for cNP:")
        println("    Demand (I + adj + engage): $(round(demand_cNP, digits=6))")
        println("    Supply (cNP):              $(round(supply_cNP, digits=6))")
        println("    Excess demand:             $(round(excess, digits=6))")
    end

    return (
        excess       = excess,
        demand_cNP   = demand_cNP,
        supply_cNP   = supply_cNP,
        agg          = agg,
        Sol_Firm     = Sol_Firm,
        Sol_Dist     = Sol_Dist,
        Sol_Unconstrained = Sol_Unconstrained,
        Sol_Constrained   = Sol_Constrained,
        Price_t      = Price_t
    )
end


## Solve for steady state equilibrium by bisecting on pp
function solve_steady(Param;
                      pp_lo=0.5,
                      pp_hi=3.0,
                      R=1.04,
                      tol=1e-4,
                      verbose=true)

    if verbose
        println("\n===============================================================")
        println("           SOLVING FOR STEADY STATE EQUILIBRIUM")
        println("===============================================================")
        println("  Bisection on pp ∈ [$pp_lo, $pp_hi]")
        println("  Interest rate R = $R")
        println("  Tolerance = $tol")
        println("===============================================================\n")
    end

    # Store final result for returning full solution
    final_result = Ref{Any}(nothing)

    # Define objective function: excess demand as function of pp
    function f(pp)
        Price_t = (R=R, SDF=1.0/R, pp=pp)
        result = excess_cNP(Price_t, Param; verbose=false)
        final_result[] = result
        if verbose
            println("  pp = $(round(pp, digits=6)):  excess = $(round(result.excess, sigdigits=4))")
        end
        return result.excess
    end

    # Use Roots.jl bisection
    pp_star = find_zero(f, (pp_lo, pp_hi), Bisection(); atol=tol)

    # Get the final result at equilibrium
    result = final_result[]

    if verbose
        println("\n---------------------------------------------------------------")
        println("  STEADY STATE SOLUTION")
        println("---------------------------------------------------------------")
        println("  Equilibrium pp:        $(round(pp_star, digits=6))")
        println("  Final excess demand:   $(round(result.excess, sigdigits=4))")
        println("---------------------------------------------------------------")
        print_aggregates(result.agg)
    end

    return (
        pp_star           = pp_star,
        Price_t           = (R=R, SDF=1.0/R, pp=pp_star),
        excess            = result.excess,
        agg               = result.agg,
        Sol_Firm          = result.Sol_Firm,
        Sol_Dist          = result.Sol_Dist,
        Sol_Unconstrained = result.Sol_Unconstrained,
        Sol_Constrained   = result.Sol_Constrained
    )
end
