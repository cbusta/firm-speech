# Tools_Distribution.jl: Functions to compute stationary distribution of firms
# State space: (d, k, x, zω) with dimensions (Nd, Nk, Nx, Nzω)
# Uses lottery method to distribute mass from continuous policy functions to discrete grid

using Base.Threads: @threads


## Helper: Compute interpolation weights for distributing mass to grid points
# Given a continuous value val and a grid, returns (index_lo, index_hi, weight_lo)
# such that val ≈ weight_lo * grid[index_lo] + (1-weight_lo) * grid[index_hi]
@inline function get_interp_weights(val, grid)
    N = length(grid)

    # Handle boundary cases
    if val <= grid[1]
        return 1, 1, 1.0
    elseif val >= grid[N]
        return N, N, 1.0
    end

    # Binary search for bracketing indices
    lo = 1
    hi = N
    while hi - lo > 1
        mid = (lo + hi) ÷ 2
        if grid[mid] <= val
            lo = mid
        else
            hi = mid
        end
    end

    # Compute weight for lower index
    weight_lo = (grid[hi] - val) / (grid[hi] - grid[lo])

    return lo, hi, weight_lo
end


## Interpolate coarse-grid policies onto the dense grid
# Called once before the distribution iteration; returns DGk, DGd, DGe of size (Ndg, Nkg, Nx, Nzω).
function interpolate_policies_to_dense(Sol_Firm, Param)
    @unpack (ObjGrid_D, ObjGrid_K, ObjDGrid_D, ObjDGrid_K, ObjGrid_X, ObjGrid_Zω) = Param
    Grid_D  = ObjGrid_D.Values
    Grid_K  = ObjGrid_K.Values
    DGrid_D = ObjDGrid_D.Values
    DGrid_K = ObjDGrid_K.Values
    Nx, Nzω    = ObjGrid_X.N, ObjGrid_Zω.N
    Ndg, Nkg   = ObjDGrid_D.N, ObjDGrid_K.N

    Gk = Sol_Firm.Gk   # (Nd, Nk, Nx, Nzω)
    Gd = Sol_Firm.Gd
    Ge = Sol_Firm.Ge

    DGk = zeros(Ndg, Nkg, Nx, Nzω)
    DGd = zeros(Ndg, Nkg, Nx, Nzω)
    DGe = zeros(Ndg, Nkg, Nx, Nzω)

    for izω in 1:Nzω
        for ix in 1:Nx
            for ik in 1:Nkg
                k = DGrid_K[ik]
                for id in 1:Ndg
                    d = DGrid_D[id]
                    DGk[id, ik, ix, izω] = linterp2(Grid_D, Grid_K, Gk[:, :, ix, izω], d, k)
                    DGd[id, ik, ix, izω] = linterp2(Grid_D, Grid_K, Gd[:, :, ix, izω], d, k)
                    DGe[id, ik, ix, izω] = linterp2(Grid_D, Grid_K, Ge[:, :, ix, izω], d, k)
                end
            end
        end
    end

    return DGk, DGd, DGe
end


## Update distribution
# Takes current distribution μ and pre-interpolated dense-grid policies, returns next period μ′.
# DGk, DGd, DGe are (Ndg, Nkg, Nx, Nzω) arrays already evaluated on the dense grid
# (computed once by interpolate_policies_to_dense before the iteration loop).
function update_distribution(μ, DGk, DGd, DGe, Param; χ=nothing)
    @unpack (ObjDGrid_D, ObjDGrid_K, ObjGrid_X, ObjGrid_Zω) = Param
    if isnothing(χ)
        χ = Param.χ
    end
    DGrid_D = ObjDGrid_D.Values
    DGrid_K = ObjDGrid_K.Values
    Grid_X  = ObjGrid_X.Values
    Prob_Zω = ObjGrid_Zω.Prob
    Prob_X  = ObjGrid_X.Prob
    Ndg, Nkg, Nx, Nzω = ObjDGrid_D.N, ObjDGrid_K.N, ObjGrid_X.N, ObjGrid_Zω.N

    μ_next = zeros(Ndg, Nkg, Nx, Nzω)

    for id in 1:Ndg
        for ik in 1:Nkg
            for ix in 1:Nx
                x = Grid_X[ix]
                for izω in 1:Nzω
                    mass = μ[id, ik, ix, izω]
                    mass < 1e-14 && continue

                    # Policy values on dense grid (pre-interpolated)
                    k′ = DGk[id, ik, ix, izω]
                    d′ = DGd[id, ik, ix, izω]
                    e  = DGe[id, ik, ix, izω]

                    # Interpolation weights for k′ and d′ on the dense grid
                    ik_lo, ik_hi, wk_lo = get_interp_weights(k′, DGrid_K)
                    id_lo, id_hi, wd_lo = get_interp_weights(d′, DGrid_D)

                    surv_mass = (1.0 - χ) * mass

                    for jzω in 1:Nzω
                        prob_zω = Prob_Zω[izω, jzω]
                        prob_zω < 1e-14 && continue

                        if x == 0
                            # x=0: engagement choice determines x′
                            ix′ = (e ≈ 1.0) ? 2 : 1
                            for (jd, wd) in ((id_lo, wd_lo), (id_hi, 1.0 - wd_lo))
                                wd < 1e-14 && continue
                                for (jk, wk) in ((ik_lo, wk_lo), (ik_hi, 1.0 - wk_lo))
                                    wk < 1e-14 && continue
                                    μ_next[jd, jk, ix′, jzω] += surv_mass * prob_zω * wd * wk
                                end
                            end
                        else
                            # x=1: stochastic depreciation
                            # Prob_X[1] = δ_x (depreciate to x′=0)
                            # Prob_X[2] = 1-δ_x (stay at x′=1)
                            for ix′ in 1:Nx
                                prob_x = Prob_X[ix′]
                                prob_x < 1e-14 && continue
                                for (jd, wd) in ((id_lo, wd_lo), (id_hi, 1.0 - wd_lo))
                                    wd < 1e-14 && continue
                                    for (jk, wk) in ((ik_lo, wk_lo), (ik_hi, 1.0 - wk_lo))
                                        wk < 1e-14 && continue
                                        μ_next[jd, jk, ix′, jzω] += surv_mass * prob_zω * prob_x * wd * wk
                                    end
                                end
                            end
                        end
                    end
                end
            end
        end
    end

    return μ_next
end


## Add entrant distribution
# New firms enter with probability χ (replacing dead firms)
# Entry distribution: assume firms enter with zero debt, capital drawn from entry distribution
function add_entrants!(μ, entry_dist, χ)
    # Total mass of entrants equals mass of dead firms
    total_mass = sum(μ)
    entry_mass = χ * total_mass / (1.0 - χ)  # To maintain steady state mass

    # Add entrants according to entry distribution
    # entry_dist should be normalized to sum to 1
    μ .+= entry_mass .* entry_dist

    return μ
end


## Create entry distribution from parameters
# Entrants start with: d=d_entry, x=0, k drawn from specified distribution, (z,ω) from specified distribution
function create_entry_distribution(Param)
    @unpack (ObjGrid_D, ObjDGrid_D, ObjDGrid_K, ObjGrid_X, ObjGrid_Zω, EntryDist) = Param
    DGrid_D = ObjDGrid_D.Values
    DGrid_K = ObjDGrid_K.Values
    Ndg, Nkg, Nx, Nzω = ObjDGrid_D.N, ObjDGrid_K.N, ObjGrid_X.N, ObjGrid_Zω.N

    # Extract entry distribution parameters
    d_entry = EntryDist.d_entry
    k_dist  = EntryDist.k_dist
    k_loc   = EntryDist.k_loc
    k_scale = EntryDist.k_scale
    zω_dist = EntryDist.zω_dist

    # Initialize entry distribution on dense grid
    entry_dist = zeros(Ndg, Nkg, Nx, Nzω)

    # Find dense grid point for entry debt (closest to d_entry)
    id_entry = argmin(abs.(DGrid_D .- d_entry))

    # Entry with x = 0 (no political capital)
    ix_entry = 1

    # Capital entry distribution on dense grid
    if k_dist == :lognormal
        k_weights = zeros(Nkg)
        for ik in 1:Nkg
            k = DGrid_K[ik]
            if k > 0
                log_k = log(k)
                k_weights[ik] = exp(-0.5 * ((log_k - k_loc) / k_scale)^2) / (k * k_scale)
            end
        end
    elseif k_dist == :pareto
        k_weights = zeros(Nkg)
        k_min = k_loc
        α_pareto = k_scale
        for ik in 1:Nkg
            k = DGrid_K[ik]
            if k >= k_min
                k_weights[ik] = α_pareto * k_min^α_pareto / k^(α_pareto + 1)
            end
        end
    elseif k_dist == :uniform
        k_weights = ones(Nkg)
    else
        k_weights = ones(Nkg)
    end
    k_weights ./= sum(k_weights)

    # (z,ω) distribution
    if zω_dist == :ergodic
        zω_weights = ObjGrid_Zω.ErgProb
    elseif zω_dist == :uniform
        zω_weights = ones(Nzω) / Nzω
    else
        zω_weights = ObjGrid_Zω.ErgProb
    end

    # Fill entry distribution on dense grid
    for ik in 1:Nkg
        for izω in 1:Nzω
            entry_dist[id_entry, ik, ix_entry, izω] = k_weights[ik] * zω_weights[izω]
        end
    end

    entry_dist ./= sum(entry_dist)

    return entry_dist
end


## Solve for stationary distribution
function solve_stationary_distribution(Sol_Firm, Param;
                                        max_iter=10000,
                                        tol=Param.tol_μ,
                                        verbose=true,
                                        include_entry=true)
    @unpack (ObjDGrid_D, ObjDGrid_K, ObjGrid_X, ObjGrid_Zω, χ) = Param
    Ndg, Nkg, Nx, Nzω = ObjDGrid_D.N, ObjDGrid_K.N, ObjGrid_X.N, ObjGrid_Zω.N

    # Initialize distribution (uniform) on dense grid
    μ = ones(Ndg, Nkg, Nx, Nzω)
    μ ./= sum(μ)

    # Interpolate coarse-grid policies onto dense grid once (outside the iteration loop)
    DGk, DGd, DGe = interpolate_policies_to_dense(Sol_Firm, Param)

    # Create entry distribution if needed
    if include_entry
        entry_dist = create_entry_distribution(Param)
    end

    # Iterate
    diff = Inf
    iter = 0
    t_total = @elapsed begin
        for i in 1:max_iter
            iter = i

            # Update distribution using pre-interpolated dense-grid policies
            μ_next = update_distribution(μ, DGk, DGd, DGe, Param; χ=χ)

            # Add entrants if specified
            if include_entry
                add_entrants!(μ_next, entry_dist, χ)
            end

            # Normalize (should already be normalized, but ensure numerical stability)
            μ_next ./= sum(μ_next)

            # Check convergence
            diff = maximum(abs.(μ_next - μ))

            if diff < tol
                μ = μ_next
                break
            end

            # Update for next iteration
            μ = μ_next

            # Print progress occasionally
            if verbose && (i % 500 == 0 || i == 1)
                println("  Distribution iteration $i: max diff = $(round(diff, sigdigits=4))")
            end
        end
    end

    if verbose
        println("\n---------------------------------------------------------------")
        println("  Distribution convergence")
        println("---------------------------------------------------------------")
        println("  Iterations:     $(iter)")
        println("  Final diff:     $(round(diff, sigdigits=6))")
        println("  Time (secs):    $(round(t_total, digits=3))")
        println("  Converged:      $(diff < tol ? "Yes" : "No")")
        println("---------------------------------------------------------------\n")
    end

    return (
        μ = μ,
        converged = diff < tol,
        iterations = iter,
        final_diff = diff
    )
end


## Compute aggregate statistics from distribution
function compute_aggregates(μ, Sol_Firm, Sol_Unconstrained, Param, Price_t)
    @unpack (ObjGrid_D, ObjGrid_K, ObjDGrid_D, ObjDGrid_K, ObjGrid_X, ObjGrid_Zω) = Param
    @unpack (α, δ, c_x) = Param
    # Coarse grids (for interpolating policies)
    Grid_D  = ObjGrid_D.Values
    Grid_K  = ObjGrid_K.Values
    # Dense grids (μ lives here)
    DGrid_D = ObjDGrid_D.Values
    DGrid_K = ObjDGrid_K.Values
    Grid_X  = ObjGrid_X.Values
    Grid_Zω = ObjGrid_Zω.Values
    Ndg, Nkg, Nx, Nzω = ObjDGrid_D.N, ObjDGrid_K.N, ObjGrid_X.N, ObjGrid_Zω.N
    pp = Sol_Unconstrained.Price_t.pp

    # Coarse-grid policy functions (interpolated below)
    Gk = Sol_Firm.Gk
    Ge = Sol_Firm.Ge
    Gd = Sol_Firm.Gd
    Is_Unconstrained = Sol_Firm.Is_Unconstrained

    # Initialize aggregates
    total_mass = 0.0
    agg_capital = 0.0
    agg_capital_next = 0.0
    agg_debt = 0.0
    agg_output = 0.0
    agg_engagement = 0.0
    agg_engagement_cost = 0.0
    frac_engaged = 0.0        # Fraction with x=1
    frac_engaging = 0.0       # Fraction choosing e=1
    frac_unconstrained = 0.0
    agg_cP = 0.0              # Political composite: Σ_{x=1} ω*z*k^α (weighted by mass)
    agg_cNP = 0.0             # Non-political composite: Σ_{x=0} z*k^α (weighted by mass)
    agg_investment = 0.0      # Aggregate gross investment: Σ (k′ - (1-δ)*k)
    agg_adj_cost = 0.0        # Aggregate capital adjustment costs: Σ kadjfnc(k′, k)

    # By firm size
    capital_dist = zeros(Nkg)

    for id in 1:Ndg
        d = DGrid_D[id]
        for ik in 1:Nkg
            k = DGrid_K[ik]
            for ix in 1:Nx
                x = Grid_X[ix]
                for izω in 1:Nzω
                    z, ω = Grid_Zω[:, izω]
                    mass = μ[id, ik, ix, izω]
                    mass < 1e-14 && continue

                    # Interpolate coarse-grid policies onto dense grid point (d, k)
                    k′       = linterp2(Grid_D, Grid_K, Gk[:, :, ix, izω], d, k)
                    e        = linterp2(Grid_D, Grid_K, Ge[:, :, ix, izω], d, k)
                    is_uncons = linterp2(Grid_D, Grid_K, Float64.(Is_Unconstrained[:, :, ix, izω]), d, k) > 0.5

                    # Output
                    price = (x == 0) ? 1.0 : ω * pp
                    output = price * z * k^α

                    # Accumulate
                    total_mass += mass
                    agg_capital += mass * k
                    agg_capital_next += mass * k′
                    agg_debt += mass * d
                    agg_output += mass * output
                    agg_engagement += mass * e
                    agg_engagement_cost += mass * c_x * e
                    frac_engaged += (x == 1) ? mass : 0.0
                    frac_engaging += (e ≈ 1.0) ? mass : 0.0
                    frac_unconstrained += is_uncons ? mass : 0.0
                    capital_dist[ik] += mass
                    # Political composite goods
                    if x == 1
                        agg_cP  += mass * ω * z * k^α   # Σ_{x=1} ω*z*k^α
                    else
                        agg_cNP += mass * z * k^α        # Σ_{x=0} z*k^α
                    end
                    # Investment and adjustment costs
                    agg_investment += mass * (k′ - (1.0 - δ) * k)
                    agg_adj_cost   += mass * kadjfnc(k′, k, Param)
                end
            end
        end
    end

    # Normalize by total mass
    agg_capital /= total_mass
    agg_capital_next /= total_mass
    agg_debt /= total_mass
    agg_output /= total_mass
    agg_engagement /= total_mass
    agg_engagement_cost /= total_mass
    frac_engaged /= total_mass
    frac_engaging /= total_mass
    frac_unconstrained /= total_mass
    capital_dist ./= total_mass
    agg_cP  /= total_mass
    agg_cNP /= total_mass
    agg_investment /= total_mass
    agg_adj_cost   /= total_mass

    # Total consumption
    agg_C = agg_cNP + pp * agg_cP

    return (
        total_mass = total_mass,
        agg_capital = agg_capital,
        agg_capital_next = agg_capital_next,
        agg_debt = agg_debt,
        agg_output = agg_output,
        agg_investment = agg_investment,
        agg_adj_cost = agg_adj_cost,
        agg_engagement = agg_engagement,
        agg_engagement_cost = agg_engagement_cost,
        frac_engaged = frac_engaged,
        frac_engaging = frac_engaging,
        frac_unconstrained = frac_unconstrained,
        capital_dist = capital_dist,
        investment_rate = (agg_capital_next - (1 - δ) * agg_capital) / agg_capital,
        agg_cP  = agg_cP,
        agg_cNP = agg_cNP,
        agg_C = agg_C
    )
end


## Print aggregate statistics
function print_aggregates(agg)
    println("\n===============================================================")
    println("                    AGGREGATE STATISTICS")
    println("===============================================================")
    println("  Total mass:              $(round(agg.total_mass, digits=4))")
    println("  Aggregate capital:       $(round(agg.agg_capital, digits=4))")
    println("  Aggregate debt:          $(round(agg.agg_debt, digits=4))")
    println("  Aggregate output:        $(round(agg.agg_output, digits=4))")
    println("  Aggregate investment:    $(round(agg.agg_investment, digits=4))")
    println("  Capital adj. costs:      $(round(agg.agg_adj_cost, digits=4))")
    println("---------------------------------------------------------------")
    println("  ENGAGEMENT")
    println("---------------------------------------------------------------")
    println("  Frac. w/ political cap:  $(round(100*agg.frac_engaged, digits=2))%")
    println("  Frac. choosing engage:   $(round(100*agg.frac_engaging, digits=2))%")
    println("  Avg. engagement rate:    $(round(100*agg.agg_engagement, digits=2))%")
    println("  Engagement costs/output: $(round(100*agg.agg_engagement_cost/agg.agg_output, digits=2))%")
    println("---------------------------------------------------------------")
    println("  FINANCIAL CONSTRAINTS")
    println("---------------------------------------------------------------")
    println("  Frac. unconstrained:     $(round(100*agg.frac_unconstrained, digits=2))%")
    println("  Debt/capital ratio:      $(round(agg.agg_debt/agg.agg_capital, digits=4))")
    println("  Investment rate:         $(round(100*agg.investment_rate, digits=2))%")
    println("---------------------------------------------------------------")
    println("  COMPOSITE GOODS")
    println("---------------------------------------------------------------")
    println("  Political (cP):          $(round(agg.agg_cP, digits=4))")
    println("  Non-political (cNP):     $(round(agg.agg_cNP, digits=4))")
    println("  Total consumption (C):   $(round(agg.agg_C, digits=4))")
    println("===============================================================\n")
end


## Compute engagement by firm size (capital deciles)
function engagement_by_size(μ, Sol_Firm, Param; n_bins=10)
    @unpack (ObjGrid_D, ObjGrid_K, ObjDGrid_D, ObjDGrid_K, ObjGrid_X, ObjGrid_Zω) = Param
    # Coarse grids for interpolation
    Grid_D  = ObjGrid_D.Values
    Grid_K  = ObjGrid_K.Values
    # Dense grids for distribution
    DGrid_K = ObjDGrid_K.Values
    Ndg, Nkg, Nx, Nzω = ObjDGrid_D.N, ObjDGrid_K.N, ObjGrid_X.N, ObjGrid_Zω.N

    Ge = Sol_Firm.Ge

    # Compute marginal distribution over k (on dense grid)
    k_dist       = zeros(Nkg)
    k_engagement = zeros(Nkg)

    for ik in 1:Nkg
        k = DGrid_K[ik]
        for id in 1:Ndg
            d = ObjDGrid_D.Values[id]
            for ix in 1:Nx
                for izω in 1:Nzω
                    mass = μ[id, ik, ix, izω]
                    k_dist[ik] += mass
                    # Interpolate engagement from coarse grid
                    e = linterp2(Grid_D, Grid_K, Ge[:, :, ix, izω], d, k)
                    k_engagement[ik] += mass * e
                end
            end
        end
    end

    # Average engagement by capital level
    avg_engagement_by_k = k_engagement ./ max.(k_dist, 1e-14)

    # Bin into deciles
    cumul_mass = cumsum(k_dist)
    cumul_mass ./= cumul_mass[end]

    bin_edges = range(0, 1, length=n_bins+1)
    bin_engagement = zeros(n_bins)
    bin_mass = zeros(n_bins)
    bin_avg_k = zeros(n_bins)

    for ik in 1:Nkg
        bin_idx = searchsortedfirst(bin_edges[2:end], cumul_mass[ik])
        bin_idx = min(bin_idx, n_bins)

        bin_engagement[bin_idx] += k_engagement[ik]
        bin_mass[bin_idx]       += k_dist[ik]
        bin_avg_k[bin_idx]      += k_dist[ik] * DGrid_K[ik]
    end

    # Compute averages
    bin_avg_engagement = bin_engagement ./ max.(bin_mass, 1e-14)
    bin_avg_k ./= max.(bin_mass, 1e-14)

    return (
        Grid_K = DGrid_K,
        k_dist = k_dist,
        avg_engagement_by_k = avg_engagement_by_k,
        bin_avg_k = bin_avg_k,
        bin_avg_engagement = bin_avg_engagement,
        bin_mass = bin_mass
    )
end
