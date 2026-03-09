# Tools_Constrained_Firms.jl: Tools for constrained firms
# Constrained firms have state variables (d, k, x, z, ω) and face a binding dividend constraint (d≥0)
# Uses joint (z,ω) grid for efficient expectation computation

using Base.Threads: @threads, nthreads

## First order condition for capital choice
# G(k′) = 0 defines the optimal capital policy
# Optimized version: pre-extract parameters to avoid repeated @unpack
@inline function GkFOC_fast(d′, k′, kval, ix, izω, λval, Gk, Ge_slice, Gλ,
                            Grid_D, Grid_K, Grid_X, Grid_Zω, Prob_Zω, Prob_X,
                            α, δ, χ, ψ_k, pp, SDF, Nx, Nzω)
    # Pre-compute common terms
    k′_αm1 = k′^(α - 1.0)
    one_minus_δ = 1.0 - δ
    one_minus_χ = 1.0 - χ

    # Compute expected marginal return to capital
    Expectation = 0.0
    @inbounds for jzω in 1:Nzω
        z′ = Grid_Zω[1, jzω]
        ω′ = Grid_Zω[2, jzω]
        prob_zω = Prob_Zω[izω, jzω]

        if Grid_X[ix] == 0
            # Future x′ depends on engagement choice today
            x′_val = Ge_slice[izω]   # Note: we pass a slice, so no id, ik, ix indices
            ix′ = (x′_val == 0.0) ? 1 : 2
            x′ = Grid_X[ix′]
            kstar′′ = linterp2(Grid_D, Grid_K, @view(Gk[:, :, ix′, jzω]), d′, k′)
            Gλ_val = linterp2(Grid_D, Grid_K, @view(Gλ[:, :, ix′, jzω]), d′, k′)
            adjλ = 1.0 + one_minus_χ * Gλ_val
            # Inline punit: x′==0 ? 1.0 : ω′*pp
            price = (x′ == 0) ? 1.0 : ω′ * pp
            # Inline dk_kadjfnc
            dk_adj = ψ_k / 2.0 * ((kstar′′ / k′) - one_minus_δ) * (-(kstar′′ / k′) - one_minus_δ)
            Expectation += prob_zω * (adjλ * (price * z′ * α * k′_αm1 + one_minus_δ) - one_minus_χ * dk_adj)
        else
            # No engagement choice today, so future x′ depends on depreciation probability
            for jx′ in 1:Nx
                x′ = Grid_X[jx′]
                prob_x = Prob_X[jx′]
                kstar′′ = linterp2(Grid_D, Grid_K, @view(Gk[:, :, jx′, jzω]), d′, k′)
                Gλ_val = linterp2(Grid_D, Grid_K, @view(Gλ[:, :, jx′, jzω]), d′, k′)
                adjλ = 1.0 + one_minus_χ * Gλ_val
                price = (x′ == 0) ? 1.0 : ω′ * pp
                dk_adj = ψ_k / 2.0 * ((kstar′′ / k′) - one_minus_δ) * (-(kstar′′ / k′) - one_minus_δ)
                Expectation += prob_x * prob_zω * (adjλ * (price * z′ * α * k′_αm1 + one_minus_δ) - one_minus_χ * dk_adj)
            end
        end
    end

    # Objective for policy function iteration: G(k′) = 0
    # Inline dk′_kadjfnc
    dk′_adj = ψ_k * ((k′ / kval) - one_minus_δ)
    G = (1.0 + dk′_adj) * (1.0 + λval) - SDF * Expectation

    return G
end


## Policy function iteration for capital - optimized with parallelization
function solve_k(Gd, Ge, Gλ, Price_t, Param, Gk_guess=nothing)
    @unpack (ObjGrid_D, ObjGrid_K, ObjGrid_X, ObjGrid_Zω) = Param
    @unpack (α, δ, χ, ψ_k) = Param
    Nd, Nk, Nx, Nzω = ObjGrid_D.N, ObjGrid_K.N, ObjGrid_X.N, ObjGrid_Zω.N
    Grid_D = ObjGrid_D.Values
    Grid_K = ObjGrid_K.Values
    Grid_X = ObjGrid_X.Values
    Grid_Zω = ObjGrid_Zω.Values
    Prob_Zω = ObjGrid_Zω.Prob
    Prob_X = ObjGrid_X.Prob
    pp = Price_t.pp
    SDF = Price_t.SDF

    # Initializing
    Gk = zeros(Nd, Nk, Nx, Nzω)
    if isnothing(Gk_guess)
        # First iteration: use default guess
        Gk .= (1.0 - δ) .* reshape(Grid_K, 1, Nk, 1, 1)
    else
        # Use previous iteration's solution as starting guess
        Gk .= Gk_guess
    end
    K_old = copy(Gk)
    K_new = copy(Gk)
    max_diff = 0.0
    niter = 0

    for iter in 1:1000
        niter = iter

        # Parallelize over the outermost loop (id)
        @threads for id in 1:Nd
            for ik in 1:Nk
                kval = Grid_K[ik]
                for ix in 1:Nx
                    # Extract slice for this (id, ik, ix) to avoid repeated indexing
                    Ge_slice = @view Ge[id, ik, ix, :]

                    for izω in 1:Nzω
                        d′ = Gd[id, ik, ix, izω]
                        λval = Gλ[id, ik, ix, izω]

                        # Create closure with pre-extracted parameters
                        f = k′ -> GkFOC_fast(d′, k′, kval, ix, izω, λval, K_old, Ge_slice, Gλ,
                                             Grid_D, Grid_K, Grid_X, Grid_Zω, Prob_Zω, Prob_X,
                                             α, δ, χ, ψ_k, pp, SDF, Nx, Nzω)

                        # Use previous solution as better bracket
                        k_prev = K_old[id, ik, ix, izω]
                        bracket_lb = max(0.01, 0.7 * k_prev)
                        bracket_ub = min(1000.0, 1.5 * k_prev + 0.1)

                        k′_star = find_zero(f, (bracket_lb, bracket_ub), Bisection())
                        K_new[id, ik, ix, izω] = k′_star
                    end
                end
            end
        end

        # Check convergence of policy function iteration
        max_diff = maximum(abs.(K_new - K_old))
        if max_diff < Param.tol_v / 100
            break
        end
        K_old .= K_new
    end
    Gk .= K_new

    return Gk, niter, max_diff
end


## Update Gd from the binding dividend constraint - optimized with parallelization
# When the dividend constraint binds (div = 0), debt policy is pinned down by:
#   d′ = R * (π + (1-δ)*k - k′ - adj(k′,k) - d)
function update_Gd(Gd, Gk, Ge, Price_t, Param)
    @unpack (ObjGrid_D, ObjGrid_K, ObjGrid_X, ObjGrid_Zω) = Param
    @unpack (α, δ, c_x, ψ_k) = Param
    Grid_D = ObjGrid_D.Values
    Grid_K = ObjGrid_K.Values
    Grid_X = ObjGrid_X.Values
    Grid_Zω = ObjGrid_Zω.Values
    Nd, Nk, Nx, Nzω = ObjGrid_D.N, ObjGrid_K.N, ObjGrid_X.N, ObjGrid_Zω.N
    pp = Price_t.pp
    R = Price_t.R
    one_minus_δ = 1.0 - δ

    # Initializing
    Gd_new = similar(Gd)

    # Parallelize over outermost loop
    @threads for id in 1:Nd
        d = Grid_D[id]
        @inbounds for ik in 1:Nk
            k = Grid_K[ik]
            for ix in 1:Nx
                x = Grid_X[ix]
                for izω in 1:Nzω
                    z = Grid_Zω[1, izω]
                    ω = Grid_Zω[2, izω]
                    # Update Gd based on the current policy functions
                    eval = Ge[id, ik, ix, izω]
                    k′ = Gk[id, ik, ix, izω]
                    # Inline profit: punit(pp, x, ω) * z * k^α - c_x * e
                    price = (x == 0) ? 1.0 : ω * pp
                    π_val = price * z * k^α - c_x * eval
                    # Inline kadjfnc: ψ_k/2 * ((k′/k) - (1-δ))^2 * k
                    adj = ψ_k / 2.0 * ((k′ / k) - one_minus_δ)^2 * k
                    # From binding dividend constraint: div = π + (1-δ)k - d - k′ - adj(k′,k) + d′/R = 0
                    # Solving for d′: d′ = R * (k′ + adj(k′,k) + d - π - (1-δ)k)
                    Gd_new[id, ik, ix, izω] = R * (k′ + adj + d - π_val - one_minus_δ * k)
                end
            end
        end
    end

    return Gd_new
end


## Update multiplier on dividend constraint - optimized with parallelization
# From the Euler equation for debt: 1 + λ = R * SDF * E[(1 + (1-χ)*λ′)]
function update_Gλ(Gd, Gk, Ge, Gλ, Price_t, Param)
    @unpack (ObjGrid_D, ObjGrid_K, ObjGrid_X, ObjGrid_Zω) = Param
    @unpack (χ) = Param
    Grid_D = ObjGrid_D.Values
    Grid_K = ObjGrid_K.Values
    Grid_X = ObjGrid_X.Values
    Grid_Zω = ObjGrid_Zω.Values
    Prob_Zω = ObjGrid_Zω.Prob
    Prob_X = ObjGrid_X.Prob
    Nd, Nk, Nx, Nzω = ObjGrid_D.N, ObjGrid_K.N, ObjGrid_X.N, ObjGrid_Zω.N
    R_SDF = Price_t.R * Price_t.SDF
    one_minus_χ = 1.0 - χ

    # Initialize
    Gλ_new = similar(Gλ)

    # Parallelize over outermost loop
    @threads for id in 1:Nd
        @inbounds for ik in 1:Nk
            for ix in 1:Nx
                for izω in 1:Nzω
                    # Update Gλ based on the current policy functions
                    Expectation = 0.0
                    d′ = Gd[id, ik, ix, izω]
                    k′ = Gk[id, ik, ix, izω]

                    for jzω in 1:Nzω
                        prob_zω = Prob_Zω[izω, jzω]

                        if Grid_X[ix] == 0
                            # Future x′ depends on engagement choice today
                            x′_val = Ge[id, ik, ix, izω]
                            ix′ = (x′_val == 0.0) ? 1 : 2
                            λ′_val = linterp2(Grid_D, Grid_K, @view(Gλ[:, :, ix′, jzω]), d′, k′)
                            temp = 1.0 + one_minus_χ * λ′_val
                            Expectation += prob_zω * temp
                        else
                            # x=1: stochastic depreciation
                            for jx′ in 1:Nx
                                prob_x = Prob_X[jx′]
                                λ′_val = linterp2(Grid_D, Grid_K, @view(Gλ[:, :, jx′, jzω]), d′, k′)
                                temp = 1.0 + one_minus_χ * λ′_val
                                Expectation += prob_x * prob_zω * temp
                            end
                        end
                    end
                    # Euler equation: 1 + λ = R * SDF * E[1 + (1-χ)*λ′]
                    # So: λ = R * SDF * E[1 + (1-χ)*λ′] - 1
                    Gλ_new[id, ik, ix, izω] = R_SDF * Expectation - 1.0
                end
            end
        end
    end

    return Gλ_new
end


## Continuation value given current state for constrained firm with x=0 - optimized
@inline function value_cons_vcont_x0_fast(izω, k′, d′, SDF, pp, V_Constrained,
                                          Grid_D, Grid_K, Grid_Zω, Prob_Zω,
                                          α, δ, χ, Nzω)
    one_minus_δ = 1.0 - δ
    one_minus_χ = 1.0 - χ
    k′_α = k′^α

    # Initializing
    vcont_n = 0.0
    vcont_p = 0.0
    @inbounds for jzω in 1:Nzω
        z′ = Grid_Zω[1, jzω]
        ω′ = Grid_Zω[2, jzω]
        prob_zω = Prob_Zω[izω, jzω]

        # No engagement, so e=0 and x′=0
        # x′=0 → punit=1.0
        Enw_n = z′ * k′_α + one_minus_δ * k′ - d′
        v_n = linterp2(Grid_D, Grid_K, @view(V_Constrained[:, :, 1, jzω]), d′, k′)
        vcont_n += SDF * prob_zω * (one_minus_χ * v_n + χ * Enw_n)

        # Engagement, so e=1 and x′=1
        # x′=1 → punit=ω′*pp
        Enw_p = ω′ * pp * z′ * k′_α + one_minus_δ * k′ - d′
        v_p = linterp2(Grid_D, Grid_K, @view(V_Constrained[:, :, 2, jzω]), d′, k′)
        vcont_p += SDF * prob_zω * (one_minus_χ * v_p + χ * Enw_p)
    end

    return vcont_n, vcont_p
end


## Continuation value given current state for constrained firm with x=1 - optimized
@inline function value_cons_vcont_x1_fast(izω, k′, d′, SDF, pp, V_Constrained,
                                          Grid_D, Grid_K, Grid_X, Grid_Zω, Prob_Zω, Prob_X,
                                          α, δ, χ, Nx, Nzω)
    one_minus_δ = 1.0 - δ
    one_minus_χ = 1.0 - χ
    k′_α = k′^α

    # Initializing
    vcont = 0.0
    @inbounds for jzω in 1:Nzω
        z′ = Grid_Zω[1, jzω]
        ω′ = Grid_Zω[2, jzω]
        prob_zω = Prob_Zω[izω, jzω]

        # Expectation over x' given current x=1
        term_x = 0.0
        for jx′ in 1:Nx
            x′ = Grid_X[jx′]
            prob_x = Prob_X[jx′]  # Conditional probability P(x' | x=1)
            # Inline punit
            price = (x′ == 0) ? 1.0 : ω′ * pp
            Enw = price * z′ * k′_α + one_minus_δ * k′ - d′
            v′ = linterp2(Grid_D, Grid_K, @view(V_Constrained[:, :, jx′, jzω]), d′, k′)
            term_x += prob_x * (one_minus_χ * v′ + χ * Enw)
        end

        vcont += SDF * prob_zω * term_x
    end

    return vcont
end


## Update value function given policy functions for constrained firm - optimized with parallelization
function value_constrained(V_Constrained, Gk, Gd, Ge, Price_t, Price_tf, Param)
    @unpack (ObjGrid_D, ObjGrid_K, ObjGrid_X, ObjGrid_Zω) = Param
    @unpack (α, δ, χ, c_x, ψ_k) = Param
    Grid_D = ObjGrid_D.Values
    Grid_K = ObjGrid_K.Values
    Grid_X = ObjGrid_X.Values
    Grid_Zω = ObjGrid_Zω.Values
    Prob_Zω = ObjGrid_Zω.Prob
    Prob_X = ObjGrid_X.Prob
    Nd, Nk, Nx, Nzω = ObjGrid_D.N, ObjGrid_K.N, ObjGrid_X.N, ObjGrid_Zω.N
    pp = Price_t.pp
    SDF = Price_t.SDF
    R_inv = 1.0 / Price_t.R
    one_minus_δ = 1.0 - δ

    # Allocating
    TV_Constrained = zeros(Nd, Nk, Nx, Nzω)
    TE_Constrained = zeros(Nd, Nk, Nx, Nzω)

    # Parallelize over outermost loop
    @threads for id in 1:Nd
        d = Grid_D[id]
        @inbounds for ik in 1:Nk
            k = Grid_K[ik]
            for izω in 1:Nzω
                z = Grid_Zω[1, izω]
                ω = Grid_Zω[2, izω]
                for ix in 1:Nx
                    x = Grid_X[ix]
                    # Engagement choice (from previous iteration)
                    eval = Ge[id, ik, ix, izω]
                    # Capital choice
                    k′ = Gk[id, ik, ix, izω]
                    # Debt choice
                    d′ = Gd[id, ik, ix, izω]

                    # Net worth (excluding engagement cost which is considered below)
                    # Inline punit
                    price = (x == 0) ? 1.0 : ω * pp
                    nw = price * z * k^α + one_minus_δ * k - d

                    # Inline kadjfnc
                    adj = ψ_k / 2.0 * ((k′ / k) - one_minus_δ)^2 * k

                    # Current period dividend (base, before engagement cost)
                    div_base = nw - k′ - adj + R_inv * d′

                    # Continuation value and engagement update
                    if x == 0
                        # Counterfactual continuations from same current state (x=0)
                        vcont_n, vcont_p = value_cons_vcont_x0_fast(izω, k′, d′, SDF, pp, V_Constrained,
                                                                     Grid_D, Grid_K, Grid_Zω, Prob_Zω,
                                                                     α, δ, χ, Nzω)

                        # Value associated with each engagement choice
                        𝓥_n = div_base + vcont_n
                        𝓥_p = div_base - c_x + vcont_p

                        # Engagement choice update with hysteresis to prevent oscillation
                        hysteresis_tol = Param.tol_v * 100
                        if eval ≈ 0.0
                            # Currently not engaging: only start if 𝓥_p > 𝓥_n + tol
                            TE_Constrained[id, ik, 1, izω] = (𝓥_p > 𝓥_n + hysteresis_tol) ? 1.0 : 0.0
                        else
                            # Currently engaging: only stop if 𝓥_n > 𝓥_p + tol
                            TE_Constrained[id, ik, 1, izω] = (𝓥_n > 𝓥_p + hysteresis_tol) ? 0.0 : 1.0
                        end

                        # Use current eval (from previous iteration) to compute the actual value this iteration
                        vnow = div_base - c_x * eval
                        vcont = (eval ≈ 1.0) ? vcont_p : vcont_n
                    else
                        # x=1: stochastic depreciation, no engagement choice
                        vcont = value_cons_vcont_x1_fast(izω, k′, d′, SDF, pp, V_Constrained,
                                                          Grid_D, Grid_K, Grid_X, Grid_Zω, Prob_Zω, Prob_X,
                                                          α, δ, χ, Nx, Nzω)
                        vnow = div_base
                        # No engagement update for x=1 (engagement already established to be 0.0)
                        # TE_Constrained[id, ik, ix, izω] = 0.0 # Not needed, so we can comment this line
                    end

                    # Update value function for constrained firm
                    TV_Constrained[id, ik, ix, izω] = vnow + vcont
                end
            end
        end
    end

    return TV_Constrained, TE_Constrained
end


## Main solver: Iterate to solve the constrained firm's problem
function solve_constrained_firm(Param; Price_t = (R=1.04, SDF=1.0/1.04, pp=1.25), max_iter = 10, max_viter = 500, verbose = true)
    @unpack (ObjGrid_D, ObjGrid_K, ObjGrid_X, ObjGrid_Zω) = Param
    @unpack (δ) = Param
    Nd, Nk, Nx, Nzω = ObjGrid_D.N, ObjGrid_K.N, ObjGrid_X.N, ObjGrid_Zω.N

    # Initialize policy functions and value function
    V_Constrained  = zeros(Nd, Nk, Nx, Nzω)
    Ge  = zeros(Nd, Nk, Nx, Nzω)           # Engagement policy
    Gλ  = zeros(Nd, Nk, Nx, Nzω)           # Multiplier on dividend constraint
    Gk  = zeros(Nd, Nk, Nx, Nzω)           # Capital policy
    Gk .= (1.0 - δ) .* reshape(ObjGrid_K.Values, 1, Nk, 1, 1)  # Initial guess
    Gd  = zeros(Nd, Nk, Nx, Nzω)           # Debt policy

    TE_Constrained = copy(Ge)

    for iter in 1:max_iter
        t_iter = @elapsed begin
            # Step 1: Solve capital policy k′ given (Gd, Ge, Gλ)
            Gk_guess = (iter == 1) ? nothing : Gk
            t_kstar  = @elapsed Gk, niter_k, diff_k = solve_k(Gd, Ge, Gλ, Price_t, Param, Gk_guess)

            # Step 2: Update debt policy Gd from binding dividend constraint
            t_gd   = @elapsed Gd_new = update_Gd(Gd, Gk, Ge, Price_t, Param)
            diff_d = maximum(abs.(Gd_new - Gd))
            Gd    .= Gd_new

            # Step 3: Update multiplier Gλ
            t_gλ   = @elapsed Gλ_new = update_Gλ(Gd, Gk, Ge, Gλ, Price_t, Param)
            diff_λ = maximum(abs.(Gλ_new - Gλ))
            Gλ    .= Gλ_new

            # Step 4: Update value function and engagement policy (value iteration)
            for viter in 1:max_viter
                TV_Constrained, TE_Constrained = value_constrained(V_Constrained, Gk, Gd, Ge, Price_t, Price_t, Param)
                vdiff = maximum(abs.(TV_Constrained .- V_Constrained))
                # Dampening for stability
                V_Constrained .= 0.9 .* V_Constrained .+ 0.1 .* TV_Constrained
                if vdiff < Param.tol_v
                    break
                end
            end
        end

        # Check convergence of engagement policy
        avg_diff = mean(abs.(Ge - TE_Constrained))

        # Print progress table
        if verbose
            println("\n---------------------------------------------------------------")
            println("  Constrained step      Iters        Max Diff             Secs    ")
            println("---------------------------------------------------------------")
            println("  Engagement (Avg)    $(lpad(iter, 7))  $(lpad(round(avg_diff, sigdigits=6), 14))  $(lpad(round(t_iter, digits=3), 15)) ")
            println("  K* (Capital)        $(lpad(niter_k, 7))  $(lpad(round(diff_k, sigdigits=6), 14))  $(lpad(round(t_kstar, digits=3), 15)) ")
            println("  Gd (Debt)                 -  $(lpad(round(diff_d, sigdigits=6), 14))  $(lpad(round(t_gd, digits=3), 15)) ")
            println("  Gλ (Multiplier)           -  $(lpad(round(diff_λ, sigdigits=6), 14))  $(lpad(round(t_gλ, digits=3), 15)) ")
            println("---------------------------------------------------------------\n")
        end

        if avg_diff < Param.tol_v * 100
            if verbose
                println("✓ Constrained problem converged after ", iter, " iterations.")
            end
            break
        end

        # Update engagement policy for next iteration
        Ge .= TE_Constrained
    end

    return (
        Price_t         = Price_t,
        V_Constrained   = V_Constrained,
        E_Constrained   = Ge,
        TE_Constrained  = TE_Constrained,
        K_Constrained   = Gk,
        D_Constrained   = Gd,
        λ_Constrained   = Gλ
    )
end
