# Tools_Unconstrained_Firms.jl: Tools for unconstrained firms
# Uses joint (z,ω) grid for efficient expectation computation

using Base.Threads: @threads, nthreads

## Solve problem of unconstrained firm

# Determine efficient scale of operation
# We do policy function iteration. Define function G(k′)
# Optimized version: pre-extract parameters to avoid repeated @unpack
@inline function Gkstar_fast(k′, kval, ix, izω, K_Unconstrained, E_Unconstrained,
                             Grid_K, Grid_X, Grid_Zω, Prob_Zω, Prob_X,
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
            x′_val = E_Unconstrained[ix, izω]   # Note: we pass a slice, so no ik index
            ix′ = (x′_val == 0.0) ? 1 : 2
            x′ = Grid_X[ix′]
            kstar′′ = linterp1(Grid_K, @view(K_Unconstrained[:, ix′, jzω]), k′)
            # Inline punit: x′==0 ? 1.0 : ω′*pp
            price = (x′ == 0) ? 1.0 : ω′ * pp
            # Inline dk_kadjfnc(kstar′′, k′, Param) - derivative w.r.t. current k
            dk_adj = ψ_k / 2.0 * ((kstar′′ / k′) - one_minus_δ) * (-(kstar′′ / k′) - one_minus_δ)
            Expectation += prob_zω * (price * z′ * α * k′_αm1 + one_minus_δ - one_minus_χ * dk_adj)
        else
            # No engagement choice today, so future x′ depends on depreciation probability
            for ix′ in 1:Nx
                x′ = Grid_X[ix′]
                prob_x = Prob_X[ix′]
                kstar′′ = linterp1(Grid_K, @view(K_Unconstrained[:, ix′, jzω]), k′)
                price = (x′ == 0) ? 1.0 : ω′ * pp
                dk_adj = ψ_k / 2.0 * ((kstar′′ / k′) - one_minus_δ) * (-(kstar′′ / k′) - one_minus_δ)
                Expectation += prob_x * prob_zω * (price * z′ * α * k′_αm1 + one_minus_δ - one_minus_χ * dk_adj)
            end
        end
    end

    # Objective for policy function iteration: G(k′) = 0
    # Inline dk′_kadjfnc
    dk′_adj = ψ_k * ((k′ / kval) - one_minus_δ)
    G = 1.0 + dk′_adj - SDF * Expectation
    return G
end

# Policy function iteration - optimized with parallelization
function solve_kstar(E_Unconstrained, Price_t, Param, K_guess=nothing)
    @unpack (ObjGrid_K, ObjGrid_X, ObjGrid_Zω) = Param
    @unpack (α, δ, χ, ψ_k) = Param
    Nk, Nx, Nzω = ObjGrid_K.N, ObjGrid_X.N, ObjGrid_Zω.N
    Grid_K = ObjGrid_K.Values
    Grid_X = ObjGrid_X.Values
    Grid_Zω = ObjGrid_Zω.Values
    Prob_Zω = ObjGrid_Zω.Prob
    Prob_X = ObjGrid_X.Prob
    pp = Price_t.pp
    SDF = Price_t.SDF

    # Initializing
    K_Unconstrained = zeros(Nk, Nx, Nzω)
    if isnothing(K_guess)
        K_Unconstrained[:, :, :] .= (1.0 - δ) * Grid_K
    else
        K_Unconstrained .= K_guess
    end
    K_old = copy(K_Unconstrained)
    K_new = copy(K_Unconstrained)
    max_diff = 0.0
    niter = 0

    for iter in 1:1000
        niter = iter

        # Parallelize over the outermost loop (ik)
        @threads for ik in 1:Nk
            kval = Grid_K[ik]

            for ix in 1:Nx
                # Extract slice for this (ik, ix) to avoid repeated indexing
                E_slice = @view E_Unconstrained[ik, ix, :]

                for izω in 1:Nzω
                    # Create closure with pre-extracted parameters
                    f = k′ -> Gkstar_fast(k′, kval, ix, izω, K_old, E_slice,
                                          Grid_K, Grid_X, Grid_Zω, Prob_Zω, Prob_X,
                                          α, δ, χ, ψ_k, pp, SDF, Nx, Nzω)

                    # Use previous solution as better bracket
                    k_prev = K_old[ik, ix, izω]
                    bracket_lb = max(0.01, 0.7 * k_prev)
                    bracket_ub = min(1000.0, 1.5 * k_prev + 0.1)

                    k′_star = find_zero(f, (bracket_lb, bracket_ub), Bisection())
                    K_new[ik, ix, izω] = k′_star
                end
            end
        end

        # Check convergence
        max_diff = maximum(abs.(K_new - K_old))
        if max_diff < 1e-4
            break
        end
        K_old .= K_new
    end
    K_Unconstrained .= K_new

    return K_Unconstrained, niter, max_diff
end


## Minimum savings policy
function update_msp(Dmsp, K_Unconstrained, E_Unconstrained, Price_tf, Param)
    @unpack (ObjGrid_K, ObjGrid_X, ObjGrid_Zω) = Param
    @unpack (α, δ, c_x) = Param
    Grid_K = ObjGrid_K.Values
    Grid_X = ObjGrid_X.Values
    Grid_Zω = ObjGrid_Zω.Values
    Nk, Nx, Nzω = ObjGrid_K.N, ObjGrid_X.N, ObjGrid_Zω.N

    TDmsp = similar(Dmsp)
    for ik in 1:Nk
        k = Grid_K[ik]
        for ix in 1:Nx
            x = Grid_X[ix]
            for izω in 1:Nzω
                z, ω = Grid_Zω[:, izω]
                # Capital at beginning of next period
                k′ = K_Unconstrained[ik, ix, izω]
                Dmsp_temp = zeros(Nzω)

                for jzω in 1:Nzω
                    z′, ω′ = Grid_Zω[:, jzω]
                    prob_zω = ObjGrid_Zω.Prob[izω, jzω]

                    # Compute flow profits for every realization of the future state
                    if Grid_X[ix] == 0
                        # Future x′ depends on engagement choice today
                        x′_val  = E_Unconstrained[ik, ix, izω]   # must be 0 or 1
                        ix′     = (x′_val == 0) ? 1 : 2
                        x′      = Grid_X[ix′]
                        kstar′′ = linterp1(Grid_K, K_Unconstrained[:, ix′, jzω], k′)
                        eval′   = 1.0 # Worst case scenario
                        dstar′′ = linterp1(Grid_K, Dmsp[:, ix′, jzω], k′)
                        πval′   = profit(Price_tf.pp, x′, ω′, z′, k′, eval′; Param)
                        dtilde  = πval′ + (1.0 - δ) * k′ - kstar′′ - kadjfnc(kstar′′, k′, Param) + (1.0/Price_tf.R) * dstar′′
                    else
                        # No engagement choice today, so future x′ depends on depreciation probability
                        dtildex  = fill(Inf, ObjGrid_X.N)
                        for ix′ in 1:ObjGrid_X.N
                            x′      = Grid_X[ix′]
                            prob_x  = ObjGrid_X.Prob[ix′]
                            prob_x  < 1e-6 && continue  # Skip states with negligible probability
                            kstar′′ = linterp1(Grid_K, K_Unconstrained[:, ix′, jzω], k′)
                            eval′   = (x′ == 0) ? 1.0 : 0.0    # Worst case scenario
                            dstar′′ = linterp1(Grid_K, Dmsp[:, ix′, jzω], k′)
                            πval′   = profit(Price_tf.pp, x′, ω′, z′, k′, eval′; Param)
                            dtildex[ix′] = πval′ + (1.0 - δ) * k′ - kstar′′ - kadjfnc(kstar′′, k′, Param) + (1.0/Price_tf.R) * dstar′′
                        end
                        dtilde = minimum(dtildex)
                    end
                    # If the state is unreachable, set to large value
                    dtilde = (prob_zω < 1e-6) ? Inf : dtilde
                    Dmsp_temp[jzω] = dtilde
                end
                TDmsp[ik, ix, izω] = minimum(Dmsp_temp)
            end
        end
    end
    return TDmsp
end

function solve_msp(K_Unconstrained, E_Unconstrained, Price_tf, Param, Dmsp)
    # Iterate on minimum savings policy until convergence
    max_diff = 0.0
    niter = 0
    for iter in 1:1000
        niter = iter
        # Update minimum savings policy given current policy functions for unconstrained firm
        TDmsp = update_msp(Dmsp, K_Unconstrained, E_Unconstrained, Price_tf, Param)
        # Check convergence of minimum savings policy
        max_diff = maximum(abs.(Dmsp - TDmsp))
        if max_diff < 1e-4
            break
        end
        Dmsp .= TDmsp
    end
    return Dmsp, niter, max_diff
end


## Update value function given optimal policy for unconstrained firm
# Note that the value for an unconstrained firm is linear/separable in current debt
# We can ignore current debt for these firms:
# We only need it for updating the engagement policy, so 𝓥_n and 𝓥_p are being affected linearly by the same d
# So we evaluate v* at d=0 for simplicity
function value_uncons(Dmsp, V_Unconstrained, K_Unconstrained, E_Unconstrained, Price_t, Price_tf, Param)
    @unpack (ObjGrid_K, ObjGrid_X, ObjGrid_Zω) = Param
    @unpack (α, δ, χ, c_x) = Param
    Grid_K = ObjGrid_K.Values
    Grid_X = ObjGrid_X.Values
    Grid_Zω = ObjGrid_Zω.Values
    Nk, Nx, Nzω = ObjGrid_K.N, ObjGrid_X.N, ObjGrid_Zω.N

    # Allocating
    TV_Unconstrained = zeros(Nk, Nx, Nzω)
    TE_Unconstrained = zeros(Nk, Nx, Nzω)

    for ik in 1:Nk
        k = Grid_K[ik]
        for izω in 1:Nzω
            z, ω = Grid_Zω[:, izω]
            for ix in 1:Nx
                x = Grid_X[ix]
                # Engagement choice
                eval = E_Unconstrained[ik, ix, izω]
                # Capital choice
                k′ = K_Unconstrained[ik, ix, izω]
                # Minimum savings policy
                d′ = Dmsp[ik, ix, izω]
                # Net worth, ex-engagement cost (will be considered below when comparing 𝓥_n and 𝓥_p)
                nw = punit(Price_t.pp, x, ω) * z * k^α + (1.0 - δ) * k
                # Current value
                vnow_base = nw - k′ - kadjfnc(k′, k, Param) + (1.0/Price_t.R) * d′

                # Continuation value
                if Grid_X[ix] == 0
                    # Counterfactual continuations from same current state (x=0)
                    vcont_n, vcont_p = value_uncons_vcont_x0(ik, izω, k′, d′, Price_t, Price_tf, V_Unconstrained, Param)
                    # Value associated with each engagement choice
                    𝓥_n = vnow_base + vcont_n
                    𝓥_p = vnow_base + vcont_p - c_x
                    # Engagement choice with hysteresis to prevent oscillation
                    # Only switch policy if benefit exceeds threshold
                    hysteresis_tol = 1e-4
                    if eval ≈ 0.0
                        # Currently not engaging: only start if 𝓥_p > 𝓥_n + tol
                        TE_Unconstrained[ik, 1, izω] = (𝓥_p > 𝓥_n + hysteresis_tol) ? 1.0 : 0.0
                    else
                        # Currently engaging: only stop if 𝓥_n > 𝓥_p + tol
                        TE_Unconstrained[ik, 1, izω] = (𝓥_n > 𝓥_p + hysteresis_tol) ? 0.0 : 1.0
                    end
                    # Use current eval (from previous iteration) to compute the actual value this iteration
                    vnow  = vnow_base - c_x * eval
                    vcont = (eval ≈ 1.0) ? vcont_p : vcont_n
                else
                    # x=1: stochastic depreciation
                    vcont = value_uncons_vcont_x1(ik, izω, k′, d′, Price_t, Price_tf, V_Unconstrained, Param)
                    vnow  = vnow_base
                end
                # Update value function for unconstrained firm
                TV_Unconstrained[ik, ix, izω] = vnow + vcont
            end
        end
    end

    return TV_Unconstrained, TE_Unconstrained
end

# Continuation value given current state (ik, izω) for unconstrained firm with x=0
function value_uncons_vcont_x0(ik, izω, k′, d′, Price_t, Price_tf, V_Unconstrained, Param)
    @unpack (ObjGrid_K, ObjGrid_X, ObjGrid_Zω) = Param
    @unpack (α, δ, χ) = Param
    Grid_K = ObjGrid_K.Values
    Grid_X = ObjGrid_X.Values
    Grid_Zω = ObjGrid_Zω.Values
    Nzω = ObjGrid_Zω.N

    # Initializing
    vcont_n = 0.0
    vcont_p = 0.0
    for jzω in 1:Nzω
        z′, ω′ = Grid_Zω[:, jzω]
        prob_zω = ObjGrid_Zω.Prob[izω, jzω]

        # No engagement, so e=0 and x′=0
        ix′      = 1
        x′       = Grid_X[ix′]
        Enw_n    = punit(Price_tf.pp, x′, ω′) * z′ * k′^α + (1.0 - δ) * k′ - d′
        v_n      = linterp1(Grid_K, V_Unconstrained[:, ix′, jzω], k′)
        vcont_n += Price_t.SDF * prob_zω * ((1.0 - χ) * v_n + χ * Enw_n)

        # Engagement, so e=1 and x′=1
        ix′      = 2
        x′       = Grid_X[ix′]
        Enw_p    = punit(Price_tf.pp, x′, ω′) * z′ * k′^α + (1.0 - δ) * k′ - d′
        v_p      = linterp1(Grid_K, V_Unconstrained[:, ix′, jzω], k′)
        vcont_p += Price_t.SDF * prob_zω * ((1.0 - χ) * v_p + χ * Enw_p)
    end

    return vcont_n, vcont_p
end

# Continuation value given current state (ik, izω) for unconstrained firm with x=1
function value_uncons_vcont_x1(ik, izω, k′, d′, Price_t, Price_tf, V_Unconstrained, Param)
    @unpack (ObjGrid_K, ObjGrid_X, ObjGrid_Zω) = Param
    @unpack (α, δ, χ, δ_x) = Param
    Grid_K = ObjGrid_K.Values
    Grid_X = ObjGrid_X.Values
    Grid_Zω = ObjGrid_Zω.Values
    Nzω = ObjGrid_Zω.N

    # Initializing
    vcont = 0.0
    for jzω in 1:Nzω
        z′, ω′ = Grid_Zω[:, jzω]
        prob_zω = ObjGrid_Zω.Prob[izω, jzω]

        # Expectation over x' given current x=1
        # ObjGrid_X.Prob contains transition probabilities FROM x=1:
        #   ObjGrid_X.Prob[1] = δ_x    → P(x'=0 | x=1) [depreciation]
        #   ObjGrid_X.Prob[2] = 1-δ_x  → P(x'=1 | x=1) [persistence]
        term_x = 0.0
        for ix′ in 1:ObjGrid_X.N
            x′      = Grid_X[ix′]
            prob_x  = ObjGrid_X.Prob[ix′]  # Conditional probability P(x' | x=1)
            Enw     = punit(Price_tf.pp, x′, ω′) * z′ * k′^α + (1.0 - δ) * k′ - d′
            v′      = linterp1(Grid_K, V_Unconstrained[:, ix′, jzω], k′)
            term_x += prob_x * ((1.0 - χ) * v′ + χ * Enw)
        end

        vcont += Price_t.SDF * prob_zω * term_x
    end

    return vcont
end


## Get unconstrained threshold
function get_unconstrained_threshold(K_Unconstrained, Dmsp, Price_t, Param)
    @unpack (ObjGrid_K, ObjGrid_X, ObjGrid_Zω) = Param
    Nk, Nx, Nzω = ObjGrid_K.N, ObjGrid_X.N, ObjGrid_Zω.N
    Grid_K = ObjGrid_K.Values

    nthresh_uncons = zeros(Nk, Nx, Nzω)
    for ik in 1:Nk
        k = Grid_K[ik]
        for ix in 1:Nx
            for izω in 1:Nzω
                nthresh_uncons[ik, ix, izω] = K_Unconstrained[ik, ix, izω] + kadjfnc(K_Unconstrained[ik, ix, izω], k, Param) - (1.0/Price_t.R) * Dmsp[ik, ix, izω]
            end
        end
    end

    return nthresh_uncons
end


## Iterate to solve the unconstrained firm's problem
function solve_unconstrained_firm(Param; Price_t = (R=1.04, SDF=1.0/1.04, pp=1.25), max_iter = 10, max_viter = 500)
    @unpack (ObjGrid_K, ObjGrid_X, ObjGrid_Zω) = Param
    Nk, Nx, Nzω = ObjGrid_K.N, ObjGrid_X.N, ObjGrid_Zω.N

    # Initialize
    V_Unconstrained  = zeros(Nk, Nx, Nzω)
    E_Unconstrained  = zeros(Nk, Nx, Nzω)
    TE_Unconstrained = zeros(Nk, Nx, Nzω)
    K_Unconstrained  = zeros(Nk, Nx, Nzω)
    Dmsp = zeros(Nk, Nx, Nzω)

    for iter in 1:max_iter
        t_iter = @elapsed begin
            # Solve the efficient capital policy given current engagement policy
            K_guess = (iter == 1) ? nothing : K_Unconstrained
            t_kstar = @elapsed K_Unconstrained, niter_k, diff_k = solve_kstar(E_Unconstrained, Price_t, Param, K_guess)

            # Get the MSP given the efficient capital and engagement policies
            t_msp = @elapsed TDmsp, niter_msp, diff_msp = solve_msp(K_Unconstrained, E_Unconstrained, Price_t, Param, Dmsp)
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
        end

        # Check convergence of engagement policy
        avg_diff = mean(abs.(E_Unconstrained - TE_Unconstrained))

        # Print table
        println("\n---------------------------------------------------------------")
        println("  Unconstrained step    Iters        Max Diff             Secs    ")
        println("---------------------------------------------------------------")
        println("  Engagement (Avg)    $(lpad(iter, 7))  $(lpad(round(avg_diff, sigdigits=6), 14))  $(lpad(round(t_iter, digits=3), 15)) ")
        println("  K* (Capital)        $(lpad(niter_k, 7))  $(lpad(round(diff_k, sigdigits=6), 14))  $(lpad(round(t_kstar, digits=3), 15)) ")
        println("  MSP (Min Savings)   $(lpad(niter_msp, 7))  $(lpad(round(diff_msp, sigdigits=6), 14))  $(lpad(round(t_msp, digits=3), 15)) ")
        println("---------------------------------------------------------------\n")

        if avg_diff < 1e-4
            println("✓ Convergence achieved after ", iter, " iterations.")
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
