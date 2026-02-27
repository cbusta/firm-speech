# Tools_Simple_Fns.jl: Simple functions for the tools module


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

# Dividends
function dividends(pp, x, ω, z, k, e, d, k′, d′, Price_t; Param)
    nw = networth(pp, x, ω, z, k, e, d; Param)
    div = nw  - k′ - kadjfnc(k′, k, Param) + d′/Price_t.R
    return div
end