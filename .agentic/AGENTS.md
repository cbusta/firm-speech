# Firm Speech Project - Agentic AI Context

## Quick Summary

Dynamic firm model analyzing political engagement decisions with financial constraints. Firms choose capital, debt, and whether to engage politically (higher prices but costs). Model features idiosyncratic shocks (productivity z, taste ω), financial frictions, and stochastic political capital dynamics.

**Key insight**: Political engagement decisions intertwined with firm dynamics and financial constraints.

## Project Structure

```
Code_Firm_Speech/
├── src/
│   ├── Firm_Speech.jl                 # Main entry point and analysis
│   ├── Parameters.jl                  # Calibration and parameter struct
│   ├── Tools_Simple_Fns.jl            # Basic helpers (profit, networth)
│   ├── Tools_Unconstrained_Firms.jl   # Unconstrained firm solver
│   ├── Tools_Constrained_Firms.jl     # Constrained firm solver
│   ├── Tools_Distribution.jl          # Stationary distribution
│   ├── Tools_Steady.jl                # Steady state equilibrium
│   ├── Tools_Joint_Process.jl         # Joint (z,ω) utilities
│   └── [Templates & examples]
├── lib/src/
│   ├── Lib_Basic.jl                   # Grids, interpolation
│   └── Lib_Stoch.jl                   # Tauchen discretization
├── results/                           # Saved solutions (JLD2)
└── [Documentation files]
```

## Economic Model

**States**: d (debt), k (capital), x (political capital ∈ {0,1}), z (productivity), ω (taste)

**Decisions**: k′ (investment), d′ (debt), e (engagement ∈ {0,1})

**Technology**: y = z·k^α with adjustment costs ψ_k/2·((k′/k)-(1-δ))²·k

**Political engagement**:

- Cost: c_x per period when e=1
- Benefit: Output price p = ω·pp when x=1, else p=1
- Transitions: e=1 → x′=1; x=1 → x′=0 w/ prob δ_x

**Financial**: Interest R, dividend constraint d′≥0, exit probability χ

**Entry**: Dead firms replaced by entrants with d=0, x=0, k from distribution

## Solution Method

### 1. Unconstrained Firms (`Tools_Unconstrained_Firms.jl`)

Firms with net worth > threshold. Policies depend on (k, x, zω).

**Main function**: `solve_unconstrained_firm(Param; Price_t, verbose)`

**Steps**:

1. **Capital (K*)**: FOC via `find_zero` (Roots.jl) with warm-start
2. **Min Savings (MSP)**: Minimum debt satisfying constraints
3. **Value (V)**: Iteration with 0.9/0.1 dampening
4. **Engagement (E)**: Compare values with hysteresis

### 2. Constrained Firms (`Tools_Constrained_Firms.jl`)

Firms with net worth ≤ threshold, dividend constraint binds. Policies depend on (d, k, x, zω).

**Main function**: `solve_constrained_firm(Param; Price_t, verbose)`

**Steps**:

1. **Capital (Gk)**: Parallelized FOC solving with `@threads`
2. **Debt (Gd)**: From binding dividend constraint
3. **Multiplier (Gλ)**: Lagrange multiplier update
4. **Value + Engagement**: Iteration with hysteresis

**Performance**: Uses `@threads`, `@inbounds`, `@inline`, pre-extracted params.

### 3. Distribution (`Tools_Distribution.jl`)

Computes stationary distribution over states using Young (2010) method.

**Main function**: `solve_stationary_distribution(Sol_Firm, Param)`

**Key functions**:

- `interpolate_policies_to_dense`: Coarse → dense grid
- `update_distribution`: Lottery method for transitions
- `create_entry_distribution`: Configurable entrant distribution
- `compute_aggregates`: Capital, output, engagement rates, etc.

**Grids**:

- **Coarse** (policies): ObjGrid_K (Nk=20), ObjGrid_D (Nd=10)
- **Dense** (distribution): ObjDGrid_K (Nkg=100), ObjDGrid_D (Ndg=100)

### 4. Steady State (`Tools_Steady.jl`)

Solves for equilibrium price pp where excess demand = 0.

**Main function**: `solve_steady(Param; pp_lo, pp_hi, R, tol, verbose)`

**Steps**:

1. `value_firm`: Combine unconstrained/constrained based on net worth
2. `excess_cNP`: Compute excess demand = (I + adj + engage) - cNP
3. Bisect on pp using `find_zero` (Roots.jl)

## Key Data Structures

### Parameters

```julia
Param = Struct_Param(
    # Preferences & technology
    β=0.98, σ=1.0, α=0.25, δ=0.07, ψ_k=5.0, χ=0.08, ϕ=0.5,

    # Political
    δ_x=0.5, c_x=0.5,

    # Grids (coarse for policies)
    ObjGrid_K, ObjGrid_D, ObjGrid_X,

    # Dense grids (for distribution)
    ObjDGrid_K, ObjDGrid_D,

    # Stochastic processes
    ObjGrid_Z, ObjGrid_ω, ObjGrid_Zω,

    # Entry distribution
    EntryDist = (d_entry=0, k_dist=:pareto, k_loc, k_scale, zω_dist=:ergodic),

    # Tolerances
    tol_v=1e-6, tol_μ=1e-12
)
```

### Solutions

```julia
# Unconstrained: (Nk, Nx, Nzω)
Sol_Unconstrained = (Price_t, V_Unconstrained, E_Unconstrained,
                      K_Unconstrained, Dmsp, nthresh_uncons)

# Constrained: (Nd, Nk, Nx, Nzω)
Sol_Constrained = (Price_t, V_Constrained, E_Constrained,
                    K_Constrained, D_Constrained, λ_Constrained)

# Combined: (Nd, Nk, Nx, Nzω)
Sol_Firm = (V_Firm, Gd, Gk, Ge, Is_Unconstrained)

# Distribution: (Ndg, Nkg, Nx, Nzω) - DENSE GRID!
Sol_Dist = (μ, converged, iterations, final_diff)

# Steady state
Steady = (pp_star, Price_t, excess, agg, Sol_Firm, Sol_Dist, ...)
```

## Typical Usage

### Option 1: Fixed Prices

```julia
include("Parameters.jl")
include("Tools_Simple_Fns.jl")
include("Tools_Unconstrained_Firms.jl")
include("Tools_Constrained_Firms.jl")
include("Tools_Distribution.jl")
include("Tools_Steady.jl")

Param = Get_Params()

Sol_Unconstrained = solve_unconstrained_firm(Param)
Sol_Constrained = solve_constrained_firm(Param; Price_t=Sol_Unconstrained.Price_t)
Sol_Firm = value_firm(Sol_Unconstrained, Sol_Constrained, Price_t; Param)
Sol_Dist = solve_stationary_distribution(Sol_Firm, Param)
agg = compute_aggregates(Sol_Dist.μ, Sol_Firm, Sol_Unconstrained, Param, Price_t)
```

### Option 2: Equilibrium

```julia
Steady = solve_steady(Param; pp_lo=0.5, pp_hi=3.0, R=1.04, verbose=true)
# Access: Steady.Sol_Firm, Steady.Sol_Dist, Steady.agg
```

## Joint (Z,ω) Process

**Purpose**: Vectorize nested (z,ω) loops into single loop over joint states.

**Structure**:

- `ObjGrid_Zω.Values`: 2×Nzω matrix, column i = [z_i, ω_i]
- `ObjGrid_Zω.Prob`: Nzω×Nzω transition matrix
- `ObjGrid_Zω.ErgProb`: Ergodic distribution vector

**Usage**:

```julia
for izω in 1:Nzω
    z, ω = ObjGrid_Zω.Values[:, izω]      # Access values
    for jzω in 1:Nzω
        prob = ObjGrid_Zω.Prob[izω, jzω]  # Single lookup!
    end
end
```

## ⚠️ CRITICAL: Political Capital Transitions

```julia
# ObjGrid_X.Prob = [δ_x, 1-δ_x]  ← Conditional on x=1
# ObjGrid_X.Prob[1] = δ_x     → P(x'=0 | x=1)
# ObjGrid_X.Prob[2] = 1-δ_x   → P(x'=1 | x=1)

# For x=0: engagement determines x' (deterministic)
# For x=1: stochastic depreciation
```

## ⚠️ CRITICAL: Dense vs Coarse Grids

```julia
# Policies computed on COARSE grids:
Sol_Firm.Gk  # (Nd, Nk, Nx, Nzω) where Nd=10, Nk=20

# Distribution computed on DENSE grids:
Sol_Dist.μ   # (Ndg, Nkg, Nx, Nzω) where Ndg=100, Nkg=100

# For plotting distribution, use:
Param.ObjDGrid_K.Values  # NOT ObjGrid_K.Values!

# To get policy at dense grid point, interpolate:
e = linterp2(Grid_D_coarse, Grid_K_coarse, Sol_Firm.Ge[:,:,ix,izω], d, k)
```

## Helper Functions

### Core Helpers

```julia
punit(pp, x, ω)                         # Output price
profit(pp, x, ω, z, k, e; Param)        # Flow profits
networth(pp, x, ω, z, k, e, d; Param)   # Net worth
kadjfnc(k′, k, Param)                   # Adjustment cost
```

### Library Functions

```julia
make_grid(type, lb, ub, n, ...)         # Grid generation
linterp1(grid, values, x)               # 1D interpolation
linterp2(grid1, grid2, values, x, y)    # 2D interpolation
make_stochproc(:tauchen, μ, σ, ρ, N)    # Discretize AR(1)
make_joint_stochproc(sp1, sp2)          # Joint process
```

## Development Status

**✅ Complete**:

- Parameter setup with entry distribution
- Coarse and dense grids
- Joint (z,ω) process with ergodic distribution
- Unconstrained firm solver (parallelized)
- Constrained firm solver (parallelized)
- Stationary distribution (Young 2010)
- Aggregate statistics
- Steady state equilibrium solver

**📋 To Do**:

- General equilibrium with household
- Comparative statics
- Calibration to data
- Welfare analysis

## Implementation Details

### Convergence

- **Policies**: `max_diff < tol_v / 100`
- **Value function**: `vdiff < tol_v`
- **Engagement**: `avg_diff < tol_v * 100` (with hysteresis)
- **Distribution**: `max_diff < tol_μ`

### Hysteresis for Engagement

```julia
hysteresis_tol = Param.tol_v * 100
if eval ≈ 0.0  # Currently not engaging
    e_new = (𝓥_p > 𝓥_n + hysteresis_tol) ? 1.0 : 0.0
else           # Currently engaging
    e_new = (𝓥_n > 𝓥_p + hysteresis_tol) ? 0.0 : 1.0
end
```

### Dampening

```julia
V_new = 0.9 * V_old + 0.1 * TV
```

### Parallelization

```julia
using Base.Threads: @threads
@threads for id in 1:Nd
    # ... computation
end
```

## Common Code Patterns

### Policy Function Iteration

```julia
K_old = initial_guess()
for iter in 1:max_iter
    K_new = solve_foc(K_old, ...)
    maximum(abs.(K_new - K_old)) < tol && break
    K_old .= K_new
end
```

### Expectation Computation

```julia
# Vectorized (recommended)
EV = reshape(reshape(V, n_other, Nzω) * ObjGrid_Zω.Prob', original_dims)
```

### Distribution Update

```julia
for id, ik, ix, izω in states
    mass = μ[id, ik, ix, izω]
    k′, d′ = policies[...]
    ik_lo, ik_hi, wk = get_interp_weights(k′, DGrid_K)
    # Distribute mass with lottery weights
    μ_next[...] += mass * prob * wk * wd
end
```

## Grid Sizes (Typical)

- Capital: Nk=20 (coarse), Nkg=100 (dense)
- Debt: Nd=10 (coarse), Ndg=100 (dense)
- Political: Nx=2
- Productivity: Nz=5
- Taste: Nω=5
- Joint: Nzω=25

## Best Practices

1. **@unpack**: Always extract params with `@unpack (α, δ, ...) = Param`
2. **Dense grids**: Use ObjDGrid for distribution, ObjGrid for policies
3. **Interpolation**: Use `linterp2` when mapping policies to dense grid
4. **Verbose control**: Pass `verbose=false` in inner loops/equilibrium
5. **Check convergence**: Monitor iteration tables
6. **Validate visually**: Plot policies before equilibrium

## Context for AI Agents

When modifying code:

1. Maintain coarse/dense grid distinction
2. Use parallelization patterns for constrained problem
3. Follow hysteresis convention for engagement
4. Update CLAUDE.md if adding major components
5. Use `find_zero` from Roots.jl for bisection
