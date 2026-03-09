# Firm Speech Project - Claude Context

## Project Overview

This project implements a dynamic firm model with political engagement decisions. Firms face idiosyncratic productivity (z) and taste shifter (ω) shocks, make capital accumulation decisions, choose debt levels, and decide whether to engage in political activities that can provide preferential treatment (higher output prices).

**Research Question**: How do firm dynamics, financial constraints, and political engagement interact? What determines which firms engage politically?

## Project Structure

```
Code_Firm_Speech/
├── src/
│   ├── Firm_Speech.jl                       # Main entry point and analysis
│   ├── Parameters.jl                        # Model parameters and calibration
│   ├── Tools_Simple_Fns.jl                  # Basic helper functions (profits, net worth)
│   ├── Tools_Unconstrained_Firms.jl         # Solving unconstrained firm problem
│   ├── Tools_Constrained_Firms.jl           # Solving constrained firm problem
│   ├── Tools_Distribution.jl                # Stationary distribution computation
│   ├── Tools_Steady.jl                      # Steady state equilibrium solver
│   ├── Tools_Joint_Process.jl               # Joint (Z,ω) process utilities
│   ├── Template_Vectorized_Expectations.jl  # Code migration templates
│   └── Example_Joint_Process_Usage.jl       # Usage examples
├── lib/
│   └── src/
│       ├── Lib_Basic.jl            # Grid generation, interpolation
│       └── Lib_Stoch.jl            # Stochastic process discretization
├── results/                        # Saved solutions (JLD2 files)
├── test/
│   └── runtests.jl                 # Unit tests
├── tex/                            # LaTeX documentation
├── JOINT_PROCESS_README.md         # Joint process documentation
├── JOINT_PROCESS_SUMMARY.md        # Quick reference
└── Project.toml                    # Julia dependencies
```

## Model Components

### State Variables
- **d**: Current debt level (grid: `ObjGrid_D`, dense: `ObjDGrid_D`)
- **k**: Current capital stock (grid: `ObjGrid_K`, dense: `ObjDGrid_K`)
- **x**: Political capital (0 or 1)
- **z**: Idiosyncratic productivity shock
- **ω**: Idiosyncratic taste shifter (affects price when politically engaged)

### Firm Types
1. **Unconstrained firms**: Net worth > threshold, choose optimal policies
2. **Constrained firms**: Net worth ≤ threshold, face dividend constraint d≥0

### Key Features

#### Production & Technology
- Production function: `y = z * k^α`
- Capital adjustment costs: `ψ_k/2 * ((k′/k) - (1-δ))^2 * k`
- Depreciation rate: `δ`

#### Political Engagement
- **Cost**: `c_x` per period when engaged (e=1)
- **Benefit**: Higher output price `p = ω * pp` when x=1, otherwise p=1
- **Dynamics**:
  - If x=0 and e=1: x′=1 with probability 1
  - If x=0 and e=0: x′=0 with probability 1
  - If x=1: x′ depreciates to 0 with probability δ_x

#### Financial Structure
- Interest rate: `R`
- Stochastic discount factor: `SDF = 1/R`
- Dividend constraint: d′ ≥ 0 (cannot distribute future earnings)
- Exit probability: `χ` (firm death/restructuring)

### Stochastic Processes

The model uses **Tauchen discretization** for AR(1) processes:

1. **Productivity (z)**:
   - AR(1): `log(z′) = ρ_z * log(z) + ε_z`
   - Parameters: `μ_z, σ_z, ρ_z`
   - Grid points: `Nz` (typically 5)

2. **Taste shifter (ω)**:
   - AR(1): `log(ω′) = ρ_ω * log(ω) + ε_ω`
   - Parameters: `μ_ω, σ_ω, ρ_ω`
   - Grid points: `Nω` (typically 5)

3. **Joint (z,ω) process**:
   - Independent processes combined via Kronecker product
   - `ObjGrid_Zω.Values`: 2×Nzω matrix where column i is `[z_i, ω_i]`
   - `ObjGrid_Zω.Prob`: Nzω×Nzω transition matrix
   - `ObjGrid_Zω.ErgProb`: Ergodic distribution vector
   - Total states: `Nzω = Nz * Nω`

## Solution Algorithm

### Unconstrained Firms (`Tools_Unconstrained_Firms.jl`)

**Main function**: `solve_unconstrained_firm(Param; Price_t, max_iter, max_viter, verbose)`

**Iterative procedure**:
1. **Capital policy iteration** (`solve_kstar`):
   - Given engagement policy E, solve FOC: `1 + dk′_kadjfnc = SDF * E[MR(k′)]`
   - Uses `find_zero` (Roots.jl) with bisection
   - Warm-start: uses previous iteration as initial guess

2. **Minimum savings policy** (`solve_msp`):
   - Compute minimum debt d′ that satisfies constraints
   - Accounts for worst-case scenarios (x′=0 when possible)

3. **Value function iteration** (`value_uncons`):
   - Update value given policies (K*, D_msp, E)
   - Dampening: `V_new = 0.9*V_old + 0.1*TV`
   - Convergence: `vdiff < Param.tol_v`

4. **Engagement policy update**:
   - For x=0: Compare `V_engage` vs `V_no_engage` with hysteresis
   - For x=1: No choice (depreciation risk)

### Constrained Firms (`Tools_Constrained_Firms.jl`)

**Main function**: `solve_constrained_firm(Param; Price_t, max_iter, max_viter, verbose)`

**Iterative procedure**:
1. **Capital policy** (`solve_k`): Parallelized FOC solving with `@threads`
2. **Debt policy** (`update_Gd`): From binding dividend constraint
3. **Multiplier update** (`update_Gλ`): Lagrange multiplier on constraint
4. **Value iteration** (`value_constrained`): With engagement policy update

**Performance optimizations**:
- `@threads` parallelization across states
- `@inbounds`, `@inline` annotations
- Pre-extracted parameters for fast inner loops
- Hysteresis in engagement to prevent oscillation

### Distribution (`Tools_Distribution.jl`)

**Main function**: `solve_stationary_distribution(Sol_Firm, Param)`

Uses **Young (2010) method** with lottery/interpolation:
1. Interpolate coarse-grid policies to dense grid (`interpolate_policies_to_dense`)
2. Update distribution with transition probabilities (`update_distribution`)
3. Add entrants with configurable entry distribution (`add_entrants!`)
4. Iterate until `max_diff < Param.tol_μ`

**Entry distribution** (`create_entry_distribution`):
- Configurable via `Param.EntryDist`
- `d_entry`: Entry debt level (default 0)
- `k_dist`: `:lognormal` or `:pareto`
- `k_loc`, `k_scale`: Distribution parameters
- `zω_dist`: `:ergodic` or `:uniform`

**Aggregate statistics** (`compute_aggregates`):
- Capital, debt, output, investment
- Engagement rates and costs
- Political vs non-political output (cP, cNP)
- Fraction unconstrained

### Steady State Equilibrium (`Tools_Steady.jl`)

**Main function**: `solve_steady(Param; pp_lo, pp_hi, R, tol, verbose)`

1. **`value_firm`**: Combines unconstrained and constrained solutions
   - For each state, checks if net worth > threshold
   - Returns unified policy functions (Gd, Gk, Ge)

2. **`excess_cNP`**: Computes excess demand for non-differentiated good
   - Demand: investment + adjustment costs + engagement costs
   - Supply: output from non-engaged firms (x=0)
   - Returns excess = demand - supply

3. **`solve_steady`**: Bisects on `pp` using `find_zero` (Roots.jl)
   - Finds equilibrium price where excess demand = 0
   - Returns full solution at equilibrium

## Key Data Structures

### Parameter Structure (`Struct_Param`)
```julia
struct Struct_Param{T,GK,DK,GD,DD,GX,GW,GZ,GZW,ED}
    # Preferences
    β::T                    # Discount factor (0.98)
    σ::T                    # Risk aversion (1.0)
    ξ::T                    # Political preference weight (0.5)

    # Technology
    α::T                    # Capital share (0.25)
    δ::T                    # Depreciation (0.07)
    ψ_k::T                  # Adjustment cost (5.0)
    χ::T                    # Exit probability (0.08)
    ϕ::T                    # Borrowing constraint (0.5)

    # Grids (coarse for policies)
    ObjGrid_K::GK          # Capital grid (log-spaced, Nk points)
    ObjGrid_D::GD          # Debt grid (double-log around 0, Nd points)

    # Dense grids (for distribution)
    ObjDGrid_K::DK         # Dense capital grid (Nkg points)
    ObjDGrid_D::DD         # Dense debt grid (Ndg points)

    # Political capital
    δ_x::T                 # Depreciation probability (0.5)
    c_x::T                 # Engagement cost (0.5)
    ObjGrid_X::GX          # Grid {0,1}

    # Stochastic processes
    ObjGrid_ω::GW          # Taste shifter (Tauchen)
    ObjGrid_Z::GZ          # Productivity (Tauchen)
    ObjGrid_Zω::GZW        # Joint process

    # Entry distribution
    EntryDist::ED          # (d_entry, k_dist, k_loc, k_scale, zω_dist)

    # Numerical
    tol_v::T               # Value function tolerance (1e-6)
    tol_μ::T               # Distribution tolerance (1e-12)
end
```

### Solution Objects
```julia
# Unconstrained solution (Nk × Nx × Nzω)
Sol_Unconstrained = (
    Price_t,           # (R, SDF, pp)
    V_Unconstrained,   # Value function
    E_Unconstrained,   # Engagement policy
    K_Unconstrained,   # Capital policy
    Dmsp,              # Minimum savings policy
    nthresh_uncons     # Net worth threshold
)

# Constrained solution (Nd × Nk × Nx × Nzω)
Sol_Constrained = (
    Price_t,
    V_Constrained,
    E_Constrained,
    K_Constrained,
    D_Constrained,
    λ_Constrained      # Multiplier on dividend constraint
)

# Combined firm solution (Nd × Nk × Nx × Nzω)
Sol_Firm = (
    V_Firm,
    Gd, Gk, Ge,
    Is_Unconstrained   # Boolean indicator
)

# Distribution (Ndg × Nkg × Nx × Nzω) - on dense grid
Sol_Dist = (
    μ,                 # Stationary distribution
    converged,
    iterations,
    final_diff
)

# Steady state equilibrium
Steady = (
    pp_star,           # Equilibrium price
    Price_t,
    excess,            # Final excess demand
    agg,               # Aggregate statistics
    Sol_Firm, Sol_Dist, Sol_Unconstrained, Sol_Constrained
)
```

## Typical Workflow

### Option 1: Fixed Prices
```julia
include("Parameters.jl")
include("Tools_Simple_Fns.jl")
include("Tools_Unconstrained_Firms.jl")
include("Tools_Constrained_Firms.jl")
include("Tools_Distribution.jl")
include("Tools_Steady.jl")

Param = Get_Params()

# Solve at fixed prices
Sol_Unconstrained = solve_unconstrained_firm(Param)
Sol_Constrained = solve_constrained_firm(Param; Price_t=Sol_Unconstrained.Price_t)
Sol_Firm = value_firm(Sol_Unconstrained, Sol_Constrained, Sol_Unconstrained.Price_t; Param)

# Compute distribution
Sol_Dist = solve_stationary_distribution(Sol_Firm, Param)

# Compute aggregates
agg = compute_aggregates(Sol_Dist.μ, Sol_Firm, Sol_Unconstrained, Param, Price_t)
print_aggregates(agg)
```

### Option 2: Solve for Equilibrium
```julia
Steady = solve_steady(Param; pp_lo=0.5, pp_hi=3.0, R=1.04, verbose=true)

# Access equilibrium solutions
Sol_Firm = Steady.Sol_Firm
Sol_Dist = Steady.Sol_Dist
agg = Steady.agg
```

## CRITICAL: Political Capital Transition Probabilities

**⚠️ IMPORTANT**: When computing continuation values for firms with x=1:

```julia
# ObjGrid_X.Prob contains CONDITIONAL transition probabilities FROM x=1:
ObjGrid_X.Prob[1] = δ_x      # P(x'=0 | x=1) - depreciation to 0
ObjGrid_X.Prob[2] = 1 - δ_x  # P(x'=1 | x=1) - persistence at 1
```

**Key Points**:
- For x=0: engagement choice directly determines x' (deterministic)
- For x=1: stochastic depreciation using ObjGrid_X.Prob
- Grid_X indexing: ix=1 → x=0, ix=2 → x=1

## Grid Conventions

### Coarse Grids (for policy functions)
- `ObjGrid_K`: Capital (Nk=20 points, log-spaced)
- `ObjGrid_D`: Debt (Nd=10 points, double-log around 0)

### Dense Grids (for distribution)
- `ObjDGrid_K`: Capital (Nkg=100 points)
- `ObjDGrid_D`: Debt (Ndg=100 points)

**Important**: Distribution μ is computed on dense grids. When plotting or computing statistics, use:
- `Param.ObjDGrid_K.Values` for capital
- `Param.ObjDGrid_D.Values` for debt
- Interpolate policy functions from coarse to dense using `linterp2`

## Development Status

### ✅ Completed
- Parameter structure and calibration
- Grid generation (capital, debt, political, stochastic)
- Joint (z,ω) process implementation
- Unconstrained firm solution (parallelized)
- Constrained firm solution (parallelized)
- Stationary distribution computation
- Entry distribution (configurable)
- Aggregate statistics
- Steady state equilibrium solver
- Market clearing for non-differentiated good

### 📋 To Do
- General equilibrium with household sector
- Comparative statics
- Model validation and calibration to data
- Welfare analysis

## Important Implementation Details

### Convergence Criteria
- **Policy functions**: `max_diff < Param.tol_v / 100`
- **Value function**: `vdiff < Param.tol_v`
- **Engagement policy**: `avg_diff < Param.tol_v * 100`
- **Distribution**: `max_diff < Param.tol_μ`

### Hysteresis for Engagement
To prevent oscillation in discrete engagement choice:
```julia
hysteresis_tol = Param.tol_v * 100
if eval ≈ 0.0
    # Currently not engaging: only start if benefit exceeds threshold
    e_new = (𝓥_p > 𝓥_n + hysteresis_tol) ? 1.0 : 0.0
else
    # Currently engaging: only stop if benefit of stopping exceeds threshold
    e_new = (𝓥_n > 𝓥_p + hysteresis_tol) ? 0.0 : 1.0
end
```

### Dampening
Value function uses dampening for stability:
```julia
V_Constrained .= 0.9 .* V_Constrained .+ 0.1 .* TV_Constrained
```

### Warm Start
Capital policy iteration uses previous solution as initial guess after first iteration.

## Helper Functions

### Basic Functions (`Tools_Simple_Fns.jl`)
- `punit(pp, x, ω)`: Output price (1 if x=0, ω*pp if x=1)
- `profit(pp, x, ω, z, k, e; Param)`: Flow profits
- `networth(pp, x, ω, z, k, e, d; Param)`: Net worth

### Adjustment Costs (`Parameters.jl`)
- `kadjfnc(k′, k, Param)`: Adjustment cost level
- `dk′_kadjfnc(k′, k, Param)`: Derivative w.r.t. k′
- `dk_kadjfnc(k′, k, Param)`: Derivative w.r.t. k

### Library Functions (`lib/src/`)
- **Lib_Basic.jl**: `make_grid`, `linterp1`, `linterp2`, `grid_info`
- **Lib_Stoch.jl**: `tauchen`, `make_stochproc`, `make_joint_stochproc`

## Tips for Working with This Code

1. **Always use @unpack**: Extract parameters with `@unpack (α, δ, ...) = Param`

2. **Check array dimensions**:
   - Unconstrained: (Nk, Nx, Nzω)
   - Constrained/Combined: (Nd, Nk, Nx, Nzω)
   - Distribution: (Ndg, Nkg, Nx, Nzω) - **dense grid!**

3. **Use helper functions**: Don't compute profits/net worth manually

4. **Monitor performance**: Check iteration counts and timing in output tables

5. **Validate with plots**: Always visualize policies before using in equilibrium

6. **Dense vs Coarse grids**: Distribution uses dense grids; interpolate policies as needed
