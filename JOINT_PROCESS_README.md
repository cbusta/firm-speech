# Joint (Z,ω) Process Documentation

## Overview

This system provides a vectorized approach to handling independent stochastic processes (Z and ω) in dynamic programming problems. Instead of nested loops over separate (iz, iω) indices, the joint process combines them into a single index using Kronecker products.

## Key Components

### 1. Core Functions (in `lib/src/Lib_Stoch.jl`)

- **`make_joint_stochproc(sp1, sp2; name="")`**: Creates joint process from two independent processes
- **`split_joint_index(i_joint, N2)`**: Converts joint index to (i1, i2)
- **`joint_index(i1, i2, N2)`**: Converts (i1, i2) to joint index

### 2. Helper Functions (in `src/Tools_Joint_Process.jl`)

- **`get_joint_index(iz, iω, Nω)`**: Get joint index from z and ω indices
- **`get_separate_indices(i_joint, Nω)`**: Get z and ω indices from joint index
- **`get_zω_values(i_joint, Param)`**: Get actual z and ω values from joint index
- **`expectation_over_zω(values, iz, iω, Param)`**: Compute expectation over (z′,ω′)
- **`compute_ev_all_states_joint(V0, Param)`**: Vectorized expectation for all states

## Usage

### Basic Setup

```julia
using Lib_Julia_Econ

include("Parameters.jl")
include("Tools_Joint_Process.jl")

# Parameters already include joint process
Param = Get_Params()

@unpack ObjGrid_Z, ObjGrid_ω, ObjGrid_Zω = Param
```

### Index Conversion

```julia
# From separate to joint
iz, iω = 2, 3
izω = get_joint_index(iz, iω, ObjGrid_ω.N)

# From joint to separate
iz, iω = get_separate_indices(izω, ObjGrid_ω.N)

# Get values
z, ω = get_zω_values(izω, Param)
```

### Array Structures

**OLD approach** (separate dimensions):
```julia
# Shape: (Nk, Nx, Nz, Nω)
K_policy = zeros(Nk, Nx, Nz, Nω)
V_function = zeros(Nk, Nx, Nz, Nω)

# Access
K_policy[ik, ix, iz, iω]
```

**NEW approach** (joint dimension):
```julia
# Shape: (Nk, Nx, Nzω)
K_policy = zeros(Nk, Nx, Nzω)
V_function = zeros(Nk, Nx, Nzω)

# Access
K_policy[ik, ix, izω]
```

### Loop Structures

**OLD approach** (nested loops):
```julia
for ik in 1:Nk
    for ix in 1:Nx
        for iz in 1:Nz
            for iω in 1:Nω
                # Inner expectation loop
                expectation = 0.0
                for iz′ in 1:Nz
                    for iω′ in 1:Nω
                        prob_zω = ObjGrid_Z.Prob[iz, iz′] * ObjGrid_ω.Prob[iω, iω′]
                        expectation += prob_zω * V[ik, ix, iz′, iω′]
                    end
                end
                # Use expectation...
            end
        end
    end
end
```

**NEW approach** (single loop):
```julia
for ik in 1:Nk
    for ix in 1:Nx
        for izω in 1:Nzω
            # Inner expectation loop
            expectation = 0.0
            for jzω in 1:Nzω
                prob_zω = ObjGrid_Zω.Prob[izω, jzω]  # Single lookup!
                expectation += prob_zω * V[ik, ix, jzω]
            end
            # Use expectation...
        end
    end
end
```

### Expectation Computations

**Simple expectation** (for one state):
```julia
# Given current (iz, iω), compute E[f(z′,ω′)]
future_values = [....]  # Vector indexed by jzω
expected_value = expectation_over_zω(future_values, iz, iω, Param)
```

**Vectorized expectation** (for all states):
```julia
# Compute E[V(z′,ω′)|z,ω] for all (k,x,z,ω) states at once
# V0 has shape (Nk, Nx, Nzω)
EV = compute_ev_all_states_joint(V0, Param)
# EV has same shape as V0
```

**Manual vectorization**:
```julia
# Reshape to (Nk*Nx, Nzω)
V0_reshaped = reshape(V0, Nk*Nx, Nzω)

# Matrix multiply: each row is E[V|current state]
EV_reshaped = V0_reshaped * ObjGrid_Zω.Prob'

# Reshape back
EV = reshape(EV_reshaped, Nk, Nx, Nzω)
```

## Migration Guide

### Step 1: Update Parameters
Already done - `Param` now includes `ObjGrid_Zω`

### Step 2: Update Array Declarations
```julia
# OLD
V = zeros(Nk, Nx, Nz, Nω)

# NEW
V = zeros(Nk, Nx, Nzω)
```

### Step 3: Update Loop Indices
```julia
# OLD
for iz in 1:Nz, iω in 1:Nω
    z = Grid_Z[iz]
    ω = Grid_ω[iω]
    # ...
end

# NEW
for izω in 1:Nzω
    iz, iω = get_separate_indices(izω, Nω)
    z, ω = get_zω_values(izω, Param)
    # ...
end
```

### Step 4: Update Probability Computations
```julia
# OLD
prob_zω = ObjGrid_Z.Prob[iz, iz′] * ObjGrid_ω.Prob[iω, iω′]

# NEW
prob_zω = ObjGrid_Zω.Prob[izω, jzω]
```

### Step 5: Update Function Signatures
```julia
# OLD
function my_function(ik, ix, iz, iω, arrays, Param)
    value = arrays.V[ik, ix, iz, iω]
    # ...
end

# NEW
function my_function(ik, ix, izω, arrays, Param)
    value = arrays.V[ik, ix, izω]
    # ...
end
```

## Performance Benefits

1. **Fewer nested loops**: 4 levels → 3 levels (eliminates innermost loop)
2. **Vectorized probability lookup**: One matrix access vs. two multiplications
3. **Better cache locality**: Contiguous memory for (z,ω) pairs
4. **Easier vectorization**: Single matrix multiplication for expectations
5. **Cleaner code**: Fewer loop indices to track

## Examples

See these files for detailed examples:
- `src/Example_Joint_Process_Usage.jl` - Usage examples and timing comparisons
- `src/Template_Vectorized_Expectations.jl` - Migration templates for common patterns

## Technical Details

### Index Ordering
Joint index `izω` maps to `(iz, iω)` using column-major ordering:
```
izω = (iz-1)*Nω + iω
iz = div(izω - 1, Nω) + 1
iω = mod(izω - 1, Nω) + 1
```

### Transition Matrix
The joint transition matrix is the Kronecker product:
```
Prob_joint[i,j] = Prob_Z[iz,jz] * Prob_ω[iω,jω]
```
where `i = get_joint_index(iz,iω,Nω)` and `j = get_joint_index(jz,jω,Nω)`.

### Expectation Formula
For independent processes:
```
E[f(z′,ω′)|z,ω] = Σ_{z′,ω′} P(z′|z) P(ω′|ω) f(z′,ω′)
                = Σ_{j} Prob_joint[i,j] f_j
```
where `i`, `j` are joint indices.

## Compatibility

- **Backward compatible**: Old separate-index code still works
- **Gradual migration**: Can mix old and new approaches during transition
- **Type-stable**: All functions properly typed for performance
- **Generic**: Works with any two independent StochProcess objects
