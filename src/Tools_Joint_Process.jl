# Tools_Joint_Process.jl: Helper functions for working with joint (Z,ω) process

"""
    get_joint_index(iz::Int, iω::Int, Nω::Int) -> izω

Convert separate z and ω indices to joint index.
Joint index ordering: izω = (iz-1)*Nω + iω
"""
@inline function get_joint_index(iz::Int, iω::Int, Nω::Int)
    return (iz-1)*Nω + iω
end

"""
    get_separate_indices(izω::Int, Nω::Int) -> (iz, iω)

Convert joint index to separate z and ω indices.
"""
@inline function get_separate_indices(izω::Int, Nω::Int)
    iz = div(izω - 1, Nω) + 1
    iω = mod(izω - 1, Nω) + 1
    return iz, iω
end

"""
    get_zω_values(izω::Int, Grid_Zω) -> (z, ω)

Get z and ω values from joint index.

# Arguments
- `izω`: Joint index
- `Grid_Zω`: Joint grid structure from make_joint_stochproc()

# Returns
- `(z, ω)`: Tuple of z and ω values
"""
@inline function get_zω_values(izω::Int, Grid_Zω)
    z = Grid_Zω.Values[1, izω]
    ω = Grid_Zω.Values[2, izω]
    return z, ω
end

"""
    expectation_over_zω(values, izω::Int, Grid_Zω)

Compute expectation over future (z′,ω′) given current joint state izω.

# Arguments
- `values`: Vector indexed by joint (z,ω) index
- `izω`: Current joint index
- `Grid_Zω`: Joint grid structure

# Returns
- Expected value: E[values(z′,ω′) | z, ω]
"""
function expectation_over_zω(values, izω::Int, Grid_Zω)
    expectation = 0.0
    for jzω in 1:Grid_Zω.N
        prob = Grid_Zω.Prob[izω, jzω]
        expectation += prob * values[jzω]
    end
    return expectation
end

"""
    compute_ev_vectorized(V0, Grid_Zω)

Compute expected continuation value for all states using vectorized operations.
V0 should have joint (z,ω) dimension as the last dimension.

# Arguments
- `V0`: Value function array with joint dimension last: (..., Nzω)
- `Grid_Zω`: Joint grid structure

# Returns
- `EV`: Expected value array with same dimensions as V0
"""
function compute_ev_vectorized(V0, Grid_Zω)
    # Get dimensions
    dims = size(V0)
    n_other = prod(dims[1:end-1])  # All dimensions except last (zω)
    Nzω = Grid_Zω.N
    
    # Reshape V0 to (n_other, Nzω)
    V0_reshaped = reshape(V0, n_other, Nzω)
    
    # Compute expectation: EV = V0 * Prob'
    EV_reshaped = V0_reshaped * Grid_Zω.Prob'
    
    # Reshape back to original dimensions
    return reshape(EV_reshaped, dims)
end
