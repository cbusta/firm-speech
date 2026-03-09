# Template_Vectorized_Expectations.jl
# Shows how to adapt expectation computations in Tools_Unconstrained_Firms.jl

using UnPack

"""
Example showing how to convert from OLD nested loop style to NEW joint process style.
"""

# ============================================================================
# EXAMPLE 1: Simple expectation in Gkstar (from Tools_Unconstrained_Firms.jl)
# ============================================================================

## OLD VERSION (lines ~18-32 in Tools_Unconstrained_Firms.jl)
function Gkstar_old(k′, ik, ix, iz, iω, K_Unconstrained, E_Unconstrained, Price_tf, Param) 
    @unpack (α, δ, χ, ObjGrid_ω, ObjGrid_Z, ObjGrid_X, ObjGrid_K) = Param
    Grid_ω = ObjGrid_ω.Values
    Grid_Z = ObjGrid_Z.Values
    Grid_X = ObjGrid_X.Values
    Grid_K = ObjGrid_K.Values
    
    # Compute expected marginal return to capital
    Expectation = 0.0
    for (iω′, ω′) in enumerate(Grid_ω)
        for (iz′, z′) in enumerate(Grid_Z)
            prob_zω = ObjGrid_ω.Prob[iω, iω′] * ObjGrid_Z.Prob[iz, iz′]
            if Grid_X[ix] == 0
                x′_val  = E_Unconstrained[ik, ix, iz, iω]
                ix′     = (x′_val == 0) ? 1 : 2
                x′      = Grid_X[ix′]
                kstar′′ = linterp1(Grid_K, K_Unconstrained[:, ix′, iz′, iω′], k′)
                term = punit(Price_tf.pp, x′, ω′) * z′ * α * (k′)^(α - 1.0) + (1.0 - δ) - (1.0 - χ) * dk_kadjfnc(kstar′′, k′, Param)
                Expectation += prob_zω * term
            else
                # Handle x=1 case...
            end
        end
    end
    return Expectation
end


## NEW VERSION (with joint process)
function Gkstar_new(k′, ik, ix, izω, K_Unconstrained, E_Unconstrained, Price_tf, Param) 
    @unpack (α, δ, χ, ObjGrid_Zω, ObjGrid_X, ObjGrid_K, ObjGrid_ω) = Param
    Grid_X = ObjGrid_X.Values
    Grid_K = ObjGrid_K.Values
    
    # Get current z and ω values
    iz, iω = get_separate_indices(izω, ObjGrid_ω.N)
    
    # Compute expected marginal return to capital
    Expectation = 0.0
    for jzω in 1:ObjGrid_Zω.N
        iz′, iω′ = get_separate_indices(jzω, ObjGrid_ω.N)
        z′, ω′ = get_zω_values(jzω, Param)
        
        prob_zω = ObjGrid_Zω.Prob[izω, jzω]  # Single lookup instead of product!
        
        if Grid_X[ix] == 0
            x′_val  = E_Unconstrained[ik, ix, izω]  # Note: izω instead of (iz,iω)
            ix′     = (x′_val == 0) ? 1 : 2
            x′      = Grid_X[ix′]
            kstar′′ = linterp1(Grid_K, K_Unconstrained[:, ix′, jzω], k′)
            term = punit(Price_tf.pp, x′, ω′) * z′ * α * (k′)^(α - 1.0) + (1.0 - δ) - (1.0 - χ) * dk_kadjfnc(kstar′′, k′, Param)
            Expectation += prob_zω * term
        else
            # Handle x=1 case...
        end
    end
    return Expectation
end


# ============================================================================
# EXAMPLE 2: Array structure changes
# ============================================================================

# OLD: Arrays have separate (iz, iω) dimensions
# Shape: (Nk, Nx, Nz, Nω)
K_Unconstrained_old = zeros(Param.ObjGrid_K.N, Param.ObjGrid_X.N, Param.ObjGrid_Z.N, Param.ObjGrid_ω.N)

# NEW: Arrays collapse (iz, iω) into single izω dimension  
# Shape: (Nk, Nx, Nzω)
K_Unconstrained_new = zeros(Param.ObjGrid_K.N, Param.ObjGrid_X.N, Param.ObjGrid_Zω.N)

# Accessing elements:
# OLD: K_Unconstrained_old[ik, ix, iz, iω]
# NEW: K_Unconstrained_new[ik, ix, izω]  where izω = get_joint_index(iz, iω, Nω)


# ============================================================================
# EXAMPLE 3: Loop structure changes
# ============================================================================

# OLD: Nested loops over z and ω
function old_loop_style(Param)
    @unpack ObjGrid_K, ObjGrid_X, ObjGrid_Z, ObjGrid_ω = Param
    result = zeros(ObjGrid_K.N, ObjGrid_X.N, ObjGrid_Z.N, ObjGrid_ω.N)
    
    for ik in 1:ObjGrid_K.N
        for ix in 1:ObjGrid_X.N
            for iz in 1:ObjGrid_Z.N
                for iω in 1:ObjGrid_ω.N
                    # Computation...
                    result[ik, ix, iz, iω] = 0.0  # placeholder
                end
            end
        end
    end
    return result
end

# NEW: Single loop over joint process
function new_loop_style(Param)
    @unpack ObjGrid_K, ObjGrid_X, ObjGrid_Zω = Param
    result = zeros(ObjGrid_K.N, ObjGrid_X.N, ObjGrid_Zω.N)
    
    for ik in 1:ObjGrid_K.N
        for ix in 1:ObjGrid_X.N
            for izω in 1:ObjGrid_Zω.N
                # Can get separate indices if needed:
                # iz, iω = get_separate_indices(izω, Param.ObjGrid_ω.N)
                # z, ω = get_zω_values(izω, Param)
                
                # Computation...
                result[ik, ix, izω] = 0.0  # placeholder
            end
        end
    end
    return result
end


# ============================================================================
# EXAMPLE 4: Fully vectorized expectation
# ============================================================================

# For computing E[V(k′, x′, z′, ω′) | z, ω] for all states at once

# OLD: Loop-based
function compute_ev_old(V0, Param)
    @unpack ObjGrid_K, ObjGrid_X, ObjGrid_Z, ObjGrid_ω = Param
    Nk, Nx, Nz, Nω = ObjGrid_K.N, ObjGrid_X.N, ObjGrid_Z.N, ObjGrid_ω.N
    
    EV = zeros(Nk, Nx, Nz, Nω)
    for ik in 1:Nk, ix in 1:Nx, iz in 1:Nz, iω in 1:Nω
        for iz′ in 1:Nz, iω′ in 1:Nω
            prob_zω = ObjGrid_Z.Prob[iz, iz′] * ObjGrid_ω.Prob[iω, iω′]
            EV[ik, ix, iz, iω] += prob_zω * V0[ik, ix, iz′, iω′]
        end
    end
    return EV
end

# NEW: Fully vectorized
function compute_ev_new(V0, Param)
    @unpack ObjGrid_K, ObjGrid_X, ObjGrid_Zω = Param
    Nk, Nx, Nzω = ObjGrid_K.N, ObjGrid_X.N, ObjGrid_Zω.N
    
    # Reshape to (Nk*Nx, Nzω)
    V0_reshaped = reshape(V0, Nk*Nx, Nzω)
    
    # Matrix multiplication: V0 * Prob'
    EV_reshaped = V0_reshaped * ObjGrid_Zω.Prob'
    
    # Reshape back to (Nk, Nx, Nzω)
    return reshape(EV_reshaped, Nk, Nx, Nzω)
end


# ============================================================================
# MIGRATION CHECKLIST
# ============================================================================

"""
To migrate existing code to use joint process:

1. Update array dimensions:
   - (Nk, Nx, Nz, Nω) → (Nk, Nx, Nzω)
   - Use Param.ObjGrid_Zω.N instead of separate Nz*Nω

2. Update loop structures:
   - for iz in 1:Nz, iω in 1:Nω → for izω in 1:ObjGrid_Zω.N
   - Get separate indices when needed: iz, iω = get_separate_indices(izω, Nω)

3. Update probability computations:
   - prob_zω = ObjGrid_Z.Prob[iz,iz′] * ObjGrid_ω.Prob[iω,iω′]
   → prob_zω = ObjGrid_Zω.Prob[izω, jzω]

4. Update value access:
   - V[ik, ix, iz, iω] → V[ik, ix, izω]
   - V[ik, ix, iz′, iω′] → V[ik, ix, jzω]

5. Use helper functions:
   - Include Tools_Joint_Process.jl at top of file
   - Use get_joint_index(), get_separate_indices(), get_zω_values()

6. Consider vectorization:
   - For expectation computations, use matrix multiplication
   - V0 * ObjGrid_Zω.Prob' computes all expectations at once
"""
