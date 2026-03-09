# Example_Joint_Process_Usage.jl
# Demonstrates how to use the joint (Z,ω) process wrapper

using Revise, UnPack
using Lib_Julia_Econ

include("Parameters.jl")
include("Tools_Joint_Process.jl")

# Get parameters with joint process
Param = Get_Params()

# Access individual and joint processes
@unpack ObjGrid_Z, ObjGrid_ω, Grid_Zω = Param

println("="^60)
println("Joint (Z,ω) Process Example")
println("="^60)
println()
println("Individual Z process: N=$(ObjGrid_Z.N)")
println("Individual ω process: N=$(ObjGrid_ω.N)")
println("Joint (Z,ω) process: N=$(Grid_Zω.N)")
println()

# ============================================================================
# Example 1: Accessing grid values
# ============================================================================
println("Example 1: Accessing grid values")
println("="^60)

# The joint grid stores values as a 2×N matrix
# Grid_Zω.Values[1, izω] = z value
# Grid_Zω.Values[2, izω] = ω value

izω = 3
z_val, ω_val = Grid_Zω.Values[:, izω]
println("Joint index izω=$izω:")
println("  z = $z_val")
println("  ω = $ω_val")
println()

# Or use helper function
z_val2, ω_val2 = get_zω_values(izω, Grid_Zω)
println("Using helper function:")
println("  z = $z_val2")
println("  ω = $ω_val2")
println()

# ============================================================================
# Example 2: Converting between indices
# ============================================================================
println("Example 2: Index conversion")
println("="^60)

iz, iω = 2, 3
izω = get_joint_index(iz, iω, Grid_Zω.N2)
println("Separate indices (iz=$iz, iω=$iω) → Joint index: izω=$izω")

iz_back, iω_back = get_separate_indices(izω, Grid_Zω.N2)
println("Joint index izω=$izω → Separate indices: (iz=$iz_back, iω=$iω_back)")
println()

# ============================================================================
# Example 3: Loop structures - OLD vs NEW
# ============================================================================
println("Example 3: Loop structure comparison")
println("="^60)

# OLD: Nested loops
println("\nOLD approach (nested loops):")
count_old = 0
for iz in 1:ObjGrid_Z.N
    for iω in 1:ObjGrid_ω.N
        z = ObjGrid_Z.Values[iz]
        ω = ObjGrid_ω.Values[iω]
        count_old += 1
    end
end
println("  Total iterations: $count_old")

# NEW: Single loop over joint grid
println("\nNEW approach (single loop):")
count_new = 0
for izω in 1:Grid_Zω.N
    z, ω = Grid_Zω.Values[:, izω]
    # Or: z, ω = get_zω_values(izω, Grid_Zω)
    count_new += 1
end
println("  Total iterations: $count_new")
println()

# ============================================================================
# Example 4: Transition probabilities - OLD vs NEW
# ============================================================================
println("Example 4: Transition probability lookup")
println("="^60)

# OLD approach: Product of separate probabilities
function old_expectation(iz, iω, values, Param)
    @unpack ObjGrid_Z, ObjGrid_ω = Param
    expectation = 0.0
    for iz′ in 1:ObjGrid_Z.N
        for iω′ in 1:ObjGrid_ω.N
            prob_zω = ObjGrid_Z.Prob[iz, iz′] * ObjGrid_ω.Prob[iω, iω′]
            izω′ = get_joint_index(iz′, iω′, ObjGrid_ω.N)
            expectation += prob_zω * values[izω′]
        end
    end
    return expectation
end

# NEW approach: Direct lookup from joint process
function new_expectation(izω, values, Grid_Zω)
    return expectation_over_zω(values, izω, Grid_Zω)
end

# Test with random values
test_values = rand(Grid_Zω.N)
iz_test, iω_test = 2, 3
izω_test = get_joint_index(iz_test, iω_test, Grid_Zω.N2)

old_result = old_expectation(iz_test, iω_test, test_values, Param)
new_result = new_expectation(izω_test, test_values, Grid_Zω)

println("OLD approach result: $old_result")
println("NEW approach result: $new_result")
println("Difference: $(abs(old_result - new_result))")
println()

# ============================================================================
# Example 5: Array dimensions
# ============================================================================
println("Example 5: Array dimensions")
println("="^60)

Nk, Nx = 20, 2

# OLD: Separate (z, ω) dimensions
K_old = zeros(Nk, Nx, ObjGrid_Z.N, ObjGrid_ω.N)
println("OLD array shape: $(size(K_old)) = (Nk=$Nk, Nx=$Nx, Nz=$(ObjGrid_Z.N), Nω=$(ObjGrid_ω.N))")
println("OLD total elements: $(length(K_old))")

# NEW: Joint (zω) dimension  
K_new = zeros(Nk, Nx, Grid_Zω.N)
println("\nNEW array shape: $(size(K_new)) = (Nk=$Nk, Nx=$Nx, Nzω=$(Grid_Zω.N))")
println("NEW total elements: $(length(K_new))")
println()

# ============================================================================
# Example 6: Vectorized expectation computation
# ============================================================================
println("Example 6: Vectorized expectation")
println("="^60)

# Create test value function: (Nk, Nx, Nzω)
V0 = rand(Nk, Nx, Grid_Zω.N)

println("Computing expectations for all states...")
@time EV = compute_ev_vectorized(V0, Grid_Zω)
println("Result shape: $(size(EV))")
println()

# ============================================================================
# Summary
# ============================================================================
println("="^60)
println("SUMMARY")
println("="^60)
println("\nKey improvements:")
println("  1. Single loop: for izω in 1:Nzω instead of nested loops")
println("  2. Direct access: Grid_Zω.Values[:, izω] returns [z, ω]")
println("  3. Single prob lookup: Grid_Zω.Prob[izω, jzω]")
println("  4. Vectorized expectations: matrix multiplication")
println("  5. Cleaner code: fewer indices to track")
println()
println("Array dimensions:")
println("  OLD: (Nk, Nx, Nz, Nω) with $(Nk*Nx*ObjGrid_Z.N*ObjGrid_ω.N) elements")
println("  NEW: (Nk, Nx, Nzω) with $(Nk*Nx*Grid_Zω.N) elements")
println()
println("Key functions:")
println("  - get_joint_index(iz, iω, Nω)")
println("  - get_separate_indices(izω, Nω)")
println("  - get_zω_values(izω, Grid_Zω)")
println("  - expectation_over_zω(values, izω, Grid_Zω)")
println("  - compute_ev_vectorized(V0, Grid_Zω)")
println()
println("="^60)
