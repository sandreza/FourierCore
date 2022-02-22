
"""
appropriate_dims(n₁, n₂, N)
# Description
Create an array of size n₁, with the value N at the location n₂, and ones elswhere
# Arguments
- `n₁`: int | size of the array
- `n₂`: int | location of modification
- `N` : int | value at the modification index n2
# Return
A tuple with all 1's except at location n₂, where it is N
"""
function appropriate_dims(n₁, n₂, N)
    y = ones(Int, n₁)
    y[n₂] = N
    return Tuple(y)
end