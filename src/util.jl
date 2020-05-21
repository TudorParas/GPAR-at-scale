using Stheno


"""
Transform the input space into a ColVecs for using GPAR
"""
function to_ColVecs(inputs)
    # Inputs is an array of 1-D vectors
    concatted = hcat(inputs...)
    # Transpose to make it into a format understandable by ColVecs
    tranposed = transpose(concatted)
    return ColVecs(tranposed)
end
