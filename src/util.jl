using Stheno

"""
Transform the input space into a ColVecs for using GPAR
"""
function to_ColVecs(inputs)
    # Inputs is an array of 1-D vectors
    concatted = hcat(inputs...)
    # Transpose so that we work with columns
    tranposed = transpose(concatted)
    return ColVecs(tranposed)
end

"""
Functions to unpack positive hyperparameters. Used for optimization purposes.
"""
function unpack_gp(params)
    # Unpack the parameters for a GP with EQ kernel
    l = exp(params[1]) + 1e-3
    process_var = exp(params[2]) + 1e-3
    noise_sigma = exp(params[3]) + 1e-3

    return l, process_var, noise_sigma
end

function unpack_gpar(params)
    # Unpack the parameters for a GPAR with EQ kernel
    time_l = exp(params[1]) + 1e-3
    time_var = exp(params[2]) + 1e-3
    out_l = exp(params[3]) + 1e-3
    out_var = exp(params[4]) + 1e-3

    noise_sigma = exp(params[5]) + 1e-3

    return time_l, time_var, out_l, out_var, noise_sigma
end

"""
The masks are meant to be used toghether when constructing a GPAR kernel so
that k((t, y), (t', y')) = k̲₁(t, t') + k₂(y, y')

Example:

xs = to_ColVecs(
[[-0.08, 0.39, 1.31],
 [-1.60, 0.58, -2.34],
 [-1.71, -0.16, 1.26]])
xs_no_first = to_ColVecs(
[[-1.60, 0.58, -2.34],
[-1.71, -0.16, 1.26]])
xs_only_first = to_ColVecs([[-0.08, 0.39, 1.31]])

ys = to_ColVecs(
[[1.82, 1.06, 0.54],
 [0.14, 1.02, -0.38],
 [-1.45, -0.007, 0.006]])
ys_no_first = to_ColVecs(
[[0.14, 1.02, -0.38],
[-1.45, -0.007, 0.006]])
ys_only_first = to_ColVecs([[1.82, 1.06, 0.54]])
vanilla_kernel = EQ()

time_mask = get_time_mask(3)  # 3 is the number of vals in an input vector
time_kernel = stretch(EQ(), time_mask)

# These two should be the same
time_K = pairwise(time_kernel, xs, ys)
time_true_K = pairwise(vanilla_kernel, xs_only_first, ys_only_first)
# Compare the two results

out_mask = get_output_mask(3)
out_kernel = stretch(EQ(), out_mask)

# These two should be the same
out_K = pairwise(out_kernel, xs, ys)
out_true_K = pairwise(vanilla_kernel, xs_no_first, ys_no_first)
"""


"""
Return mask used to only select time features from inputs.
"""
function get_time_mask(input_length)
    time_mask  = zeros(input_length)
    time_mask[1] = 1  # only select the first (time) element
    return time_mask
end

"""
Return mask used to only select previous output features from inputs.
"""
function get_output_mask(input_length)
    if input_length <= 1
        throw(DomainError(input_length, "Input length must be integer greater than 1"))
    end
    out_mask = zeros(input_length - 1, input_length)
    for row in 1:(input_length - 1)
        out_mask[row, row + 1] = 1  # select this feature
    end
    return out_mask
end
