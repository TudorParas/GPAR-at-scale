using Stheno, Random
using Optim
using Zygote: gradient

using IJulia
IJulia.installkernel("Julia nodeps", "--depwarn=no") # supress warnings

include("util.jl")
"""
Script including utilities to optimize the hyperparameters of your GP.
"""

"""
Compute the optimised posterior for a EQ kenel GP.

input_locations: array of length L
outputs: array of length L

# Examples:
'''julia-repl
julia> create_optim_gp_post([0.5, 0.2], [-2, 2])

In this example we have the mappings:
0.5 -> -2
0.2 -> 2
"""
function create_optim_gp_post(
    input_locations,
    outputs;
    kernel_structure::Kernel = EQ(),
    debug::Bool = false,
)
    # Helper function used to compute negative log marignal lik for params
    function nlml(params)
        l, process_var, noise_sigma = unpack_gp(params)
        curr_kernel = Stheno.kernel(kernel_structure; l = l, s = process_var^2)
        f = GP(curr_kernel, GPC())

        return -logpdf(f(input_locations, noise_sigma^2), outputs)
    end
    # Optimize the parameters
    params = randn(3)
    results = Optim.optimize(nlml, params, NelderMead();)
    opt_l, opt_process_var, opt_noise_sigma = unpack_gp(results.minimizer)
    if debug
        println()
        println("Optimum L: $(opt_l) ")
        println("Optimum noise: $(opt_noise_sigma)")
        println("Optimum Process Variance: $(opt_process_var)")
    end
    gp_kernel = kernel(kernel_structure, l = opt_l, s = opt_process_var^2)
    gp = GP(gp_kernel, GPC())
    gp_post = gp | (gp(input_locations, opt_noise_sigma^2) ← outputs)

    return gp_post
end

"""
Compute the optimised posterior for a EQ kenel GPAR.

time_input: array of length L
prev_outputs: array of arrays of length L
outputs: array of length L

# Examples:
'''julia-repl
julia> create_optim_gpar_post([1, 2], [[0.5, 0.2], [4.2, 4.5]], [-2, 2])

In this example we have the mappings:
(1, 0.5, 4.2) -> -2
(2, 0.2, 4.5) -> 2
"""
function create_optim_gpar_post(
    input_locations,
    outputs;
    time_kernel::Kernel = EQ(),
    out_kernel::Kernel = EQ(),
    multi_input::Bool = true,  # False if it is the first gpar kernel
    debug::Bool = false,
)
    if !multi_input  # same case as usual gp
        return create_optim_gp_post(
            input_locations,
            outputs,
            kernel_structure = time_kernel;
            debug = debug,
        )
    end
    # Otherwise create an optimised GPAR
    input_length = length(input_locations)
    input_locations = to_ColVecs(input_locations)  # transform into ColVecs
    # Helper function that strings together the GPAR kernel
    function create_gpar_kernel(time_l, time_var, out_l, out_var)
        time_mask = get_time_mask(input_length)
        masked_time_kernel = stretch(time_kernel, time_mask)
        scaled_time_kernel =
            kernel(masked_time_kernel, l = time_l, s = time_var^2)

        out_mask = get_output_mask(input_length)
        masked_out_kernel = stretch(out_kernel, out_mask)
        scaled_out_kernel = kernel(masked_out_kernel, l = out_l, s = out_var^2)

        final_kernel = scaled_time_kernel + scaled_out_kernel
        return final_kernel
    end

    # Helper function used to compute negative log marignal lik for params
    function nlml(params)
        time_l, time_var, out_l, out_var, noise_sigma = unpack_gpar(params)
        kernel = create_gpar_kernel(time_l, time_var, out_l, out_var)
        f = GP(kernel, GPC())

        return -logpdf(f(input_locations, noise_sigma^2), outputs)
    end
    # Optimize the parameters
    params = randn(5)
    results = Optim.optimize(nlml, params, NelderMead())

    opt_time_l, opt_time_var, opt_out_l, opt_out_var, opt_noise_sigma =
        unpack_gpar(results.minimizer)
    if debug
        println()
        println("Optimum time L: $(opt_time_l) ")
        println("Optimum time var: $(opt_time_var)")
        println("Optimum outputs l: $(opt_out_l)")
        println("Optimum outputs l: $(opt_out_var)")
        println("Optimum Noise std: $(opt_noise_sigma)")
    end
    gpar_kernel =
        create_gpar_kernel(opt_time_l, opt_time_var, opt_out_l, opt_out_var)
    gpar = GP(gpar_kernel, GPC())
    gpar_post = gpar | (gpar(input_locations, opt_noise_sigma^2) ← outputs)

    return gpar_post
end
