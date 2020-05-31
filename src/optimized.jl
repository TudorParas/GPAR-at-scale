using Stheno, Random, Plots
using Optim
using Zygote: gradient

using IJulia
IJulia.installkernel("Julia nodeps", "--depwarn=no") # supress warnings

include("data\\toy_data.jl")
include("util.jl")
"""
Script including utilities to optimize the hyperparameters of your GP.
"""


SAMPLES = 1  # samples we take for the GP representation

x, y_obs, x_true, y_true = generate_small_dataset()

function unpack_gp(params)
    # Unpack the parameters for a GP with EQ kernel
    l = exp(params[1]) + 1e-3
    process_var = exp(params[2]) + 1e-3
    noise_sigma = exp(params[3]) + 1e-3

    return l, process_var, noise_sigma
end

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

f1_gp_post = create_optim_gp_post(x, y_obs[1], Matern52())
f2_gp_post = create_optim_gp_post(x, y_obs[2], Matern52())
f3_gp_post = create_optim_gp_post(x, y_obs[3], EQ())

f1_gpar_post = create_optim_gpar_post(
    x,
    y_obs[1];
    time_kernel = EQ(),
    multi_input = false,
    debug = true,
)
f2_gpar_post = create_optim_gpar_post(
    [x, y_obs[1]],
    y_obs[2];
    time_kernel = Matern52(),
    out_kernel = EQ(),
    multi_input = true,
    debug = true,
)
f3_gpar_post = create_optim_gpar_post(
    [x, y_obs[1], y_obs[2]],
    y_obs[3];
    time_kernel = Matern52(),
    out_kernel = Matern52(),
    multi_input = true,
    debug = true,
)


# Plotting
gr();
overall_plot = plot(layout = (3, 1), legend = false);
# Plot data
plot!(overall_plot, x_true, y_true, color = :orange, label = "True")
scatter!(
    overall_plot,
    x,
    y_obs,
    color = :black,
    label = "Observations",
    markersize = 3,
    markeralpha = 0.8,
)

# Plot IGP results
plot!(
    overall_plot[1],
    f1_gp_post(x_true),
    samples = SAMPLES,
    color = :green,
    fillalpha = 0.5,
    linealpha = 1,
)
plot!(
    overall_plot[2],
    f2_gp_post(x_true),
    samples = SAMPLES,
    color = :green,
    fillalpha = 0.3,
    linealpha = 1,
)
plot!(
    overall_plot[3],
    f3_gp_post(x_true),
    samples = SAMPLES,
    color = :green,
    fillalpha = 0.3,
    linealpha = 1,
)

# Plot GPAR
# First plot is easy
y1_mean_vals = mean(f1_gpar_post(x_true))
plot!(overall_plot[1], x_true, y1_mean_vals, color = :blue, linealpha = 1)
# Use x and mean of y1 to feed into f2_gpar
x_y1_true = to_ColVecs([x_true, y1_mean_vals])
y2_mean_vals = mean(f2_gpar_post(x_y1_true))
plot!(overall_plot[2], x_true, y2_mean_vals, color = :blue, linealpha = 1)

# Plot f3
# Again, collect x, y1, and y2 to feed as input location to f3
x_y1_y2_true = to_ColVecs([x_true, y1_mean_vals, y2_mean_vals])
y3_mean_vals = mean(f3_gpar_post(x_y1_y2_true))
plot!(overall_plot[3], x_true, y3_mean_vals, color = :blue, linealpha = 1)

display(overall_plot)
