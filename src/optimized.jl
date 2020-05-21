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

function unpack_eq(params)
    # Unpack the parameters for a GP with EQ kernel
    l = exp(params[1]) + 1e-6
    noise_sigma = exp(params[2]) + 1e-6

    return l, noise_sigma
end

"""
Compute the optimised posterior for a EQ kenel GP.

input_locations: array of arrays of length L
outputs: array of length L

# Examples:
'''julia-repl
julia> create_optim_gp_post([[0.5, 0.2], [4.2, 4.5]], [-2, 2])

In this example we have the mappings:
(0.5, 4.2) -> -2
(0.2, 4.5) -> 2
"""
function create_optim_gp_post(
    input_locations,
    outputs;
    multi_input::Bool = false,
    debug::Bool = false,
)
    # If multi-dimnesional inputs then use colvecs
    if multi_input
        input_locations = to_ColVecs(input_locations)
    end
    # Helper function used to compute negative log marignal lik for params
    function nlml(params)
        l, noise_sigma = unpack_eq(params)
        kernel = stretch(EQ(), l)
        f = GP(kernel, GPC())

        return -logpdf(f(input_locations, noise_sigma^2), outputs)
    end
    # Optimize the parameters
    params = randn(2)
    results = Optim.optimize(
        nlml,
        params -> gradient(nlml, params)[1],
        params,
        BFGS();
        inplace = false,
    )
    opt_l, opt_noise_sigma = unpack_eq(results.minimizer)
    if debug
        println()
        println("Optimum L: $(opt_l) ")
        println("Optimum noise: $(opt_noise_sigma)")
    end

    gp = GP(stretch(EQ(), opt_l), GPC())
    gp_post = gp | (gp(input_locations, opt_noise_sigma^2) ← outputs)

    return gp_post
end

f1_gp_post = create_optim_gp_post(x, y_obs[1])
f2_gp_post = create_optim_gp_post(x, y_obs[2])
f3_gp_post = create_optim_gp_post(x, y_obs[3])

f1_gpar_post = create_optim_gp_post(x, y_obs[1], debug = true)
f2_gpar_post = create_optim_gp_post(
    [x, y_obs[1]],
    y_obs[2],
    multi_input = true,
    debug = true,
)
f3_gpar_post = create_optim_gp_post(
    [x, y_obs[1], y_obs[2]],
    y_obs[3],
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
