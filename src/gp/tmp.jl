using GPARatScale
using Stheno
using Plots

using Stheno: cholesky
using TemporalGPs: to_sde, smooth, SArrayStorage, decorrelate, posterior_rand
using Stheno
using LinearAlgebra
using Random
using Distributions

x, y_obs, x_true, y_true = generate_big_dataset()
# Model X=R²; (x, y1) -> y2; i.e  v=(x, y1), y = y2, and z=(pseudo_x, pseudo_y1)
time_loc = x
y1 = y_obs[1]
y2 = y_obs[2]
y3 = y_obs[3]
# Get the testing data
test_time_loc = x_true
test_y1 = y_true[1]
test_y2 = y_true[2]
test_y3 = y_true[3]
println("Generating predictions fo Y1")
# Get predictions for y1
_, y1_out = get_sde_predictions(
    x,
    y1,
    x_true;
    kernel_structure = Matern52(),
    i_log_time_l=-3,
    i_log_time_var=0.2,
    i_log_noise_sigma=-10,
    debug = true,
)
y1_out = [f.m[1] for f in y1_out]

# Predictions for y2.
# Generate u, the pseudo-points, by taking every third element
pseudo_y1 = range(minimum(test_y1); stop = maximum(test_y1), length=120)

println("Generating Y2 temporal GP predictions")
_, y2_out_temporal = get_sde_predictions(
    x,
    y2,
    x_true;
    kernel_structure = Matern52(),
    i_log_time_l=-3,
    i_log_time_var=0.2,
    i_log_noise_sigma=-10,
    debug = true,
)


y2_out_temporal_mean = [f.m[1] for f in y2_out_temporal]
y2_out_temporal_std = [f.P[1] for f in y2_out_temporal]


# Setup data
input_locations = [y1]
pseudo_input_locations = [pseudo_y1]
outputs = y2
inference_time_loc = test_time_loc
inference_input_locations = [test_y1]
out_kernel_structure = Matern52()
time_kernel_structure = Matern52()

i_log_time_l=nothing
i_log_time_var=nothing
i_log_out_l=nothing
i_log_out_var=nothing
i_log_noise_sigma=nothing
optimization_time_limit = 10.0
debug=true
storage = SArrayStorage(Float64)

println("Generating Y2 scaled GPAR")
# y2_out, y2_std = get_gpar_scaled_predictions(
#     [y1],
#     [pseudo_y1],
#     time_loc,
#     y2,  # outputs
#     # Inference information
#     test_time_loc,  # Time locations at which we do inference
#     [test_y1];
#     optimization_time_limit = 170.0,
#     debug=true
#     )  # Input locations from prev outputs

# Hardcoded
input_locations = to_ColVecs(input_locations)
pseudo_input_locations= to_ColVecs(pseudo_input_locations)
inference_input_locations = to_ColVecs(inference_input_locations)
# TODO: Optimize hyperparameters using the DTC objective
println("Starting optimization")
opt_params = get_optim_scaled_gpar_params(
    input_locations,
    pseudo_input_locations,
    time_loc,
    outputs;
    out_kernel = Matern52(),
    time_kernel = Matern52(),
    # Initial log param values. Temporal noise and output noise are the same.
    i_log_time_l=i_log_time_l, i_log_time_var=i_log_time_var, i_log_out_l=i_log_out_l,
    i_log_out_var=i_log_out_var, i_log_noise_sigma=i_log_noise_sigma,
    optimization_time_limit = optimization_time_limit,
    debug=debug,
    storage = SArrayStorage(Float64), # storage used for TemporalGPs
    )
opt_time_l, opt_time_var, opt_out_l, opt_out_var, opt_noise_sigma = opt_params
out_kernel = kernel(out_kernel_structure, l=opt_out_l, s=opt_out_var^2)
time_kernel = kernel(time_kernel_structure, l=opt_time_l, s=opt_time_var^2)
# TODO: look into having different noise values
temporal_noise_sigma = opt_noise_sigma
# Compute q(u) ~ p(u | y).
q_u, U_u = compute_q_u(
    input_locations,
    pseudo_input_locations,
    time_loc,
    outputs;
    out_kernel = out_kernel,  # kernel of the GP used on input points
    time_kernel = time_kernel,  # kernel of the GP used on temporal locations
    temporal_noise_sigma = temporal_noise_sigma,  # TODO: This will need to be optimized
    debug=debug,
    storage = storage, # storage used for TemporalGPs
)
# Concatenate the training and test data for use in the LGSSM
time_loc_concat = vcat(time_loc, inference_time_loc)
# Make sure that input_loc_concat is also ColVecs
input_loc_concat = ColVecs(hcat(input_locations.X, inference_input_locations.X))
# Outputs, which a sentinel value in the places where we make inference.
outputs_concat = vcat(outputs, repeat([0], length(inference_time_loc)))

# Get the permutation for sorting the locations. Need sorted order for LGSSM
sorting_perm = sortperm(time_loc_concat)
reverse_perm = sortperm(sorting_perm)
# Sort the inputs so that time is ascendent (for LGSSM)
time_loc_star = time_loc_concat[sorting_perm]
input_loc_star = input_loc_concat[sorting_perm]
outputs_star = outputs_concat[sorting_perm]
# Code that generates the fₓ sample; i.e. the GP acting on prev outputs.
Cfu_star = pairwise(out_kernel, input_loc_star, pseudo_input_locations)  # Cf*u

function generate_fx_sample()
    # Generate an fx sample by sampling q_u and plugging it in qf_mean.
    # This is possible because Cff_hat is low rank (many 0 val eigenvals)
    m_e = rand(q_u)
    # Also add the prior mean here if nonzero
    fx_sample = Cfu_star * (U_u \ (m_e))
end

# Create the LGSSM. Assume infinite noise at the inference locations.
noise_vector_concat = vcat(
    repeat([temporal_noise_sigma^2], length(time_loc)),
    repeat([1e10], length(inference_time_loc)))  # infinite variance in inference loc
noise_vector_star = noise_vector_concat[sorting_perm]

time_prior = GP(time_kernel, GPC())
time_sde = to_sde(time_prior, SArrayStorage(Float64))
time_lgssm_star = time_sde(time_loc_star, noise_vector_star)

# Baga un posterior rand
rng = MersenneTwister(53)
fx = generate_fx_sample()
# Subtract fx to focus on fₜ
y_star = outputs_star - fx
# Compute fₜ by calling decorrelate
samples = posterior_rand(rng, time_lgssm_star, y_star)

# Function that does monte carlo sampling for computing ∫p(f* | fₓ, y) * q(fₓ)
# function monte_carlo_f_star(iterations)
#     acc = []
#     for _ in 1:iterations
#         fx = generate_fx_sample()
#         # Subtract fx to focus on fₜ
#         y_star = outputs_star - fx
#         # Compute fₜ by calling decorrelate
#         # _, y_smooth, _ = smooth(time_lgssm_star, y_star)
#         samples = posterior_rand(rng, time_lgssm_star, y_star, 5)
#         # Retrieve the means of the Gaussian distributions
#         y_smooth = [f.m[1] for f in y_smooth]
#         # Add back fx to get f*
#         f_star = fx + y_smooth
#         push!(acc, f_star)
#     end
#     # Divide by the number of iterations as in Monte Carlo sampling
#     return mean(acc), std(acc)
#
# end
#
# # TODO: find a way to also return variance.
# inferred_f_star_mean, inferred_f_star_std = monte_carlo_f_star(100)
# # Only get predictions at the inference locations
# inferred_outputs = inferred_f_star_mean[reverse_perm][length(time_loc) + 1:length(inferred_f_star_mean)]
# inferred_stds = inferred_f_star_std[reverse_perm][length(time_loc) + 1:length(inferred_f_star_std)]
#
# y2_out, y2_std  = inferred_outputs, inferred_stds


# Plotting
gr()
overall_plot = plot(layout = (3, 1), legend = false);

function plot_result(plot_ref, posterior_mean, std; ylimits=nothing, standard_devs=3, color=:green)
    # Plot mean
    plot!(plot_ref, x_true, posterior_mean, color = color, linealpha = 1)
    # Plot error bars
    plot!(plot_ref, x_true, [posterior_mean posterior_mean];
    linewidth=0.0,
    linecolor=:green,
    fillrange=[posterior_mean .- standard_devs .* std, posterior_mean .+ standard_devs * std],
    fillalpha=0.3,
    fillcolor=color,
    ylims = ylimits
    );
end

# scatter!(overall_plot[1], time_loc, y1,color = :black, alpha=0.05)
plot!(overall_plot[1], test_time_loc, test_y1, color = :orange)
plot!(overall_plot[1], test_time_loc, y1_out, color = :green)
plot!(overall_plot[1], test_time_loc, y1_out, color= :blue)

# scatter!(overall_plot[2], time_loc, y2,color = :black, alpha=0.05)
plot!(overall_plot[2], test_time_loc, test_y2, color = :orange)
# plot_result(overall_plot[2], y2_out_temporal_mean, y2_out_temporal_std, color = :green)
plot_result(overall_plot[2], y2_out, y2_std, color = :blue)
# plot!(overall_plot[2], test_time_loc, y2_out_temporal, color = :green)
# plot!(overall_plot[2], test_time_loc, y2_out, color= :blue)


display(overall_plot)
