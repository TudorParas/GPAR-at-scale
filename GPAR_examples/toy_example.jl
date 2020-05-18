using Stheno, Random, Plots
using Distributions

# Where to start and stop input observations
START = 0
STOP_OBS = 1
STOP_TRUE = 1.0  # might be different to see how they generalize

DATA_SAMPLES = 30
TRUE_SAMPLES = 1000 # discretised points used to model true function
NOISE_MU = 0  # mean of noise
NOISE_SIGMA = 0.05
SAMPLES = 1  # samples we take for the GP representation

function generate_data()
    f1 = (x -> -sin.(10 * pi .* (x .+ 1)) ./ (2 .* x .+ 1) - x .^ 4)
    f2 = ((x, y1) -> cos.(y1) .^ 2 + sin.(3 .* x))
    f3 = ((x, y1, y2) -> y2 .* (y1 .^ 2) + 3 * x)
    # Generate True function
    x_true = range(START, STOP_TRUE, length = TRUE_SAMPLES)
    y1_true = f1(x_true)
    y2_true = f2(x_true, y1_true)
    y3_true = f3(x_true, y1_true, y2_true)
    y_true = hcat(y1_true, y2_true, y3_true)

    # Create the noisy normal distribution
    normal_noise = Normal(NOISE_MU, NOISE_SIGMA^2)
    # Add the noise to the observations
    x = range(START, STOP_OBS, length = DATA_SAMPLES)
    y1 = f1(x) + rand(normal_noise, DATA_SAMPLES)
    y2 = f2(x, y1) + rand(normal_noise, DATA_SAMPLES)
    y3 = f3(x, y1, y2) + rand(normal_noise, DATA_SAMPLES)
    y_obs = hcat(y1, y2, y3)

    return x, y_obs, x_true, y_true
end

x, y_obs, x_true, y_true = generate_data();

# Generate the Independent GPs
f1_gp = GP(stretch(EQ(), 10), GPC())
f2_gp = GP(EQ(), GPC())
f3_gp = GP(EQ(), GPC())

f1_gp_posterior = f1_gp | (f1_gp(x, NOISE_SIGMA^2) ← y_obs[:, 1])
f2_gp_posterior = f2_gp | (f2_gp(x, NOISE_SIGMA^2) ← y_obs[:, 2])
f3_gp_posterior = f3_gp | (f3_gp(x, NOISE_SIGMA^2) ← y_obs[:, 3])

# # Attempt to create GP modelling all outputs
# full_gp = GP(EQ() + PerEQ(0.5) , GPC())
# full_gp_post =
#     full_gp |
#     (full_gp(repeat(x, 3), NOISE_SIGMA^2) ← vcat(y_obs[:, 1], y_obs[:, 2], y_obs[:, 3]))

# Create GPAR model
f1_gpar = GP(stretch(EQ(), 10), GPC())
f2_gpar = GP(EQ(), GPC())
f3_gpar = GP(EQ(), GPC())
# Compute the posteriors
f1_gpar_post = f1_gpar | (f1_gpar(x, NOISE_SIGMA^2) ← y_obs[:, 1])
# We represent the input locations as a 2 x 30 matrix
f2_data_locations = ColVecs(transpose(hcat(x, y_obs[:, 1])))

f2_gpar_post =
    f2_gpar | (
        f2_gpar(f2_data_locations, NOISE_SIGMA^2) ← y_obs[:, 2]
    )
# We represent the input locations as a 3 x 30 matrix
f3_data_locations = ColVecs(transpose(hcat(x, y_obs[:, 1], y_obs[:, 2])))
f3_gpar_post =
    f3_gpar | (
        f3_gpar(f3_data_locations, NOISE_SIGMA^2) ← y_obs[:, 3]
    )

# Plotting
plotly();
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
    f1_gp_posterior(x_true),
    samples = SAMPLES,
    color = :green,
    fillalpha = 0.5,
    linealpha = 1,
)
plot!(
    overall_plot[2],
    f2_gp_posterior(x_true),
    samples = SAMPLES,
    color = :green,
    fillalpha = 0.3,
    linealpha = 1,
)
plot!(
    overall_plot[3],
    f3_gp_posterior(x_true),
    samples = SAMPLES,
    color = :green,
    fillalpha = 0.3,
    linealpha = 1,
)

# # Plot multi-output gp
# all_y_mean = mean(full_gp_post(repeat(x_true, 3)))
# plot!(overall_plot[1], x_true, all_y_mean[1:TRUE_SAMPLES], color=:red)
# plot!(overall_plot[2], x_true, all_y_mean[TRUE_SAMPLES + 1 : 2 * TRUE_SAMPLES], color=:red)
# plot!(overall_plot[3], x_true, all_y_mean[2 * TRUE_SAMPLES + 1 : 3 * TRUE_SAMPLES], color=:red)

# Plot GPAR
# First plot is easy
y1_mean_vals = mean(f1_gpar_post(x_true))
plot!(
    overall_plot[1],
    x_true,
    y1_mean_vals,
    color = :blue,
    linealpha = 1,
)
# Use x and mean of y1 to feed into f2_gpar
x_y1_true = ColVecs(transpose(hcat(x_true, y1_mean_vals)))
y2_mean_vals = mean(f2_gpar_post(x_y1_true))
plot!(overall_plot[2], x_true, y2_mean_vals, color = :blue, linealpha = 1)

# Plot f3
# Again, collect x, y1, and y2 to feed as input location to f3
x_y1_y2_true = ColVecs(transpose(hcat(x_true, y1_mean_vals, y2_mean_vals)))
y3_mean_vals = mean(f3_gpar_post(x_y1_y2_true))
plot!(overall_plot[3], x_true, y3_mean_vals, color = :blue, linealpha = 1)
