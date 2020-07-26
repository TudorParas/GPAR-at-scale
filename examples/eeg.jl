using Plots, CSV
using GPARatScale, Stheno

using Suppressor  # for supressing plot warnings

PLOTTING_SAMPLES = 1

train_data = CSV.read(joinpath(@__DIR__, "datasets\\eeg\\eeg_train.csv"))
test_data = CSV.read(joinpath(@__DIR__, "datasets\\eeg\\eeg_test.csv"))

# Split training data into columns
train_time = train_data[:, 1]
train_f3 = train_data[:, 2]
train_f4 = train_data[:, 3]
train_f5 = train_data[:, 4]
train_f6 = train_data[:, 5]
train_fz = train_data[:, 6]
train_f1 = train_data[:, 7]
train_f2 = train_data[:, 8]
# Split test data into columns
test_time = test_data[:, 1]
test_fz = test_data[:, 2]
test_f1 = test_data[:, 3]
test_f2 = test_data[:, 4]

# Range for which we have data for fz, f1, and f2
data_range = 1:156
test_range = 157:256
# Train IGPs
fz_gp_post = create_optim_gp_post(
    train_time[data_range],
    convert(Array{Float32}, train_fz[data_range]),
    kernel_structure = Matern52();
    # i_log_l=-2.0, i_log_process_var=2.0, i_log_noise_sigma=-2.0,
    debug = true,
)
f1_gp_post = create_optim_gp_post(
    train_time[data_range],
    convert(Array{Float32}, train_f1[data_range]),
    kernel_structure = Matern52();
    # i_log_l=-2.0, i_log_process_var=2.0, i_log_noise_sigma=-2.0,
    debug = true,
)
f2_gp_post = create_optim_gp_post(
    train_time[data_range],
    convert(Array{Float32}, train_f2[data_range]),
    kernel_structure = Matern12();
    i_log_l=-2.0, i_log_process_var=1.0, i_log_noise_sigma=-3.0,
    debug = true,
)

# Train GPAR models
fz_gpar_post = create_optim_gpar_post(
    [
        train_time[data_range], train_f3[data_range], train_f4[data_range],
        train_f5[data_range], train_f6[data_range]
    ],
    convert(Array{Float32}, train_fz[data_range]);
    time_kernel = Matern52(),
    out_kernel = Matern52(),
    i_log_time_l=-3.0, i_log_time_var=1.0, i_log_out_l=6.0,
    i_log_out_var=4.0, i_log_noise_sigma=-2.0,
    multi_input=true,
    debug = true,
)
f1_gpar_post = create_optim_gpar_post(
    [
        train_time[data_range], train_f3[data_range], train_f4[data_range],
        train_f5[data_range], train_f6[data_range], train_fz[data_range]
    ],
    convert(Array{Float32}, train_f1[data_range]);
    time_kernel = Matern52(),
    out_kernel = Matern52(),
    # i_log_time_l=-3.0, i_log_time_var=1.0, i_log_out_l=6.0,
    # i_log_out_var=4.0, i_log_noise_sigma=-2.0,
    multi_input=true,
    debug = true,
)
f2_gpar_post = create_optim_gpar_post(
    [
        train_time[data_range], train_f3[data_range], train_f4[data_range],
        train_f5[data_range], train_f6[data_range], train_fz[data_range],
        train_f1[data_range]
    ],
    convert(Array{Float32}, train_f2[data_range]);
    time_kernel = Matern52(),
    out_kernel = Matern52(),
    # i_log_time_l=-3.0, i_log_time_var=1.0, i_log_out_l=6.0,
    # i_log_out_var=4.0, i_log_noise_sigma=-2.0,
    multi_input=true,
    debug = true,
)


# @suppress begin
# PLOTTING
gr();
overall_plot = plot(layout = (3, 1), xlabel="time", legend = false, size=(1000, 500));

ylabel!("fz", subplot=1)
ylabel!("f1", subplot=2)
ylabel!("f2", subplot=3)

# DATA PLOTS
function plot_data(plot_ref, train_y, test_y; xlimit=[0.35, 1], ylimit=nothing)
    scatter!(
        plot_ref,
        train_time[data_range],
        train_y[data_range],
        color = :orange,
        markersize = 2.3,
        markeralpha = 0.8,
        markershape = :circle,
        xlims = xlimit,
        ylims = ylimit,
        label = "FZ train",
    )
    scatter!(
        plot_ref,
        test_time,
        test_y,
        color = :orange,
        markersize = 1.8,
        markeralpha = 0.8,
        markershape = :star8,
        label = "FZ test",
    )
end
# Plot fz
plot_data(overall_plot[1], train_fz, test_fz, ylimit=[-15, 5])
# Plot f1
plot_data(overall_plot[2], train_f1, test_f1, ylimit=[-15, 5])
# Plot f2
plot_data(overall_plot[3], train_f2, test_f2, ylimit=[-15, 7])

# # PLOT IGP
plot!(
    overall_plot[1],
    fz_gp_post(train_time),
    samples = PLOTTING_SAMPLES,
    color = :green,
    fillalpha = 0.5,
    linealpha = 1,
    ylims = [-15, 5]
)
plot!(
    overall_plot[2],
    f1_gp_post(train_time),
    samples = PLOTTING_SAMPLES,
    color = :green,
    fillalpha = 0.3,
    linealpha = 1,
    ylims = [-15, 5]
)
plot!(
    overall_plot[3],
    f2_gp_post(train_time),
    samples = PLOTTING_SAMPLES,
    color = :green,
    fillalpha = 0.3,
    linealpha = 1,
    ylims = [-15, 7]
)


# PLOT GPAR
function plot_gpar(plot_ref, means, stds; ylimits=nothing, standard_devs=3, color=:blue)
    # Helper functions for plotting mean of GPAR and standard deviation
    # Plot mean
    plot!(plot_ref, train_time, means, color = color, linealpha = 1)
    plot!(plot_ref, train_time, [means means];
    linewidth=0.0,
    linecolor=color,
    fillrange=[means .- standard_devs .* stds, means .+ standard_devs * stds],
    fillalpha=0.3,
    fillcolor=color,
    ylims = ylimits,
    );
end
# FZ
fz_input = to_ColVecs(
    [
        train_time, train_f3,  train_f4, train_f5, train_f6
    ])
fz_marginals = marginals(fz_gpar_post(fz_input))

fz_means = [f.μ for f in fz_marginals]
fz_stds = [f.σ for f in fz_marginals]
plot_gpar(overall_plot[1], fz_means, fz_stds, ylimits=[-15, 5])
# F1
f1_input = to_ColVecs(
    [
        train_time, train_f3,  train_f4, train_f5, train_f6, fz_means
    ])
f1_marginals = marginals(f1_gpar_post(f1_input))

f1_means = [f.μ for f in f1_marginals]
f1_stds = [f.σ for f in f1_marginals]
plot_gpar(overall_plot[2], f1_means, f1_stds, ylimits=[-15, 5])
# F2
f2_input = to_ColVecs(
    [
        train_time, train_f3,  train_f4, train_f5, train_f6, fz_means, f1_means
    ])
f2_marginals = marginals(f2_gpar_post(f2_input))

f2_means = [f.μ for f in f2_marginals]
f2_stds = [f.σ for f in f2_marginals]
plot_gpar(overall_plot[3], f2_means, f2_stds, ylimits=[-15, 7])

# Use the scaled GPAR
s_fz_means, s_fz_stds = get_gpar_scaled_predictions(
    [
        train_f3[data_range], train_f4[data_range],
        train_f5[data_range], train_f6[data_range]
    ],
    [
        train_f3[data_range], train_f4[data_range],
        train_f5[data_range], train_f6[data_range]
    ],
    train_time[data_range],
    convert(Array{Float64}, train_fz[data_range]),
    # Inference Info
    train_time,
    [
        train_f3,  train_f4, train_f5, train_f6
    ],
    optimization_time_limit = 3.0,
    # i_log_time_l=-3.0, i_log_time_var=1.0, i_log_out_l=6.0,
    # i_log_out_var=4.0, i_log_noise_sigma=-2.0,
    debug = true,
)
plot_gpar(overall_plot[1], s_fz_means, s_fz_stds, ylimits=[-15, 5]; color=:black)

s_f1_means, s_f1_stds = get_gpar_scaled_predictions(
    [
        train_f3[data_range], train_f4[data_range],
        train_f5[data_range], train_f6[data_range], train_fz[data_range]
    ],
    [
        train_f3[data_range], train_f4[data_range],
        train_f5[data_range], train_f6[data_range], train_fz[data_range]
    ],
    train_time[data_range],
    convert(Array{Float64}, train_f1[data_range]),
    # Inference Info
    train_time,
    [
        train_f3,  train_f4, train_f5, train_f6, fz_means
    ],
    optimization_time_limit = 3.0,
    # i_log_time_l=-3.0, i_log_time_var=1.0, i_log_out_l=6.0,
    # i_log_out_var=4.0, i_log_noise_sigma=-2.0,
    debug = true,
)
plot_gpar(overall_plot[2], s_f1_means, s_f1_stds, ylimits=[-15, 5]; color=:black)

s_f2_means, s_f2_stds = get_gpar_scaled_predictions(
    [
        train_f3[data_range], train_f4[data_range],
        train_f5[data_range], train_f6[data_range], train_fz[data_range],
        train_f1[data_range]
    ],
    [
        train_f3[data_range], train_f4[data_range],
        train_f5[data_range], train_f6[data_range], train_fz[data_range],
        train_f1[data_range]
    ],
    train_time[data_range],
    convert(Array{Float64}, train_f2[data_range]),
    # Inference Info
    train_time,
    [
        train_f3,  train_f4, train_f5, train_f6, fz_means, f1_means
    ],
    optimization_time_limit = 3.0,
    # i_log_time_l=-3.0, i_log_time_var=1.0, i_log_out_l=6.0,
    # i_log_out_var=4.0, i_log_noise_sigma=-2.0,
    debug = true,
)
plot_gpar(overall_plot[3], s_f2_means, s_f2_stds, ylimits=[-15, 7]; color=:black)

display(overall_plot)
# end
