"""
Script in which I'll store examples
"""

using Plots

include("data\\toy_data.jl")
include("optimized.jl")

function run_optimized()
    SAMPLES = 1  # samples we take for the GP representation

    x, y_obs, x_true, y_true = generate_small_dataset()
    f1_gp_post = create_optim_gp_post(x, y_obs[1], kernel_structure=Matern52())
    f2_gp_post = create_optim_gp_post(x, y_obs[2], kernel_structure=Matern52())
    f3_gp_post = create_optim_gp_post(x, y_obs[3], kernel_structure=EQ())

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
end

run_optimized()