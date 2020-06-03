"""
Script in which I'll store examples
"""

using Plots
using Suppressor

include("data\\toy_data.jl")
include("optimized.jl")
include("temporal_gp_inference.jl")

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


    # Plotting. Supress warnings caused by stheno
    @suppress begin
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
end


function run_temporal_gp_inference()
    x, y_obs, x_true, y_true = generate_big_dataset()

    _, f1_out = get_sde_predictions(
        x,
        y_obs[1],
        x_true;
        kernel_structure = Matern52(),
        debug = true,
    )
    _, f2_out = get_sde_predictions(x, y_obs[2], x_true)
    _, f3_out = get_sde_predictions(x, y_obs[3], x_true)
    # Plotting
    gr();
    overall_plot = plot(layout = (3, 1), legend = false);
    # Plot data
    # # Code used to plot the the magnitude of the observation noise
    # scatter!(overall_plot[1], x, y_obs[1], color=:black, alpha=0.1)
    # scatter!(overall_plot[2], x, y_obs[2], color=:black, alpha=0.1)
    # scatter!(overall_plot[3], x, y_obs[3], color=:black, alpha=0.1)

    plot!(overall_plot, x_true, y_true, color = :orange, label = "True")

    # Plot posterior mean of the IGP results
    f1_posterior_mean = [f.m[1] for f in f1_out]
    plot!(overall_plot[1], x_true, f1_posterior_mean, color = :green, linealpha = 1)
    f2_posterior_mean = [f.m[1] for f in f2_out]
    plot!(overall_plot[2], x_true, f2_posterior_mean, color = :green, linealpha = 1)
    f3_posterior_mean = [f.m[1] for f in f3_out]
    plot!(overall_plot[3], x_true, f3_posterior_mean, color = :green, linealpha = 1)

    display(overall_plot)

    function compute_mser(out_true, out_predict)
        return mean((out_true - out_predict) .^ 2)
    end

    mser1 = compute_mser(y_true[1], f1_posterior_mean)
    mser2 = compute_mser(y_true[2], f2_posterior_mean)
    mser3 = compute_mser(y_true[3], f3_posterior_mean)

    println("MSER 1: $(mser1)")
    println("MSER 2: $(mser2)")
    println("MSER 3: $(mser3)")
end

run_temporal_gp_inference()
