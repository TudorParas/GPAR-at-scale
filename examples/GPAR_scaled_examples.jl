using GPARatScale
using Stheno
using Plots

function small_synthetic_dataset()
    # Data generation and preprocessing
    x, y_obs, x_true, y_true = generate_small_dataset()
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
    lgssm1, y1_out = get_sde_predictions(
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
    every_third = range(1; stop = length(x), step = 3)
    pseudo_y1 = y1[every_third]

    println("Generating Y2 scaled GPAR")
    y2_out = get_gpar_scaled_predictions(
        [y1],
        [pseudo_y1],
        time_loc,
        y2,  # outputs
        # Inference information
        test_time_loc,  # Time locations at which we do inference
        [y1_out]
        )  # Input locations from prev outputs

    # Predictions for y3
    println("Generating Y3 scaled GPAR")
    dim1 = range(minimum(y1), stop=maximum(y1), length=5)
    dim2 = range(minimum(y2_out), stop=maximum(y2_out), length=5)
    pseudo_y3 = vec([collect(i) for i in Iterators.product(dim1, dim2)])
    pseudo_y3 = ColVecs(hcat(pseudo_y3...))

    y3_out = get_gpar_scaled_predictions(
        [y1, y2],
        pseudo_y3,
        time_loc,
        y3,  # outputs
        # Inference information
        test_time_loc,  # Time locations at which we do inference
        [y1_out, y2_out]
        )  # Input locations from prev outputs

    # Plotting
    gr()
    overall_plot = plot(layout = (3, 1), legend = false);

    scatter!(overall_plot[1], time_loc, y1,color = :black)
    plot!(overall_plot[1], test_time_loc, test_y1, color = :orange)
    plot!(overall_plot[1], test_time_loc, y1_out, color= :blue)

    scatter!(overall_plot[2], time_loc, y2,color = :black)
    plot!(overall_plot[2], test_time_loc, test_y2, color = :orange)
    plot!(overall_plot[2], test_time_loc, y2_out, color= :blue)

    scatter!(overall_plot[3], time_loc, y3, color = :black)
    plot!(overall_plot[3], test_time_loc, test_y3, color = :orange)
    plot!(overall_plot[3], test_time_loc, y3_out, color= :blue)

    display(overall_plot)

end


function big_synthetic_dataset()
    # Data generation and preprocessing
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

    println("Generating Y2 scaled GPAR")
    y2_out, y2_std = get_gpar_scaled_predictions(
        [y1],
        [pseudo_y1],
        time_loc,
        y2,  # outputs
        # Inference information
        test_time_loc,  # Time locations at which we do inference
        [y1_out];
        optimization_time_limit = 170.0,
        debug=true
        )  # Input locations from prev outputs

    # Predictions for y3
    dim1 = range(minimum(test_y1), stop=maximum(test_y1), length=13)
    dim2 = range(minimum(test_y2), stop=maximum(test_y2), length=13)
    pseudo_y3 = vec([collect(i) for i in Iterators.product(dim1, dim2)])
    pseudo_y3 = ColVecs(hcat(pseudo_y3...))

    println("Generating Y3 temporal GP predictions")
    _, y3_out_temporal = get_sde_predictions(
        x,
        y3,
        x_true;
        kernel_structure = Matern52(),
        i_log_time_l=-3,
        i_log_time_var=0.2,
        i_log_noise_sigma=-10,
        debug = true,
    )
    y3_out_temporal_mean = [f.m[1] for f in y3_out_temporal]
    y3_out_temporal_std = [f.P[1] for f in y3_out_temporal]

    println("Generating Y3 scaled GPAR")
    y3_out, y3_std = get_gpar_scaled_predictions(
        [y1, y2],
        pseudo_y3,
        time_loc,
        y3,  # outputs
        # Inference information
        test_time_loc,  # Time locations at which we do inference
        [y1_out, y2_out];
        optimization_time_limit = 250.0,
        debug=true
        )  # Input locations from prev outputs

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

    # scatter!(overall_plot[3], time_loc, y3, color = :black, alpha=0.05)
    plot!(overall_plot[3], test_time_loc, test_y3, color = :orange)
    # plot_result(overall_plot[3], y3_out_temporal_mean, y3_out_temporal_std, color = :green)
    plot_result(overall_plot[3], y3_out, y3_std, color = :blue)
    # plot!(overall_plot[3], test_time_loc, y3_out_temporal, color = :green)
    # plot!(overall_plot[3], test_time_loc, y3_out, color= :blue)

    display(overall_plot)

end

# small_synthetic_dataset()

big_synthetic_dataset()


# x, y_obs, x_true, y_true = generate_small_dataset()
# # Model X=R²; (x, y1) -> y2; i.e  v=(x, y1), y = y2, and z=(pseudo_x, pseudo_y1)
# time_loc = x
# y1 = y_obs[1]
# y2 = y_obs[2]
# y3 = y_obs[3]
# # Get the testing data
# test_time_loc = x_true
# test_y1 = y_true[1]
# test_y2 = y_true[2]
# test_y3 = y_true[3]
#
# # Get predictions for y1
# lgssm1, y1_out = get_sde_predictions(
#     x,
#     y1,
#     x_true;
#     kernel_structure = Matern52(),
#     debug = true,
# )
