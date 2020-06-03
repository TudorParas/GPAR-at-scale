using Stheno
using TemporalGPs: to_sde, smooth, SArrayStorage
using Plots
using Optim
using Zygote: gradient
using Random

include("data\\toy_data.jl")

function unpack_params(params)
    l = exp(params[1]) + 1e-3
    process_var = exp(params[2]) + 1e-3
    noise_sigma = exp(params[3]) + 1e-3

    return l, process_var, noise_sigma
end

"""
Create an LGSSM conditioned on the observed data and output the marginals of
the observations at the output locations
"""
function get_sde_predictions(
    data_locations,
    data_outputs,
    output_locations;
    kernel_structure::Kernel = Matern52(),
    sde_storage::SArrayStorage = SArrayStorage(Float64),
    debug::Bool = true,
)
    # Concatanete all the input locations
    latent_locations = vcat(data_locations, output_locations)
    # Concatenate the knwon outputs with the outputs we want to infer
    outputs = vcat(data_outputs, repeat([0], length(output_locations)))
    # Get the permutation for sorting the latent vars since TemporalGPs expects
    # a timeseries. Use the permutation for later sorting
    sorting_perm = sortperm(latent_locations)
    reverse_perm = sortperm(sorting_perm)  # used to reverse the sorting
    # Sort the inputs and output so that the input locations are in
    # ascending order. This is needed in order to model a timeseries.
    s_latent_locations = latent_locations[sorting_perm]
    s_outputs = outputs[sorting_perm]
    function create_lgssm(params)
        # Transform the gp into a  Linear Gaussian State Space model (LGSSM)
        # This is done by indexing into the LTISDE.
        l, process_var, noise_sigma = unpack_params(params)
        if debug
            println("Creating LGSSM with parameters\n\tl=$(l)\n\tprocess_var=$(process_var)\n\tnoise_sigma=$(noise_sigma)")
        end
        # Generate the GP
        gp_kernel = kernel(kernel_structure, l = l, s = process_var^2)
        gp = GP(gp_kernel, GPC())
        # Transform the gp into an LTISDE
        gp_sde = to_sde(gp, sde_storage)
        # We assume (almost) infinite noise for the output locations
        noise_vector = vcat(
            repeat([noise_sigma^2], length(data_locations)),
            repeat([1e10], length(output_locations)),
        )
        s_noise_vector = noise_vector[sorting_perm]
        lgssm = gp_sde(s_latent_locations, s_noise_vector)
        return lgssm
    end
    # TODO: Actually run the optimization insttead of jsut hard-coding parameters
    # Helper function used to compute negative log marignal lik for params
    # function nlml(params)
    #     lgssm = create_lgssm(params)
    #     result = -logpdf(lgssm, s_outputs)
    #     print(result)
    #     return result
    # end
    # Optimize the parameters of the GP
    # params = randn(3)
    # results = Optim.optimize(nlml, params, NelderMead();)
    # print(results.minimizer)
    params = [log(10), log(1), log(0.05)]
    opt_lgssm = create_lgssm(params)
    # Get the marginal posteriors for all observations
    _, y_smooth, _ = smooth(opt_lgssm, s_outputs)

    output_observations =
        y_smooth[reverse_perm][length(data_outputs)+1:length(y_smooth)]
    return output_observations
end

x, y_obs, x_true, y_true = generate_big_dataset()

f1_out = get_sde_predictions(
    x,
    y_obs[1],
    x_true;
    kernel_structure = Matern52(),
    debug = true,
)
f2_out = get_sde_predictions(x, y_obs[2], x_true)
f3_out = get_sde_predictions(x, y_obs[3], x_true)
# Plotting
gr();
overall_plot = plot(layout = (3, 1), legend = false);
# Plot data
plot!(overall_plot, x_true, y_true, color = :orange, label = "True")

# Plot posterior mean of the IGP results
f1_posterior_mean = [f.m[1] for f in f1_out]
plot!(overall_plot[1], x_true, f1_posterior_mean, color = :green, linealpha = 1)
f2_posterior_mean = [f.m[1] for f in f2_out]
plot!(overall_plot[2], x_true, f2_posterior_mean, color = :green, linealpha = 1)
f3_posterior_mean = [f.m[1] for f in f3_out]
plot!(overall_plot[3], x_true, f3_posterior_mean, color = :green, linealpha = 1)
