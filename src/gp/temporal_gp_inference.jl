using Stheno
using TemporalGPs: to_sde, smooth, SArrayStorage
using Optim
using Zygote: gradient
using Random

using GPARatScale
export create_lgssm, get_sde_predictions

"""
Create LGSSM for the latent locations and the given observation noise.
The user can specify a noise vector, in which case it overrides the noise from
the parameters.
"""
function create_lgssm(
    latent_locations,
    l,
    process_var,
    noise_sigma,
    kernel_structure::Kernel;
    sde_storage::SArrayStorage = SArrayStorage(Float64),
    noise_vector = nothing,
    debug::Bool = false,
)
    # Transform the gp into a  Linear Gaussian State Space model (LGSSM)
    # This is done by indexing into the LTISDE.
    # Generate the GP
    gp_kernel = kernel(kernel_structure, l = l, s = process_var^2)
    gp = GP(gp_kernel, GPC())
    # Transform the gp into an LTISDE
    gp_sde = to_sde(gp, sde_storage)
    if isnothing(noise_vector)
        lgssm = gp_sde(latent_locations, noise_sigma^2)
        return lgssm
    else
        lgssm = gp_sde(latent_locations, noise_vector)
        return lgssm
    end
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
    # Initial log param values which will be put in the optimization
    i_log_time_l=nothing, i_log_time_var=nothing, i_log_noise_sigma=nothing,
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

    # Helper function used to compute negative log marignal lik for params
    function nlml(params)
        l, process_var, noise_sigma = unpack_gp(params)
        lgssm = create_lgssm(
            data_locations,
            l,
            process_var,
            noise_sigma,
            kernel_structure,
        )
        return -logpdf(lgssm, data_outputs)
    end
    # Optimize the parameters of the GP
    params = parse_initial_gp_params(i_log_time_l, i_log_time_var, i_log_noise_sigma)
    results = Optim.optimize(nlml, params, NelderMead())
    opt_l, opt_process_var, opt_noise_sigma = unpack_gp(results.minimizer)
    if debug
        println("Finished optimizing parameters:")
        println("\tOptimum L: $(opt_l) ")
        println("\tOptimum Process Variance: $(opt_process_var)")
        println("\tOptimum noise: $(opt_noise_sigma)")
        println()
    end
    # Create the noise vector passed into the creation of the LGSSM.
    # We assume (almost) infinite noise for the output locations
    noise_vector = vcat(
        repeat([opt_noise_sigma^2], length(data_locations)),
        repeat([1e10], length(output_locations)),
    )
    s_noise_vector = noise_vector[sorting_perm]

    opt_lgssm = create_lgssm(
        s_latent_locations,
        opt_l,
        opt_process_var,
        opt_noise_sigma,
        kernel_structure,
        noise_vector = s_noise_vector,
        debug = debug,
    )
    # Get the marginal posteriors for all observations
    _, y_smooth, _ = smooth(opt_lgssm, s_outputs)

    output_observations =
        y_smooth[reverse_perm][length(data_outputs)+1:length(y_smooth)]
    return opt_lgssm, output_observations
end
