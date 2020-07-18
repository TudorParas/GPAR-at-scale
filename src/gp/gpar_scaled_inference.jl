using Stheno: cholesky
using TemporalGPs: to_sde, smooth, SArrayStorage, decorrelate, posterior_rand
using Stheno
using LinearAlgebra
using Random
using Distributions
using Plots

using GPARatScale

export get_gpar_scaled_predictions


"""
Compute scaled GPAR predictions. This is only useful in input domain has at
least 2 dimensions, one of which is time.

In the case of 1D input (only time), use get_sde_predictions from temporal_gp_inference.jl
"""
function get_gpar_scaled_predictions(
    # Training information
    input_locations, # input locations from previous outputs. Should be array of arrays
    pseudo_input_locations, # pseudo-input locations in the samee domain as v
    time_loc, # the time locations correspondiing to v
    outputs,  # the ouput training data coresp to v
    # Inference information
    inference_time_loc,  # Time locations at which we do inference
    inference_input_locations;  # Input locations from prev outputs
    out_kernel_structure = Matern52(),  # kernel of the GP used on input points
    time_kernel_structure = Matern52(),  # kernel of the GP used on temporal locations
    # Initial log param values. Temporal noise and output noise are the same.
    i_log_time_l=nothing, i_log_time_var=nothing, i_log_out_l=nothing,
    i_log_out_var=nothing, i_log_noise_sigma=nothing,
    optimization_time_limit = 1000.0,
    debug::Bool=false,
    storage = SArrayStorage(Float64), # storage used for TemporalGPs
    )
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

    # Function that does monte carlo sampling for computing ∫p(f* | fₓ, y) * q(fₓ)
    function monte_carlo_f_star(iterations)
        acc = []
        for _ in 1:iterations
            fx = generate_fx_sample()
            # Subtract fx to focus on fₜ
            y_star = outputs_star - fx
            # Compute fₜ by calling decorrelate
            _, y_smooth, _ = smooth(time_lgssm_star, y_star)
            # Retrieve the means of the Gaussian distributions
            y_smooth = [f.m[1] for f in y_smooth]
            # Add back fx to get f*
            f_star = fx + y_smooth
            push!(acc, f_star)
        end
        # Divide by the number of iterations as in Monte Carlo sampling
        return mean(acc), std(acc)

    end

    # TODO: find a way to also return variance.
    inferred_f_star_mean, inferred_f_star_std = monte_carlo_f_star(100)
    # Only get predictions at the inference locations
    inferred_outputs = inferred_f_star_mean[reverse_perm][length(time_loc) + 1:length(inferred_f_star_mean)]
    inferred_stds = inferred_f_star_std[reverse_perm][length(time_loc) + 1:length(inferred_f_star_std)]
    
    return inferred_outputs, inferred_stds
end
"""
Compute the approximate posterior multivariate distribution over the
pseudo-points. This will later be used to sample from q(f)
"""
function compute_q_u(
    input_locations, # input locations from previous outputs. Should be array of arrays
    pseudo_input_locations, # pseudo-input locations in the samee domain as v
    time_loc, # the time locations correspondiing to v
    outputs;  # the ouput training data coresp to v
    out_kernel = Matern52(),  # kernel of the GP used on input points
    time_kernel = Matern52(),  # kernel of the GP used on temporal locations
    temporal_noise_sigma = 0.05,  # TODO: This will need to be optimized
    debug::Bool=false,
    storage = SArrayStorage(Float64), # storage used for TemporalGPs
    )
    # Compute sizes of input locations and pseudo-input locations
    N = length(input_locations)  # lenth of training data
    M = length(pseudo_input_locations)  # length of pseudo-points
    # Compute covariance matrices
    Cfu = pairwise(out_kernel, input_locations, pseudo_input_locations)  # N x M
    Cuu = pairwise(out_kernel, pseudo_input_locations, pseudo_input_locations)  # M x M
    # Do the choleksy decomposition to use for inverses
    U_u = cholesky(Symmetric(Cuu)).U  # M x M array
    L_u = U_u'
    # Create the time LGSSM
    # TODO: unite this with the one in the other func
    time_prior = GP(time_kernel, GPC())
    time_sde = to_sde(time_prior, SArrayStorage(Float64))
    time_lgssm = time_sde(time_loc, temporal_noise_sigma^2)

    # With all the prerequisites done, now work on computing q(u) ~ p(u|y)

    # Compute B_ef. First compute beta_interm = U_y' \ Cfu
    beta_interm = zeros((N, M)) # zero initialization
    # Iterate over the columns of Cfu and call decorrelate to construct beta
    for col_index = 1:M
        col = Cfu[:, col_index]
        # Call decorrelate to compute the specific column in beta
        _, col_beta = decorrelate(time_lgssm, col)
        beta_interm[:, col_index] = col_beta
    end

    B_ef = L_u \ beta_interm'  # M x N array

    # Compute b_y by calling decorrelate.
    # TODO: Do outputs - mean(f) here instead of outputs to handle non-zero mean.
    _, b_y = decorrelate(time_lgssm, outputs)

    # Continue as in the usual DTC case.
    # This is because we no longer have N x N arrays.
    D = B_ef * B_ef' + I  # M x M array. It is Λ_ϵ
    chol_D = cholesky(Symmetric(D))  # low cost decomp to compute Lambda_e
    m_e = chol_D \ (B_ef * b_y)  # M array. It is m_ϵ

    # Now use these to compute the mean and var of q(f)
    q_u = MvNormal(m_e, Symmetric(inv(D)))

    # Also return U_u because we need it later
    return q_u, U_u
end


# Change Cfu to Cf*u ...f -> f*
# qf_mean = mean(f) + Cfu * (U_u \ (m_e))  # N array

# First two terms in Cff hat cancel out because of DTC. Check that
# Cff_hat = Cff - Cfu * (U_u \ (L_u \ Cuf)) +s Cfu * (U_u \ (D \ (L_u \ Cuf))) # N x N

#        Assume these two cancel each other out. Test this, and replace with 0.
# THen you sample
# Cff_hat = Cff - Cfu * inv(U_u)* inv(L_u) * Cuf + Cfu * inv(U_u) * inv(D) * inv(L_u) * Cuf
# q_fx =  MvNormal(qf_mean, Symmetric(Cff_hat))  # Normal distribution from which we sample

# TO generate sample from q_fx* you sample from q(e), then plug that sample in your qf_mean

# SAMPLE FROM qe, plug in qf_mean (in m_e), and use qf_mean as your fx sample
# We can do this bc Cff_hat is low rank (has alsmost all 0 eigenvals)

#  Create code that p(f* | fx, y) using smoothing

#(y - fx) -> posterior rand should do sampling and smoothing
