using Stheno: cholesky
using TemporalGPs: to_sde, smooth, SArrayStorage, decorrelate, posterior_rand
using Stheno
using LinearAlgebra
using Random
using Distributions
using Plots

using GPARatScale

function approx_posterior_ref(fx, y, u, noise_matrix)
    U_y = cholesky(Symmetric(noise_matrix)).U
    U = cholesky(Symmetric(cov(u))).U

    B_εf = U' \ (U_y' \ cov(fx, u))'

    b_y = U_y' \ (y - mean(fx))

    D = B_εf * B_εf' + I
    Λ_ε = cholesky(Symmetric(D))

    m_ε = Λ_ε \ (B_εf * b_y)

    return D, m_ε
end

"""
Compute scaled GPAR predictions. This is only useful in input domain has at
least 2 dimensions, one of which is time.
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
    debug::Bool=false,
    storage = SArrayStorage(Float64), # storage used for TemporalGPs
    )
    input_locations = to_ColVecs(input_locations)
    pseudo_input_locations= to_ColVecs(pseudo_input_locations)
    inference_input_locations = to_ColVecs(inference_input_locations)
    # TODO: Optimize hyperparameters using the DTC objective
    out_kernel = out_kernel_structure
    time_kernel = time_kernel_structure

    temporal_noise_sigma = 0.05
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
        acc = zeros(length(time_loc_star))
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
            acc = acc + f_star
        end
        # Divide by the number of iterations as in Monte Carlo sampling
        return acc / iterations

    end

    # TODO: find a way to also return variance.
    inferred_f_star = monte_carlo_f_star(100)
    # Only get predictions at the inference locations
    inferred_outputs = inferred_f_star[reverse_perm][length(time_loc) + 1:length(inferred_f_star)]

    return inferred_outputs
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

# Data generation and preprocessing
x, y_obs, x_true, y_true = generate_small_dataset()
# Model X=R²; (x, y1) -> y2; i.e  v=(x, y1), y = y2, and z=(pseudo_x, pseudo_y1)
time_loc = x
y1 = y_obs[1]
y2 = y_obs[2]
y3 = y_obs[3]
# Generate u, the pseudo-points, by taking every third element
every_third = range(1; stop = length(x), step = 3)
pseudo_y1 = y1[every_third]
# Get the testing data
test_time_loc = x_true
test_y1 = y_true[1]
test_y2 = y_true[2]
test_y3 = y_true[3]

# Run the code



y2_out = get_gpar_scaled_predictions(
    [y1],
    [pseudo_y1],
    time_loc,
    y2,  # outputs
    # Inference information
    test_time_loc,  # Time locations at which we do inference
    [test_y1]
    )  # Input locations from prev outputs

dim1 = range(minimum(y1), stop=maximum(y1), length=5)
dim2 = range(minimum(y2_out), stop=maximum(y2_out), length=5)
pseudo_y3 = vec([collect(i) for i in Iterators.product(dim1, dim2)])
pseudo_y3 = ColVecs(hcat(pseudo_y3...))

# input_locations = to_ColVecs([y1, y2])
# pseudo_input_locations= to_ColVecs(pseudo_y3)
# inference_input_locations = to_ColVecs([test_y1, y2_out])
#
# time_loc_concat = vcat(time_loc, test_time_loc)
# # Make sure that input_loc_concat is also ColVecs
# input_loc_concat = vcat(input_locations, inference_input_locations)
# # input_loc_concat = [arr[1] for arr in input_loc_concat]
# # input_loc_concat = to_ColVecs([input_loc_concat])
#
# sorting_perm = sortperm(time_loc_concat)
# input_loc_star = input_loc_concat[sorting_perm]
# # Code that generates the fₓ sample; i.e. the GP acting on prev outputs.
# Cfu_star = pairwise(Matern52(), input_loc_star, pseudo_input_locations)  # Cf*u

# TODO: figure out how multi-dim inputs work
y3_out = get_gpar_scaled_predictions(
    [y1, y2],
    pseudo_y3,
    time_loc,
    y3,  # outputs
    # Inference information
    test_time_loc,  # Time locations at which we do inference
    [test_y1, y2_out]
    )  # Input locations from prev outputs

# Plotting
gr()

overall_plot = plot(layout = (3, 1), legend = false);
scatter!(overall_plot[2], time_loc, y2,color = :black)
plot!(overall_plot[2], test_time_loc, test_y2, color = :orange)
plot!(overall_plot[2], test_time_loc, y2_out, color= :blue)

scatter!(overall_plot[3], time_loc, y3, color = :black)
plot!(overall_plot[3], test_time_loc, test_y3, color = :orange)
plot!(overall_plot[3], test_time_loc, y3_out, color= :blue)

display(overall_plot)

# # Get the dimensions
# N = length(y1)
# M = length(pseudo_y1)
# # Generate GP priors
# kern = Matern52()
# gp_prior = GP(kern, GPC())
# # Compute Cfu by using finiteGP. This was to test the pairwise function.
# OBSERVATION_NOISE_SIGMA = 0.05
#
# f = gp_prior(y1, OBSERVATION_NOISE_SIGMA^2)
# u = gp_prior(pseudo_y1, OBSERVATION_NOISE_SIGMA^2)
#
# # Compute DTC using our function
# time_kernel = Matern52()
# temporal_noise_sigma = 0.04
#
# # Create time LGSSM
# time_prior = GP(time_kernel, GPC())
# time_finite_gp = time_prior(time_loc, temporal_noise_sigma^2)
# noise_matrix = cov(time_finite_gp)
#
# time_sde = to_sde(time_prior, SArrayStorage(Float64))
# time_lgssm = time_sde(time_loc, temporal_noise_sigma^2)
#
# # APPROX posterior
# Cff = cov(f)  # N x N
# Cfu = cov(f, u)  # N x M array
# Cuf = Cfu'  # M x N
# Cuu = cov(u)  # M x M
# # Upper and lower cholesky factorizations of Cuu
# U_u = cholesky(Symmetric(Cuu)).U  # M x M array
# L_u = U_u'
#
# # Compute B_ef. First compute beta_interm = U_y' \ Cfu
# beta_interm = zeros((N, M)) # zero initialization
# # Iterate over the columns of Cfu and call decorrelate to construct beta
# for col_index = 1:M
#     col = Cfu[:, col_index]
#     # Call decorrelate to compute that column in beta
#     _, col_beta = decorrelate(time_lgssm, col)
#     beta_interm[:, col_index] = col_beta
# end
#
# B_ef = L_u \ beta_interm'  # M x N array
#
# # Compute b_y by calling decorrelate
# _, b_y = decorrelate(time_lgssm, outputs - mean(f))
#
# # Continue as in the usual DTC case.
# # This is because we no longer have N x N arrays.
# D = B_ef * B_ef' + I  # M x M array. It is Λ_ϵ
# chol_D = cholesky(Symmetric(D))  # low cost decomp to compute Lambda_e
# m_e = chol_D \ (B_ef * b_y)  # M array. It is m_ϵ
#
# # Compare against reference
#
# D_ref, m_ε_ref, = approx_posterior_ref(f, outputs, u, noise_matrix)
#
# D_diff = sum(D - D_ref)
# m_e_diff = sum(m_e - m_ε_ref)  # Stays
# println("D diff $(D_diff)")
# println("m_e diff $(m_e_diff)")
#
# # Now use these to compute the mean and var of q(f)
# # q_e = MvNormal(m_e, Symmetric(inv(D)))
# q_e = compute_q_u(
#     input_locations,
#     pseudo_input_locations, # pseudo-input locations in the samee domain as v
#     time_loc, # the time locations correspondiing to v
#     outputs;  # the ouput training data coresp to v
#     out_kernel = kern,  # kernel of the GP used on input points
#     time_kernel = time_kernel,  # kernel of the GP used on temporal locations
#     temporal_noise_sigma = temporal_noise_sigma  # TODO: This will need to be optimized
#     )
#
# # Get the testing data
# test_time_loc = x_true
# test_y1 = y_true[1]
# true_test_outputs = y_true[2]
#
# # Concatenate data locations with inference locations to get f*
# time_loc_concat = vcat(time_loc, test_time_loc)
# y1_concat = vcat(y1, test_y1)
# # Outputs, which a sentinel value in the places where we make inference.
# outputs_concat = vcat(outputs, repeat([0], length(test_time_loc)))
# # Get the permutation for sorting the locations
# sorting_perm = sortperm(time_loc_concat)
# reverse_perm = sortperm(sorting_perm)
# # Sort the inputs so that time is ascendent (for LGSSM)
# time_loc_star = time_loc_concat[sorting_perm]
# y1_star = y1_concat[sorting_perm]
# outputs_star = outputs_concat[sorting_perm]
#
# Cfu_star = pairwise(kern, y1_star, pseudo_y1)  # Cf*u
#
# function generate_fx_sample()
#     # Generate an fx sample by sampling q_e and plugging it in qf_mean.
#     # This is possible because Cff_hat is low rank (many 0 val eigenvals)
#     m_e = rand(q_e)
#     # Also add the prior mean here if nonzero
#     fx_sample = Cfu_star * (U_u \ (m_e))
# end
#
# # Create the LGSSM And get ft by doing y - fx and using pseudo_rand
# noise_vector_concat = vcat(
#     repeat([temporal_noise_sigma^2], length(time_loc)),
#     repeat([1e10], length(test_time_loc)))  # infinite variance in inference loc
# noise_vector_star = noise_vector_concat[sorting_perm]
#
# time_lgssm_star = time_sde(time_loc_star, noise_vector_star)
#
# function monte_carlo_f_star(iterations)
#     acc = zeros(length(time_loc_star))
#     for _ in 1:iterations
#         fx = generate_fx_sample()
#         y_star = outputs_star - fx
#
#         _, y_smooth, _ = smooth(time_lgssm_star, y_star)
#         y_smooth = [f.m[1] for f in y_smooth]
#         f_star = fx + y_smooth
#         acc = acc + f_star
#     end
#     return acc / iterations
# end
#
# inferred_f = monte_carlo_f_star(100)
# inf_test_out = inferred_f[reverse_perm][length(time_loc) + 1:length(inferred_f)]


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
