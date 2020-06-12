using Stheno: cholesky
using TemporalGPs: to_sde, smooth, SArrayStorage, decorrelate
using Stheno
using LinearAlgebra
using Optim

using GPARatScale

export get_optim_scaled_gpar_params, compute_gpar_dtc_objective

function get_optim_scaled_gpar_params(
    input_locations, # input locations from previous outputs. Should be array of arrays
    pseudo_input_locations, # pseudo-input locations in the samee domain as v
    time_loc, # the time locations correspondiing to v
    outputs;  # the ouput training data coresp to v
    out_kernel = Matern52(),  # kernel of the GP used on input points
    time_kernel = Matern52(),  # kernel of the GP used on temporal locations
    # Initial log param values. Temporal noise and output noise are the same.
    i_log_time_l=nothing, i_log_time_var=nothing, i_log_out_l=nothing,
    i_log_out_var=nothing, i_log_noise_sigma=nothing,
    debug::Bool=false,
    storage = SArrayStorage(Float64), # storage used for TemporalGPs
    )
    input_locations = to_ColVecs(input_locations)
    pseudo_input_locations = to_ColVecs(pseudo_input_locations)

    # counter = 0 # counter used for debugging
    function nlml(params)
        time_l, time_var, out_l, out_var, noise_sigma = unpack_gpar(params)
        gpar_kernel = kernel(out_kernel, l=out_l, s=out_var^2)
        gp_prior = GP(gpar_kernel, GPC())
        # Create the FiniteGPs
        f = gp_prior(input_locations, noise_sigma^2)
        u = gp_prior(pseudo_input_locations, noise_sigma^2)
        # Create kernel for temporal GP
        temporal_kernel = kernel(time_kernel, l=time_l, s=time_var^2)
        dtc, _ = compute_gpar_dtc_objective(
            f,
            u,
            time_loc, # the time locations correspondiing to v
            outputs;  # the ouput training data coresp to v
            time_kernel=temporal_kernel,  # kernel of the GP used on temporal locations
            temporal_noise_sigma=noise_sigma,
            storage=storage,
        )
        return -dtc
    end
    params = parse_initial_gpar_params(
        i_log_time_l, i_log_time_var, i_log_out_l,
        i_log_out_var, i_log_noise_sigma)
    if debug
        i_time_l, i_time_var, i_out_l, i_out_var, i_noise_sigma = unpack_gpar(params)
        println("Generating scaled GPAR with initial parameters:")
        println("\ti_time_l=$(i_time_l); i_time_var=$(i_time_var); i_out_l=$(i_out_l); i_out_var=$(i_out_var); i_noise_sigma=$(i_noise_sigma)")
    end

    results = Optim.optimize(nlml, params, NelderMead())
    opt_params = unpack_gpar(results.minimizer)

    if debug
        opt_time_l, opt_time_var, opt_out_l, opt_out_var, opt_noise_sigma =
            opt_params
        println("Finished optimizing parameters:")
        println("\tOptimum time L: $(opt_time_l) ")
        println("\tOptimum time var: $(opt_time_var)")
        println("\tOptimum outputs l: $(opt_out_l)")
        println("\tOptimum outputs var: $(opt_out_var)")
        println("\tOptimum Noise std: $(opt_noise_sigma)")
        println()
    end

    return opt_params
end

"""
Given the FiniteGPs at the input and pseudo-input locations, compute the DTC
objective using LGSSM for acceleration.
"""
function compute_gpar_dtc_objective(
    f,
    u,  # Finite GP on input and pseudo-input locations
    time_loc, # the time locations correspondiing to v
    outputs;  # the ouput training data coresp to v
    time_kernel = Matern52(),  # kernel of the GP used on temporal locations
    temporal_noise_sigma = 0.04,
    storage = SArrayStorage(Float64),
)
    # Compute the nr of inputs and pseudo-inputs
    N = length(f)
    M = length(u)
    # Compute the noise matrix and the temporal sde
    time_prior = GP(time_kernel, GPC())
    # TODO: Compute the matrix without actually generating a time GP
    time_finite_gp = time_prior(time_loc, temporal_noise_sigma^2)
    noise_matrix = cov(time_finite_gp)
    # Create the time LGSSM used for speeding up computation
    time_sde = to_sde(time_prior, storage)
    time_lgssm = time_sde(time_loc, temporal_noise_sigma^2)
    # Compute Cfu, the covariance matrix between the two FiniteGPs
    Cfu = cov(f, u)
    # Compute alpha
    _, alpha = decorrelate(time_lgssm, outputs - mean(f))
    # Compute beta
    beta = zeros((N, M)) # zero initialization
    # Iterate over the columns of Cfu and call decorrelate to construct beta
    for col_index = 1:M
        # TODO: Compute the coresponding column. This avoids storing the Cfu matrix
        # col = pairwise(out_kernel, input_locations, [pseudo_input_locations[col_index]])
        col = Cfu[:, col_index]
        # Call decorrelate to compute that column in beta
        _, col_beta = decorrelate(time_lgssm, col)
        beta[:, col_index] = col_beta
    end
    # Compute the DTC
    A = cholesky(Symmetric(cov(u))).U' \ (beta)'
    Λ_ε = cholesky(Symmetric(A * A' + I))

    tmp =
        logdet(noise_matrix) + logdet(Λ_ε) + sum(abs2, alpha) -
        sum(abs2, Λ_ε.U' \ (A * alpha))
    dtc = -(length(outputs) * typeof(tmp)(log(2π)) + tmp) / 2

    return dtc, A
end
