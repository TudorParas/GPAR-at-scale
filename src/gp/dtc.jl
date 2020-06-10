using Stheno: cholesky
using TemporalGPs: to_sde, smooth, SArrayStorage, decorrelate
using Stheno
using LinearAlgebra

using GPARatScale

function compute_gpar_dtc_objective(
    f,
    u,  # Finite GP on input and pseudo-input locations
    # input_locations, # input locations from previous outputs
    # pseudo_input_locations, # pseudo-input locations in the samee domain as v
    time_loc, # the time locations correspondiing to v
    outputs;  # the ouput training data coresp to v
    # out_kernel = Matern52(),  # kernel of the GP used on input points
    time_kernel = Matern52(),  # kernel of the GP used on temporal locations
    # out_noise_sigma = 0.05,
    temporal_noise_sigma = 0.04,
    storage = SArrayStorage(Float64),
)
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
