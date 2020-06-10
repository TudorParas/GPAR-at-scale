using GPARatScale
using Stheno
using LinearAlgebra

function test_dtc()
    # Example function, used for comparisson purposes
    function _compute_intermediates(f, y, u, noise_matrix)
        chol_Σy = cholesky(noise_matrix) # passed in the noise matrix
        # Lower triangular chol decomp of Cuu backsolved with
        # (Cfu¹ * L)ᵀ^-1 * chol(c)
        A = cholesky(Symmetric(cov(u))).U' \ (chol_Σy.U' \ cov(f, u))'
        Λ_ε = cholesky(Symmetric(A * A' + I))
        δ = chol_Σy.U' \ (y - mean(f))

        tmp =
            logdet(chol_Σy) + logdet(Λ_ε) + sum(abs2, δ) -
            sum(abs2, Λ_ε.U' \ (A * δ))
        _dtc = -(length(y) * typeof(tmp)(log(2π)) + tmp) / 2
        return _dtc, chol_Σy, A
    end

    # Example testing the two methods
    x, y_obs, _, _ = generate_small_dataset()
    # Model X=R²; (x, y1) -> y2; i.e  v=(x, y1), y = y2, and z=(pseudo_x, pseudo_y1)
    time_loc = x
    y1 = y_obs[1]
    outputs = y_obs[2]
    # Generate u, the pseudo-points, by taking every third element
    every_third = range(1; stop = length(x), step = 3)
    pseudo_y1 = y1[every_third]
    # Get the dimensions
    N = length(y1)
    M = length(pseudo_y1)
    # Generate GP priors
    kern = Matern52()
    gp_prior = GP(kern, GPC())
    # Compute Cfu by using finiteGP. This was to test the pairwise function.
    OBSERVATION_NOISE_SIGMA = 0.05

    f = gp_prior(y1, OBSERVATION_NOISE_SIGMA^2)
    u = gp_prior(pseudo_y1, OBSERVATION_NOISE_SIGMA^2)

    # Compute DTC using our function
    time_kernel = Matern52()
    temporal_noise_sigma = 0.04
    gpar_dtc, gpar_A = compute_gpar_dtc_objective(f, u, time_loc, outputs;
                            time_kernel=time_kernel, temporal_noise_sigma=temporal_noise_sigma)

    # Compute DTC using known to work function. Compute noise matrix first
    time_prior = GP(time_kernel, GPC())
    time_finite_gp = time_prior(time_loc, temporal_noise_sigma^2)
    noise_matrix = cov(time_finite_gp)

    func_dtc, _, func_A =  _compute_intermediates(f, outputs, u, noise_matrix)

    dtc_diff = (gpar_dtc - func_dtc)
    A_diff = sum(gpar_A - func_A)

    println("DTC difference $(dtc_diff)")
    println("A difference $(A_diff)")
end

test_dtc()
