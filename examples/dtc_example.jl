using GPARatScale
using Stheno
using LinearAlgebra

"""
Compare the dtc computed by our function against the Stheno DTC
"""
function compare_dtc_with_Stheno_dtc()
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


function compare_optimum_params(;nr_pseudo_points=400, use_same_points=false)

    # Get dataset
    x, y_obs, x_true, y_true = generate_small_dataset()
    # Expand y_obs and generate the pseudo points
    y1 = y_obs[1]
    y2 = y_obs[2]
    y3 = y_obs[3]

    if use_same_points
        println("Using pseudopoints same as input points")
        pseudo_f2 = [y1]
        pseudo_f3 = [y1, y2]
    else
        println("Generating $(nr_pseudo_points) pseudo-points on grid.")
        pseudo_f2 = [range(minimum(y1), stop=maximum(y1), length=nr_pseudo_points)]
        # Creating pseudo inputs in this case is harder since we need them on a grid
        nr_pseudo_points_pd = convert(Integer, ceil(sqrt(nr_pseudo_points)))  # pseudo points per dimension

        dim1 = range(minimum(y1), stop=maximum(y1), length=nr_pseudo_points_pd)
        dim2 = range(minimum(y2), stop=maximum(y2), length=nr_pseudo_points_pd)
        # Collect them in a grid and transform into ColVecs
        pseudo_f3 = vec([collect(i) for i in Iterators.product(dim1, dim2)])
        pseudo_f3 = ColVecs(hcat(pseudo_f3...))
    end
    # Create initial parameters
    out_kernel = Matern52()
    time_kernel = Matern52()
    i_log_time_l = 1.0
    i_log_time_var = 1.5
    i_log_out_l = 1.0
    i_log_out_var = 1.0
    i_log_noise_sigma=-3.0

    println("Optimizing GPAR parameters for f2")
    f2_gpar, opt_params_f2 = create_optim_gpar(
        [x, y1],
        y2;
        time_kernel = time_kernel,
        out_kernel = out_kernel,
        multi_input = true,
        debug = true,
        i_log_time_l=i_log_time_l, i_log_time_var=i_log_time_var, i_log_out_l=i_log_out_l,
        i_log_out_var=i_log_out_var, i_log_noise_sigma=i_log_noise_sigma
    )

    pseudo_y1 = range(minimum(y1), stop=maximum(y1), length=nr_pseudo_points)
    println("Optimizing Scaled GPAR for f2")
    scaled_opt_params_f2 = get_optim_scaled_gpar_params(
        [y1],
        pseudo_f2,
        x,
        y2;
        out_kernel = out_kernel,
        time_kernel = time_kernel,
        debug = true,
        i_log_time_l=i_log_time_l, i_log_time_var=i_log_time_var, i_log_out_l=i_log_out_l,
        i_log_out_var=i_log_out_var, i_log_noise_sigma=i_log_noise_sigma
        )

    println("Optimizing GPAR parameters for f3")
    f3_gpar, opt_params_f3 = create_optim_gpar(
        [x, y1, y2],
        y3;
        time_kernel = time_kernel,
        out_kernel = out_kernel,
        multi_input = true,
        debug = true,
        i_log_time_l=i_log_time_l, i_log_time_var=i_log_time_var, i_log_out_l=i_log_out_l,
        i_log_out_var=i_log_out_var, i_log_noise_sigma=i_log_noise_sigma,
    )

    # println("Using the pseudo-points:")
    # println(pseudo_loc)
    println("Optimizing Scaled GPAR for f3")
    scaled_opt_params_f3 = get_optim_scaled_gpar_params(
        [y1, y2],
        pseudo_f3,
        x,
        y3;
        out_kernel = out_kernel,
        time_kernel = time_kernel,
        debug = true,
        i_log_time_l=i_log_time_l, i_log_time_var=i_log_time_var, i_log_out_l=i_log_out_l,
        i_log_out_var=i_log_out_var, i_log_noise_sigma=i_log_noise_sigma
        )

    # Round everthing to 3 digits
    opt_params_f2 = [round(i, digits=3) for i in opt_params_f2]
    scaled_opt_params_f2 = [round(i, digits=3) for i in scaled_opt_params_f2]
    opt_params_f3 = [round(i, digits=3) for i in opt_params_f3]
    scaled_opt_params_f3 = [round(i, digits=3) for i in scaled_opt_params_f3]

    println("For f2 got \n\tOpt params: $(opt_params_f2)\n Scaled opt params: $(scaled_opt_params_f2)")
    println("For f3 got \n\tOpt params: $(opt_params_f3)\n Scaled opt params: $(scaled_opt_params_f3)")

end
# compare_dtc_with_Stheno_dtc()
compare_optimum_params(nr_pseudo_points=400, use_same_points=false)
