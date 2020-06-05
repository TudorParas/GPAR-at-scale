using Stheno: cholesky
using TemporalGPs: to_sde, smooth, SArrayStorage, decorrelate
using Plots
using Optim
using Zygote: gradient
using Random

include("data\\toy_data.jl")
include("util.jl")

# Create function that computes dtc objective that takes in a Stheno GP, and the
# input locations, and y1 locations

# This should use TemporalGPs inside. Replace obs noice cov matrix with the
# decorrelate op in TemporalGPs
# Line 263 on compute_invariants
# Instead of chol_Σy.U' \ (y - mean(f)) do decorrelate(LGSSM, y - mean(f)) - > exactly the same as what woul've happened if computed the covar matrix and then backsolve against y - mean(f)
function get_dtc_objective(f, u, time, y1, pseudo_time, pseudo_y1)
    # Turn the GPs into an LTISDE
    sde = to_sde(gp, SArrayStorage(Float64))
    # Turn the LTISDEs into LGSSMs by indexing
    chol_Σy = cholesky(f.Σy)
    # Instead of chol_Σy.U' \ (y - mean(f)) do decorrelate(LGSSM, y - mean(f))
    # Exactly the same as what woul've happened if computed the covar matrix and
    # then backsolve against y - mean(f)
    A = cholesky(Symmetric(cov(u))).U' \ (chol_Σy.U' \ cov(f, u))'
    Λ_ε = cholesky(Symmetric(A * A' + I))
    δ = chol_Σy.U' \ (y - mean(f))

    tmp = logdet(chol_Σy) + logdet(Λ_ε) + sum(abs2, δ) - sum(abs2, Λ_ε.U' \ (A * δ))
    _dtc = -(length(y) * typeof(tmp)(log(2π)) + tmp) / 2
    return _dtc, chol_Σy, A
end

# # Example function
# function _compute_intermediates(f::FiniteGP, y::AV{<:Real}, u::FiniteGP)
#     consistency_check(f, y, u)
#     chol_Σy = cholesky(f.Σy)
#     # Lower triangular chol decomp of Cuu backsolved with
#     # (Cfu¹ * L)ᵀ^-1 * chol(c)
#     A = cholesky(Symmetric(cov(u))).U' \ (chol_Σy.U' \ cov(f, u))'
#     Λ_ε = cholesky(Symmetric(A * A' + I))
#     δ = chol_Σy.U' \ (y - mean(f))
#
#     tmp = logdet(chol_Σy) + logdet(Λ_ε) + sum(abs2, δ) - sum(abs2, Λ_ε.U' \ (A * δ))
#     _dtc = -(length(y) * typeof(tmp)(log(2π)) + tmp) / 2
#     return _dtc, chol_Σy, A
# end

x, y_obs, _, _ = generate_small_dataset()
# Model X=R²; (x, y1) -> y2; i.e  v=(x, y1), y = y2, and z=(pseudo_x, pseudo_y1)
time_loc = x
outputs = y_obs[1]
# Generate u, the pseudo-points, by taking every third element
every_third = range(1; stop=length(x), step=3)

pseudo_time_loc = time_loc[every_third]
pseudo_outputs = outputs[every_third]
# Get the dimensions
N = length(time_loc)
M = length(pseudo_time_loc)
# TODO: Perfor  m this concatination when extending to D > 1
# Concatenate inputs so we treat them together
# inp = to_ColVecs([time_loc, y1])
# pseudo_inp = to_ColVecs([pseudo_time_loc, pseudo_y1])

# Generate GP priors
kern = Matern52()
f_prior = GP(kern, GPC())
u_prior = GP(kern, GPC())

# Compute Cfu by using finiteGP. This was to test the pairwise function.
f = f_prior(time_loc, 0.1)
u = f_prior(pseudo_time_loc, 0.1)
Cfu_naive = cov(f, u)


# Compute Cfu by using pairwise
Cfu = pairwise(kern, time_loc, pseudo_time_loc)

# TODO: Use this column by column computation when computing B
Cfu_smart = zeros(N, M)
# Compute column by column
for col_index in 1:M
    Cfu_smart[:, col_index] = pairwise(kern, time_loc, [pseudo_time_loc[col_index]])
end

# Create LTISDE from gps
storage = SArrayStorage(Float64)

f_sde = to_sde(f_prior, storage)
u_sde = to_sde(u_prior, storage)
# Turn the LTISDE into LGSSM by indexing
NOISE_SIGMA = 0.05
# TODO: This is where extending D > 1 failed. SDE doesn't support multi-dim inp
f_lgssm = f_sde(time_loc, NOISE_SIGMA)
u_lgssm = u_sde(pseudo_time_loc, NOISE_SIGMA)

# Here we'll naively compute alpha and Beta
L_naive = cholesky(f.Σy).U'
alpha_naive = L_naive \ outputs
beta_naive = L_naive \ Cfu_naive

# Compute alpha
_, alpha = decorrelate(f_lgssm, outputs)

# Compute beta
beta = zeros((N, M)) # zero initialization
# Iterate over the columns of Cfu and call decorrelate to construct beta
for col_index in 1:M
    col = Cfu[:, col_index]
    # Call decorrelate to compute that column in beta
    _, col_beta = decorrelate(f_lgssm, col)
    beta[:, col_index] = col_beta
end

# Compute A from A = cholesky(Symmetric(cov(u))).U' \ (chol_Σy.U' \ cov(f, u))'
