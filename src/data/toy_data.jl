using Distributions

export generate_small_dataset, generate_big_dataset

START = 0
STEP_SIZE = 0.03
NOISE_MU = 0  # mean of noise
NOISE_SIGMA = 0.05

f1 = (x -> -sin.(10 * pi .* (x .+ 1)) ./ (2 .* x .+ 1) - x .^ 4)
f2 = ((x, y1) -> cos.(y1) .^ 2 + sin.(3 .* x))
f3 = ((x, y1, y2) -> y2 .* (y1 .^ 2) + 3 * x)

function _generate_toy_data(data_samples, true_samples)
    # Generate True function
    stop = STEP_SIZE * data_samples
    x_true = range(START, stop, length=true_samples)
    y1_true = f1(x_true)
    y2_true = f2(x_true, y1_true)
    y3_true = f3(x_true, y1_true, y2_true)
    y_true = [y1_true, y2_true, y3_true]

    # Create the noisy normal distribution
    normal_noise = Normal(NOISE_MU, NOISE_SIGMA^2)
    # Add the noise to the observations
    x = range(START, stop, length = data_samples)
    y1 = f1(x) + rand(normal_noise, data_samples)
    y2 = f2(x, y1) + rand(normal_noise, data_samples)
    y3 = f3(x, y1, y2) + rand(normal_noise, data_samples)
    y_obs = [y1, y2, y3]

    return x, y_obs, x_true, y_true
end

SMALL_DATA_SAMPLES = 30
SMALL_TRUE_SAMPLES = 1000 # discretised points used to model true function

generate_small_dataset() = _generate_toy_data(SMALL_DATA_SAMPLES, SMALL_TRUE_SAMPLES)

BIG_DATA_SAMPLES = 10000
BIG_TRUE_SAMPLES = 1000000 # discretised points used to model true function

generate_big_dataset() = _generate_toy_data(BIG_DATA_SAMPLES, BIG_TRUE_SAMPLES)
