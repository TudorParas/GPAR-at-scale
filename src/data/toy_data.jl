using Distributions

export generate_small_dataset, generate_big_dataset

START = 0
STEP_SIZE = 1 / 30
NOISE_MU = 0  # mean of noise

function _generate_toy_data(
    data_samples,
    true_samples,
    f1,
    f2,
    f3;
    observation_noise = 0.05,
    extended_true_period = 0,
    nuke_interval=0  # Number of elements to remove from the middle of the data
)
    # Generate True function
    stop = STEP_SIZE * data_samples
    x_true = range(START, stop + extended_true_period, length = true_samples)
    y1_true = f1(x_true)
    y2_true = f2(x_true, y1_true)
    y3_true = f3(x_true, y1_true, y2_true)
    y_true = [y1_true, y2_true, y3_true]

    # Create the noisy normal distribution
    normal_noise = Normal(NOISE_MU, observation_noise^2)
    # Add the noise to the observations
    # TODO: make noise scale with y
    x = range(START, stop, length = data_samples)
    x = nuke(x, nuke_interval)
    y1 = f1(x) + rand(normal_noise, data_samples - nuke_interval)
    y2 = f2(x, y1) + rand(normal_noise, data_samples - nuke_interval)
    y3 = f3(x, y1, y2) + rand(normal_noise, data_samples - nuke_interval)
    y_obs = [y1, y2, y3]

    return x, y_obs, x_true, y_true
end

function nuke(x, nuke_interval)
    # Remove two intervals of data
    one_third = length(x) ÷ 3
    two_thirds = 2 * one_third
    removed_elms = nuke_interval ÷ 2
    # Remove nuke_interval elms from the middle
    x = vcat(
        x[1:one_third],
        x[one_third + 1 + removed_elms: two_thirds],
        x[two_thirds + 1 + removed_elms: length(x)]
        )
    return x
end

SMALL_DATA_SAMPLES = 30
SMALL_TRUE_SAMPLES = 1000 # discretised points used to model true function
# Functions used to generate data
f1_small = (x -> -sin.(10 * pi .* (x .+ 1)) ./ (2 .* x .+ 1) - x .^ 4)
f2_small = ((x, y1) -> cos.(y1) .^ 2 + sin.(3 .* x))
f3_small = ((x, y1, y2) -> y2 .* (y1 .^ 2) + 3 * x)

observation_noise_small = 0.05
generate_small_dataset() = _generate_toy_data(
    SMALL_DATA_SAMPLES,
    SMALL_TRUE_SAMPLES,
    f1_small,
    f2_small,
    f3_small;
    observation_noise=observation_noise_small,
)

BIG_DATA_SAMPLES = 10000
BIG_TRUE_SAMPLES = 1000000 # discretised points used to model true function

f1_big = (x -> 3 .+ -sin.(1 / 10 * pi .* (x .+ 1)) - x .^ 0.3)
f2_big = ((x, y1) -> cos.(y1) .^ 2 + sin.(pi / 20 .* x))
f3_big = ((x, y1, y2) -> y2 .* (y1 .^ 2) + 0.1 .* x)

observation_noise_big = 0.8
extended_true_period_big = 50  # plot the true data for up to 50 datapoints after
nuke_interval_big = 600 # remove 600 data points. spanning a rangee of 20
generate_big_dataset() = _generate_toy_data(
    BIG_DATA_SAMPLES,
    BIG_TRUE_SAMPLES,
    f1_big,
    f2_big,
    f3_big,
    observation_noise=observation_noise_big,
    extended_true_period=extended_true_period_big,
    nuke_interval=nuke_interval_big
)
