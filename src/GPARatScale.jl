module GPARatScale
    using Stheno, TemporalGPs, Distributions, Optim, Zygote, Random
    # Data generation
    include(joinpath("data", "toy_data.jl"))

    # GPAR
    include(joinpath("gp", "optimized.jl"))
    # Temporal GP
    include(joinpath("gp", "temporal_gp_inference.jl"))
    # Util
    include("util.jl")
end
