using Pkg

Pkg.activate(joinpath(@__DIR__, ".."))
Pkg.instantiate()

using Random
using Statistics
using SOMSOS

rng = MersenneTwister(20260406)

sim = simulate_glam_style_regression_data(
    rng,
    SimulationConfig(
        n_subjects = 48,
        n_features = 8,
        min_repeats = 12,
        max_repeats = 18,
    );
    active_indices = [1, 3, 5],
    active_values = [2.6, -2.2, 1.8],
    true_t = 0.85,
    true_intercept = 0.4,
)

result = sample_regression(rng, sim.data, RegressionConfig(), 200; burnin = 100, thin = 20)

println("saved draws: ", length(result.samples))
println("posterior mean t: ", round(result.summary.mean_t; digits = 3))
println("posterior inclusion probabilities: ", round.(result.summary.pip; digits = 3))
println("last log posterior: ", round(result.final_sample.logposterior; digits = 2))
println("true active set: ", findall(==(1), sim.true_gamma))
