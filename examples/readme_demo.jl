using Pkg

Pkg.activate(joinpath(@__DIR__, ".."))
Pkg.instantiate()

using Random
using Statistics
using SOMSOS

rng = MersenneTwister(20260406)

function make_demo_data(rng; n_patients = 32)
    patients = LabeledPatientMatrix[]
    y = Int[]
    true_beta = [2.1, 0.0, -1.5]
    true_t = 0.8
    true_intercept = 0.2

    for _ in 1:n_patients
        component1 = hcat(
            -1.3 .+ 0.25 .* randn(rng, 5),
            0.15 .* randn(rng, 5),
            -1.0 .+ 0.25 .* randn(rng, 5),
        )
        component2 = hcat(
            1.3 .+ 0.25 .* randn(rng, 6),
            0.15 .* randn(rng, 6),
            1.0 .+ 0.25 .* randn(rng, 6),
        )
        patient = LabeledPatientMatrix(vcat(component1, component2), vcat(fill(1, 5), fill(2, 6)))
        signal = working_design([patient], true_t)[1, :]
        eta = true_intercept + sum(signal .* true_beta)
        push!(patients, patient)
        push!(y, rand(rng) < 1 / (1 + exp(-eta)) ? 1 : 0)
    end

    return ScalarOnMatrixData(patients, y)
end

data = make_demo_data(rng)
aware = sample_regression(rng, data, RegressionConfig(), 200; burnin = 100, thin = 20)
naive = sample_naive_regression(rng, data, RegressionConfig(), 200; burnin = 100, thin = 20)

println("cluster-aware mean t: ", round(aware.summary.mean_t; digits = 3))
println("cluster-aware PIPs: ", round.(aware.summary.pip; digits = 3))
println("naive PIPs: ", round.(naive.summary.pip; digits = 3))
println("first five aware probabilities: ", round.(predict_probabilities(aware, data)[1:5]; digits = 3))
