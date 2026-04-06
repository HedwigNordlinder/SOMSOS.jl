using Test
using Random
using Statistics

using SOMSOS

function make_test_data(rng; n_patients = 24)
    patients = LabeledPatientMatrix[]
    y = Int[]
    true_beta = [2.2, 0.0, -1.7, 0.0]
    true_t = 0.78
    true_intercept = 0.25

    for _ in 1:n_patients
        component1 = hcat(
            -1.2 .+ 0.25 .* randn(rng, 5),
            0.10 .* randn(rng, 5),
            -1.1 .+ 0.25 .* randn(rng, 5),
            0.10 .* randn(rng, 5),
        )
        component2 = hcat(
            1.2 .+ 0.25 .* randn(rng, 6),
            0.10 .* randn(rng, 6),
            1.1 .+ 0.25 .* randn(rng, 6),
            0.10 .* randn(rng, 6),
        )
        patient = LabeledPatientMatrix(vcat(component1, component2), vcat(fill(1, 5), fill(2, 6)))
        signal = working_design([patient], true_t)[1, :]
        eta = true_intercept + sum(signal .* true_beta)
        push!(patients, patient)
        push!(y, rand(rng) < 1 / (1 + exp(-eta)) ? 1 : 0)
    end

    return ScalarOnMatrixData(patients, y)
end

@testset "SOMSOS" begin
    patient = LabeledPatientMatrix([1.0 2.0; 3.0 4.0; 5.0 6.0], [0, 1, 1])
    data = ScalarOnMatrixData([patient], [1])
    design = build_component_design(data)

    @test patient.z == [1, 2, 2]
    @test design.component1[1, :] ≈ [1.0, 2.0] ./ 3
    @test design.component2[1, :] ≈ [8.0, 10.0] ./ 3
    @test working_design(design, 0.25)[1, :] ≈ 0.25 .* design.component1[1, :] .+ 0.75 .* design.component2[1, :]
    @test naive_average_matrix(data)[1, :] ≈ [3.0, 4.0]

    rng = MersenneTwister(20260406)
    sim = make_test_data(rng)

    @test n_subjects(sim) == 24
    @test n_features(sim) == 4
    @test all(in((0, 1)), sim.y)

    result = sample_regression(rng, sim, RegressionConfig(), 80; burnin = 20, thin = 15)
    naive = sample_naive_regression(rng, sim, RegressionConfig(), 80; burnin = 20, thin = 15)

    @test length(result.samples) == 4
    @test length(result.logposterior_trace) == 80
    @test length(result.t_trace) == 80
    @test length(result.active_trace) == 80
    @test all(isfinite, result.logposterior_trace)
    @test 0.0 <= result.summary.mean_t <= 1.0
    @test result.final_sample.logposterior ≈ logposterior(sim, result.final_sample, RegressionConfig())
    @test all(in((0, 1)), result.final_sample.gamma)
    @test length(result.summary.pip) == 4
    @test all(0.0 .<= result.summary.pip .<= 1.0)
    @test length(result.summary.mean_beta) == 4
    @test length(result.summary.mean_active_beta) == 4
    @test result.summary.beta_acceptance >= 0.0
    @test result.summary.intercept_acceptance >= 0.0
    @test result.summary.t_acceptance >= 0.0
    @test length(predict_probabilities(result, sim)) == n_subjects(sim)

    @test length(naive.samples) == 4
    @test length(naive.logposterior_trace) == 80
    @test length(naive.active_trace) == 80
    @test all(isfinite, naive.logposterior_trace)
    @test naive.final_sample.logposterior ≈ logposterior(sim, naive.final_sample, RegressionConfig())
    @test length(naive.summary.pip) == 4
    @test all(0.0 .<= naive.summary.pip .<= 1.0)
    @test length(naive.summary.mean_active_beta) == 4
    @test length(predict_probabilities(naive, sim)) == n_subjects(sim)
end
