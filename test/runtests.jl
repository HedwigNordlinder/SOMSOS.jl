using Test
using Random
using Statistics

using SOMSOS

@testset "SOMSOS" begin
    patient = LabeledPatientMatrix([1.0 2.0; 3.0 4.0; 5.0 6.0], [0, 1, 1])
    data = ScalarOnMatrixData([patient], [1])
    design = build_component_design(data)

    @test patient.z == [1, 2, 2]
    @test design.component1[1, :] ≈ [1.0, 2.0] ./ 3
    @test design.component2[1, :] ≈ [8.0, 10.0] ./ 3
    @test working_design(design, 0.25)[1, :] ≈ 0.25 .* design.component1[1, :] .+ 0.75 .* design.component2[1, :]

    rng = MersenneTwister(20260406)
    sim = simulate_glam_style_regression_data(
        rng,
        SimulationConfig(
            n_subjects = 24,
            n_features = 6,
            min_repeats = 10,
            max_repeats = 14,
        );
        active_indices = [1, 3],
        active_values = [2.5, -2.0],
        true_t = 0.82,
        true_intercept = 0.3,
    )

    @test n_subjects(sim.data) == 24
    @test n_features(sim.data) == 6
    @test all(in((0, 1)), sim.data.y)

    result = sample_regression(rng, sim.data, RegressionConfig(), 80; burnin = 20, thin = 15)

    @test length(result.samples) == 4
    @test length(result.logposterior_trace) == 80
    @test length(result.t_trace) == 80
    @test length(result.active_trace) == 80
    @test all(isfinite, result.logposterior_trace)
    @test 0.0 <= result.summary.mean_t <= 1.0
    @test result.final_sample.logposterior ≈ logposterior(sim.data, result.final_sample, RegressionConfig())
    @test all(in((0, 1)), result.final_sample.gamma)
    @test length(result.summary.pip) == 6
    @test all(0.0 .<= result.summary.pip .<= 1.0)
    @test length(result.summary.mean_beta) == 6
    @test result.summary.beta_acceptance >= 0.0
    @test result.summary.intercept_acceptance >= 0.0
    @test result.summary.t_acceptance >= 0.0
end
