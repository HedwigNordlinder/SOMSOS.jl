module SOMSOS

using Random
using LinearAlgebra
using Statistics
using Distributions

export LabeledPatientMatrix,
       ScalarOnMatrixData,
       ComponentDesign,
       RegressionConfig,
       RegressionSample,
       RegressionSummary,
       RegressionResult,
       SimulationConfig,
       SimulationResult,
       labeled_patients,
       n_subjects,
       n_features,
       build_component_design,
       working_design,
       sample_regression,
       logposterior,
       simulate_glam_style_regression_data

"""
    LabeledPatientMatrix(x, z)

One patient's repeated measurements together with a fixed two-component allocation for
each row.

`x` must be an `n_i x p` matrix and `z` must have length `n_i`. Labels may be supplied
either as `0/1` or `1/2`; internally they are stored as `1/2`.
"""
struct LabeledPatientMatrix
    x::Matrix{Float64}
    z::Vector{Int}
    function LabeledPatientMatrix(x::AbstractMatrix{<:Real}, z::AbstractVector{<:Integer})
        size(x, 1) > 0 || throw(ArgumentError("Each patient matrix must have at least one row."))
        size(x, 2) > 0 || throw(ArgumentError("Each patient matrix must have at least one feature."))
        size(x, 1) == length(z) || throw(ArgumentError("The allocation vector must match the number of matrix rows."))
        return new(Matrix{Float64}(x), _normalize_labels(z))
    end
end

function _normalize_labels(z::AbstractVector{<:Integer})
    labels = Int.(collect(z))
    isempty(labels) && throw(ArgumentError("Allocation vectors must contain at least one label."))
    if all(in((1, 2)), labels)
        return labels
    elseif all(in((0, 1)), labels)
        return labels .+ 1
    end
    throw(ArgumentError("Allocations must use labels {0,1} or {1,2}."))
end

"""
    labeled_patients(x, z)

Construct a vector of [`LabeledPatientMatrix`](@ref) objects from a vector of matrices and
a matching vector of allocation vectors.

This is the main bridge point from `HierarchicalMogSampler.jl`: if `x` is the observed
matrix list and `z` is `sample.z`, then `labeled_patients(x, z)` is directly usable here.
"""
function labeled_patients(
    x::Vector{<:AbstractMatrix{<:Real}},
    z::Vector{<:AbstractVector{<:Integer}},
)
    length(x) == length(z) || throw(ArgumentError("The matrix list and allocation list must have the same length."))
    return [LabeledPatientMatrix(Xi, zi) for (Xi, zi) in zip(x, z)]
end

"""
    ScalarOnMatrixData(patients, y)
    ScalarOnMatrixData(x, z, y)

Binary scalar responses together with patient-level labeled matrices for the SOMSOS model.

All patients must have the same number of features. `y` is validated to contain only `0`
and `1`.
"""
struct ScalarOnMatrixData
    patients::Vector{LabeledPatientMatrix}
    y::Vector{Int}
    function ScalarOnMatrixData(
        patients::Vector{LabeledPatientMatrix},
        y::AbstractVector{<:Integer},
    )
        isempty(patients) && throw(ArgumentError("ScalarOnMatrixData requires at least one patient."))
        labels = Int.(collect(y))
        length(patients) == length(labels) || throw(ArgumentError("The response vector must match the number of patients."))
        all(in((0, 1)), labels) || throw(ArgumentError("Responses must be binary values in {0,1}."))
        p = size(patients[1].x, 2)
        for (i, patient) in pairs(patients)
            size(patient.x, 2) == p || throw(ArgumentError("Patient $i has a different feature dimension."))
        end
        return new(copy(patients), labels)
    end
end

ScalarOnMatrixData(
    x::Vector{<:AbstractMatrix{<:Real}},
    z::Vector{<:AbstractVector{<:Integer}},
    y::AbstractVector{<:Integer},
) = ScalarOnMatrixData(labeled_patients(x, z), y)

"""
    n_subjects(data)

Return the number of patients in `data`.
"""
n_subjects(data::ScalarOnMatrixData) = length(data.patients)

"""
    n_features(data)

Return the feature dimension of each patient matrix in `data`.
"""
n_features(data::ScalarOnMatrixData) = size(data.patients[1].x, 2)

"""
    ComponentDesign

Deterministic design matrices derived from the labeled patient matrices.

For patient `i`, `component1[i, :]` and `component2[i, :]` are the sums of the rows
assigned to components 1 and 2 respectively, each divided by the patient's total number of
repeated measurements. This matches the original GLAM regression stage exactly: these are
not within-component means.
"""
struct ComponentDesign
    component1::Matrix{Float64}
    component2::Matrix{Float64}
end

"""
    build_component_design(data)

Convert labeled patient matrices into the component-specific patient summaries used by the
scalar-on-matrix regression stage.
"""
function build_component_design(data::ScalarOnMatrixData)
    n = n_subjects(data)
    p = n_features(data)
    component1 = zeros(n, p)
    component2 = zeros(n, p)

    for i in 1:n
        patient = data.patients[i]
        Xi = patient.x
        zi = patient.z
        n_i = size(Xi, 1)
        for j in 1:n_i
            if zi[j] == 1
                component1[i, :] .+= Xi[j, :]
            else
                component2[i, :] .+= Xi[j, :]
            end
        end
        component1[i, :] ./= n_i
        component2[i, :] ./= n_i
    end

    return ComponentDesign(component1, component2)
end

"""
    RegressionConfig(; kwargs...)

Hyperparameters and proposal scales for the scalar-on-matrix spike-and-slab logistic
regression model extracted from the original GLAM demo.

The model uses:
- spike-and-slab inclusion indicators `gamma`,
- Gaussian coefficients `beta`,
- a Gaussian prior on the intercept,
- a Beta prior on the mixing weight `t`,
- a Beta-Bernoulli prior on feature inclusion,
- an inverse-Gamma prior on the slab variance.
"""
Base.@kwdef struct RegressionConfig
    a_alpha::Float64 = 2.0
    b_alpha::Float64 = 2.0
    a_omega::Float64 = 1.0
    b_omega::Float64 = 6.0
    a_tau::Float64 = 2.0
    b_tau::Float64 = 2.0
    sigma0::Float64 = 3.0
    beta_step::Float64 = 0.30
    intercept_step::Float64 = 0.45
    u_step::Float64 = 0.28
end

mutable struct RegressionState
    beta::Vector{Float64}
    gamma::Vector{Int}
    intercept::Float64
    u::Float64
    omega::Float64
    tau2::Float64
end

"""
    RegressionSample

One saved posterior draw from the SOMSOS regression sampler.

Fields:
- `beta`: coefficient vector.
- `gamma`: spike-and-slab inclusion indicators.
- `intercept`: logistic intercept.
- `t`: scalar mixing weight in `[0,1]`.
- `omega`: marginal feature-inclusion probability.
- `tau2`: slab variance.
- `logposterior`: joint log posterior for this draw.
"""
struct RegressionSample
    beta::Vector{Float64}
    gamma::Vector{Int}
    intercept::Float64
    t::Float64
    omega::Float64
    tau2::Float64
    logposterior::Float64
end

"""
    RegressionSummary

Posterior summaries averaged across saved draws.
"""
struct RegressionSummary
    mean_beta::Vector{Float64}
    pip::Vector{Float64}
    mean_t::Float64
    mean_intercept::Float64
    beta_acceptance::Float64
    intercept_acceptance::Float64
    t_acceptance::Float64
end

"""
    RegressionResult

Output from [`sample_regression`](@ref).

Fields:
- `samples`: saved posterior draws after burn-in and thinning.
- `summary`: posterior means and inclusion probabilities.
- `logposterior_trace`: log posterior at every iteration.
- `t_trace`: sampled `t` trace at every iteration.
- `active_trace`: number of active features at every iteration.
- `final_sample`: final Gibbs/MALA state, whether or not it was saved.
"""
struct RegressionResult
    samples::Vector{RegressionSample}
    summary::RegressionSummary
    logposterior_trace::Vector{Float64}
    t_trace::Vector{Float64}
    active_trace::Vector{Int}
    final_sample::RegressionSample
end

"""
    SimulationConfig(; n_subjects=96, n_features=6, min_repeats=14, max_repeats=22)

Settings for the built-in simulator that reproduces the full GLAM-style allocation-plus-
regression data regime.
"""
Base.@kwdef struct SimulationConfig
    n_subjects::Int = 96
    n_features::Int = 6
    min_repeats::Int = 14
    max_repeats::Int = 22
end

"""
    SimulationResult

Output from [`simulate_glam_style_regression_data`](@ref).

Fields:
- `data`: labeled patient matrices and binary responses ready for SOMSOS.
- `true_beta`: true regression coefficients.
- `true_gamma`: active-feature indicators.
- `true_t`: true scalar mixing weight.
- `true_intercept`: true logistic intercept.
- `true_component2_prob`: patient-specific probability of component 2.
- `true_global_means`: global component means used to simulate patient-specific means.
"""
struct SimulationResult
    data::ScalarOnMatrixData
    true_beta::Vector{Float64}
    true_gamma::Vector{Int}
    true_t::Float64
    true_intercept::Float64
    true_component2_prob::Vector{Float64}
    true_global_means::Matrix{Float64}
end

@inline sigmoid(x::Float64) = x >= 0 ? inv(1 + exp(-x)) : exp(x) / (1 + exp(x))
@inline log1pexp(x::Float64) = x > 0 ? x + log1p(exp(-x)) : log1p(exp(x))
@inline bernoulli_loglik(y::Int, eta::Float64) = y * eta - log1pexp(eta)

function logistic_loglik_sum(y::Vector{Int}, eta::Vector{Float64})
    total = 0.0
    @inbounds for i in eachindex(y)
        total += bernoulli_loglik(y[i], eta[i])
    end
    return total
end

function _make_true_parameters(
    p::Int;
    active_indices::Vector{Int},
    active_values::Vector{Float64},
)
    beta = zeros(p)
    gamma = zeros(Int, p)
    for (val, idx) in zip(active_values, active_indices)
        1 <= idx <= p || continue
        gamma[idx] = 1
        beta[idx] = val
    end

    global_means = zeros(2, p)
    for ell in 1:p
        if gamma[ell] == 1
            global_means[1, ell] = -1.5 - 0.15 * ell
            global_means[2, ell] = 1.5 + 0.15 * ell
        else
            global_means[1, ell] = -0.15 * iseven(ell)
            global_means[2, ell] = 0.15 * iseven(ell)
        end
    end
    return beta, gamma, global_means
end

"""
    simulate_glam_style_regression_data([rng], config=SimulationConfig();
        active_indices=[1], active_values=[2.5], true_t=0.8, true_intercept=1.0)

Simulate the full labeled repeated-measurement plus binary-response regime used by the
original GLAM demo.

This is primarily intended for testing and README examples. The repeated-measurement
portion matches the earlier `HierarchicalMogSampler.jl` package in spirit and label
convention: labels are returned as `1/2`.
"""
function simulate_glam_style_regression_data(
    rng::AbstractRNG,
    config::SimulationConfig = SimulationConfig();
    active_indices::Vector{Int} = [1],
    active_values::Vector{Float64} = [2.5],
    true_t::Float64 = 0.80,
    true_intercept::Float64 = 1.0,
)
    config.n_subjects > 0 || throw(ArgumentError("n_subjects must be positive."))
    config.n_features > 0 || throw(ArgumentError("n_features must be positive."))
    config.min_repeats > 0 || throw(ArgumentError("min_repeats must be positive."))
    config.max_repeats >= config.min_repeats ||
        throw(ArgumentError("max_repeats must be at least min_repeats."))

    n = config.n_subjects
    p = config.n_features
    beta, gamma, global_means = _make_true_parameters(
        p;
        active_indices = active_indices,
        active_values = active_values,
    )

    Sigma_true = [
        Matrix(Diagonal(fill(0.45, p))),
        Matrix(Diagonal(fill(0.60, p))),
    ]

    patients = Vector{LabeledPatientMatrix}(undef, n)
    y = Vector{Int}(undef, n)
    true_component2_prob = Vector{Float64}(undef, n)

    for i in 1:n
        n_i = rand(rng, config.min_repeats:config.max_repeats)
        prob2 = rand(rng, Beta(2.4, 2.0))
        true_component2_prob[i] = prob2

        lambda = [rand(rng, Gamma(5.0, 1 / 5.0)) for _ in 1:2]
        mu = [
            rand(rng, MvNormal(global_means[k, :], Symmetric(Sigma_true[k] / (1.5 * lambda[k]))))
            for k in 1:2
        ]

        Xi = Matrix{Float64}(undef, n_i, p)
        zi = Vector{Int}(undef, n_i)
        sums = zeros(2, p)
        for j in 1:n_i
            k = rand(rng) < prob2 ? 2 : 1
            zi[j] = k
            xij = rand(rng, MvNormal(mu[k], Symmetric(Sigma_true[k] / lambda[k])))
            Xi[j, :] = xij
            sums[k, :] .+= xij
        end

        s1 = sums[1, :] ./ n_i
        s2 = sums[2, :] ./ n_i
        weighted = true_t .* s1 .+ (1 - true_t) .* s2
        eta = true_intercept + dot(weighted, beta .* gamma)
        y[i] = rand(rng) < sigmoid(eta) ? 1 : 0

        patients[i] = LabeledPatientMatrix(Xi, zi)
    end

    return SimulationResult(
        ScalarOnMatrixData(patients, y),
        beta,
        gamma,
        true_t,
        true_intercept,
        true_component2_prob,
        global_means,
    )
end

simulate_glam_style_regression_data(config::SimulationConfig = SimulationConfig(); kwargs...) =
    simulate_glam_style_regression_data(Random.default_rng(), config; kwargs...)

"""
    working_design(design, t)
    working_design(data, t)

Compute the patient-by-feature regression design matrix

`t * component1 + (1 - t) * component2`.
"""
working_design(design::ComponentDesign, t::Float64) = t .* design.component1 .+ (1 - t) .* design.component2
working_design(data::ScalarOnMatrixData, t::Float64) = working_design(build_component_design(data), t)

function _initialize_state(rng::AbstractRNG, design::ComponentDesign, y::Vector{Int})
    p = size(design.component1, 2)
    w = working_design(design, 0.5)
    yc = Float64.(y) .- mean(y)
    scores = zeros(p)
    beta = zeros(p)
    for ell in 1:p
        x = w[:, ell]
        xc = x .- mean(x)
        scores[ell] = abs(dot(xc, yc)) / max(norm(xc) * norm(yc), eps())
        beta[ell] = 0.8 * sign(dot(xc, yc) + 1e-6)
    end
    order = sortperm(scores, rev = true)
    n_active = min(2, p)
    gamma = zeros(Int, p)
    gamma[order[1:n_active]] .= 1
    beta .+= 0.05 .* randn(rng, p)
    mean_y = clamp(mean(y), 1e-3, 1 - 1e-3)
    intercept = log(mean_y / (1 - mean_y))
    omega = n_active / p
    return RegressionState(beta, gamma, intercept, 0.0, omega, 1.0)
end

function _regression_eta(design::ComponentDesign, state::RegressionState)
    t = sigmoid(state.u)
    w = working_design(design, t)
    return state.intercept .+ w * (state.beta .* state.gamma)
end

function _logposterior_beta(
    design::ComponentDesign,
    y::Vector{Int},
    state::RegressionState,
    beta::Vector{Float64},
)
    t = sigmoid(state.u)
    w = working_design(design, t)
    eta = state.intercept .+ w * (beta .* state.gamma)
    ll = logistic_loglik_sum(y, eta)
    grad = transpose(w .* reshape(Float64.(state.gamma), 1, :)) * (Float64.(y) .- sigmoid.(eta))
    grad = vec(grad) .- beta ./ state.tau2
    lp = ll - 0.5 * dot(beta, beta) / state.tau2
    return lp, grad
end

function _logposterior_intercept(
    design::ComponentDesign,
    y::Vector{Int},
    state::RegressionState,
    cfg::RegressionConfig,
    intercept::Float64,
)
    t = sigmoid(state.u)
    w = working_design(design, t)
    eta = intercept .+ w * (state.beta .* state.gamma)
    ll = logistic_loglik_sum(y, eta)
    grad = sum(Float64.(y) .- sigmoid.(eta)) - intercept / (cfg.sigma0^2)
    lp = ll - 0.5 * intercept^2 / (cfg.sigma0^2)
    return lp, grad
end

function _logposterior_u(
    design::ComponentDesign,
    y::Vector{Int},
    state::RegressionState,
    cfg::RegressionConfig,
    u::Float64,
)
    t = sigmoid(u)
    w = working_design(design, t)
    active_beta = state.beta .* state.gamma
    eta = state.intercept .+ w * active_beta
    ll = logistic_loglik_sum(y, eta)
    delta = design.component1 .- design.component2
    dη_dt = delta * active_beta
    dt_du = t * (1 - t)
    grad = sum((Float64.(y) .- sigmoid.(eta)) .* (dt_du .* dη_dt))
    grad += cfg.a_alpha * (1 - t) - cfg.b_alpha * t
    lp = ll + cfg.a_alpha * log(max(t, eps())) + cfg.b_alpha * log(max(1 - t, eps()))
    return lp, grad
end

function _logposterior(
    design::ComponentDesign,
    y::Vector{Int},
    state::RegressionState,
    cfg::RegressionConfig,
)
    eta = _regression_eta(design, state)
    t = sigmoid(state.u)
    lp = logistic_loglik_sum(y, eta)
    lp += -0.5 * state.intercept^2 / (cfg.sigma0^2)
    lp += sum(logpdf.(Normal(0.0, sqrt(state.tau2)), state.beta))
    lp += logpdf(Beta(cfg.a_omega, cfg.b_omega), state.omega)
    for g in state.gamma
        prob = g == 1 ? state.omega : 1 - state.omega
        lp += log(max(prob, eps()))
    end
    lp += logpdf(InverseGamma(cfg.a_tau, cfg.b_tau), state.tau2)
    lp += cfg.a_alpha * log(max(t, eps())) + cfg.b_alpha * log(max(1 - t, eps()))
    return lp
end

function isotropic_logpdf(x::Vector{Float64}, mean::Vector{Float64}, step::Float64)
    d = length(x)
    diff = x .- mean
    return -0.5 * (d * log(2π * step^2) + dot(diff, diff) / step^2)
end

function mala_step_vector!(
    rng::AbstractRNG,
    current::Vector{Float64},
    step::Float64,
    target_grad::Function,
)
    lp, grad = target_grad(current)
    mean_forward = current .+ 0.5 * step^2 .* grad
    proposal = mean_forward .+ step .* randn(rng, length(current))
    lp_prop, grad_prop = target_grad(proposal)
    mean_reverse = proposal .+ 0.5 * step^2 .* grad_prop
    logq_forward = isotropic_logpdf(proposal, mean_forward, step)
    logq_reverse = isotropic_logpdf(current, mean_reverse, step)
    log_accept = lp_prop + logq_reverse - lp - logq_forward
    if log(rand(rng)) < log_accept
        current .= proposal
        return true
    end
    return false
end

function mala_step_scalar(
    rng::AbstractRNG,
    current::Float64,
    step::Float64,
    target_grad::Function,
)
    lp, grad = target_grad(current)
    mean_forward = current + 0.5 * step^2 * grad
    proposal = mean_forward + step * randn(rng)
    lp_prop, grad_prop = target_grad(proposal)
    mean_reverse = proposal + 0.5 * step^2 * grad_prop
    logq_forward = logpdf(Normal(mean_forward, step), proposal)
    logq_reverse = logpdf(Normal(mean_reverse, step), current)
    log_accept = lp_prop + logq_reverse - lp - logq_forward
    if log(rand(rng)) < log_accept
        return proposal, true
    end
    return current, false
end

function _update_gamma!(
    rng::AbstractRNG,
    design::ComponentDesign,
    y::Vector{Int},
    state::RegressionState,
)
    t = sigmoid(state.u)
    w = working_design(design, t)
    eta = _regression_eta(design, state)
    for ell in eachindex(state.gamma)
        contrib = w[:, ell] .* state.beta[ell]
        eta_zero = eta .- state.gamma[ell] .* contrib
        ll_zero = logistic_loglik_sum(y, eta_zero)
        eta_one = eta_zero .+ contrib
        ll_one = logistic_loglik_sum(y, eta_one)
        logodds = log(max(state.omega, eps())) - log(max(1 - state.omega, eps())) + (ll_one - ll_zero)
        newgamma = rand(rng) < sigmoid(logodds) ? 1 : 0
        eta .= eta_zero .+ newgamma .* contrib
        state.gamma[ell] = newgamma
    end
    return nothing
end

function _sample_from_state(
    design::ComponentDesign,
    y::Vector{Int},
    state::RegressionState,
    cfg::RegressionConfig,
)
    lp = _logposterior(design, y, state, cfg)
    return RegressionSample(
        copy(state.beta),
        copy(state.gamma),
        state.intercept,
        sigmoid(state.u),
        state.omega,
        state.tau2,
        lp,
    )
end

function _summarize_samples(
    samples::Vector{RegressionSample},
    beta_accept::Int,
    intercept_accept::Int,
    t_accept::Int,
    n_iters::Int,
)
    isempty(samples) && throw(ArgumentError("At least one saved sample is required to summarize the chain."))
    p = length(samples[1].beta)
    beta = zeros(p)
    pip = zeros(p)
    t = 0.0
    intercept = 0.0
    for sample in samples
        beta .+= sample.beta
        pip .+= sample.gamma
        t += sample.t
        intercept += sample.intercept
    end
    m = length(samples)
    return RegressionSummary(
        beta ./ m,
        pip ./ m,
        t / m,
        intercept / m,
        beta_accept / n_iters,
        intercept_accept / n_iters,
        t_accept / n_iters,
    )
end

"""
    logposterior(data, sample, cfg)
    logposterior(design, y, sample, cfg)

Evaluate the joint log posterior for a saved SOMSOS draw.
"""
function logposterior(
    design::ComponentDesign,
    y::AbstractVector{<:Integer},
    sample::RegressionSample,
    cfg::RegressionConfig,
)
    state = RegressionState(
        copy(sample.beta),
        copy(sample.gamma),
        sample.intercept,
        let t = clamp(sample.t, eps(), 1 - eps())
            log(t / (1 - t))
        end,
        sample.omega,
        sample.tau2,
    )
    return _logposterior(design, Int.(collect(y)), state, cfg)
end

function logposterior(
    data::ScalarOnMatrixData,
    sample::RegressionSample,
    cfg::RegressionConfig,
)
    return logposterior(build_component_design(data), data.y, sample, cfg)
end

"""
    sample_regression([rng], data, cfg, n_iters; burnin=0, thin=1)

Run the scalar-on-matrix spike-and-slab logistic regression sampler on patient-level
labeled matrices.

The input boundary is deliberately simple: every patient contributes a repeated-measurement
matrix and a fixed two-class allocation vector. The function converts these into the same
component-specific patient summaries used in the original GLAM regression stage and then
runs the extracted sampler.
"""
function sample_regression(
    rng::AbstractRNG,
    data::ScalarOnMatrixData,
    cfg::RegressionConfig,
    n_iters::Integer;
    burnin::Integer = 0,
    thin::Integer = 1,
)
    n_iters > 0 || throw(ArgumentError("n_iters must be positive."))
    0 <= burnin < n_iters || throw(ArgumentError("burnin must satisfy 0 <= burnin < n_iters."))
    thin > 0 || throw(ArgumentError("thin must be positive."))

    design = build_component_design(data)
    p = size(design.component1, 2)
    state = _initialize_state(rng, design, data.y)

    samples = RegressionSample[]
    logpost_trace = Float64[]
    t_trace = Float64[]
    active_trace = Int[]
    beta_accept = 0
    intercept_accept = 0
    t_accept = 0

    for iter in 1:n_iters
        _update_gamma!(rng, design, data.y, state)
        state.omega = rand(rng, Beta(cfg.a_omega + sum(state.gamma), cfg.b_omega + p - sum(state.gamma)))
        state.tau2 = rand(rng, InverseGamma(cfg.a_tau + p / 2, cfg.b_tau + dot(state.beta, state.beta) / 2))

        beta_target = β -> _logposterior_beta(design, data.y, state, β)
        beta_accept += mala_step_vector!(rng, state.beta, cfg.beta_step, beta_target) ? 1 : 0

        intercept_target = b0 -> _logposterior_intercept(design, data.y, state, cfg, b0)
        state.intercept, accepted = mala_step_scalar(rng, state.intercept, cfg.intercept_step, intercept_target)
        intercept_accept += accepted ? 1 : 0

        u_target = u -> _logposterior_u(design, data.y, state, cfg, u)
        state.u, accepted = mala_step_scalar(rng, state.u, cfg.u_step, u_target)
        t_accept += accepted ? 1 : 0

        current_t = sigmoid(state.u)
        push!(logpost_trace, _logposterior(design, data.y, state, cfg))
        push!(t_trace, current_t)
        push!(active_trace, sum(state.gamma))

        if iter > burnin && ((iter - burnin) % thin == 0)
            push!(samples, _sample_from_state(design, data.y, state, cfg))
        end
    end

    if isempty(samples)
        push!(samples, _sample_from_state(design, data.y, state, cfg))
    end

    final_sample = _sample_from_state(design, data.y, state, cfg)
    summary = _summarize_samples(samples, beta_accept, intercept_accept, t_accept, n_iters)
    return RegressionResult(samples, summary, logpost_trace, t_trace, active_trace, final_sample)
end

sample_regression(data::ScalarOnMatrixData, cfg::RegressionConfig, n_iters::Integer; kwargs...) =
    sample_regression(Random.default_rng(), data, cfg, n_iters; kwargs...)

end
