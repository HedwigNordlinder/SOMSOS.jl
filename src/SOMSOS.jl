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
       NaiveRegressionSample,
       NaiveRegressionSummary,
       NaiveRegressionResult,
       labeled_patients,
       n_subjects,
       n_features,
       build_component_design,
       working_design,
       naive_average_matrix,
       sample_regression,
       sample_naive_regression,
       predict_probabilities,
       logposterior

"""
    LabeledPatientMatrix(x, z)

One patient's repeated measurements together with a fixed two-component allocation for
each row.

`x` must be an `n_i x p` matrix and `z` must have length `n_i`. Labels may be supplied as
`0/1` or `1/2`; internally they are stored as `1/2`.
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

function _normalize_binary_vector(y::AbstractVector{<:Integer})
    values = Int.(collect(y))
    all(in((0, 1)), values) || throw(ArgumentError("Responses must be binary values in {0,1}."))
    return values
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

Binary scalar responses together with patient-level labeled matrices for the cluster-aware
SOMSOS model.

All patients must have the same number of features. `y` must contain only `0` and `1`.
"""
struct ScalarOnMatrixData
    patients::Vector{LabeledPatientMatrix}
    y::Vector{Int}
    function ScalarOnMatrixData(
        patients::Vector{LabeledPatientMatrix},
        y::AbstractVector{<:Integer},
    )
        isempty(patients) && throw(ArgumentError("ScalarOnMatrixData requires at least one patient."))
        labels = _normalize_binary_vector(y)
        length(patients) == length(labels) || throw(ArgumentError("The response vector must match the number of patients."))
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

Deterministic design matrices derived from labeled patient matrices.

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
    build_component_design(patients)
    build_component_design(data)
    build_component_design(x, z)

Convert labeled patient matrices into the component-specific patient summaries used by the
cluster-aware scalar-on-matrix regression stage.
"""
function build_component_design(patients::Vector{LabeledPatientMatrix})
    isempty(patients) && throw(ArgumentError("At least one labeled patient is required."))
    n = length(patients)
    p = size(patients[1].x, 2)
    component1 = zeros(n, p)
    component2 = zeros(n, p)

    for i in 1:n
        patient = patients[i]
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

build_component_design(data::ScalarOnMatrixData) = build_component_design(data.patients)
build_component_design(
    x::Vector{<:AbstractMatrix{<:Real}},
    z::Vector{<:AbstractVector{<:Integer}},
) = build_component_design(labeled_patients(x, z))

"""
    working_design(design, t)
    working_design(data, t)
    working_design(patients, t)

Compute the patient-by-feature design matrix `t * component1 + (1 - t) * component2`.
"""
working_design(design::ComponentDesign, t::Float64) = t .* design.component1 .+ (1 - t) .* design.component2
working_design(data::ScalarOnMatrixData, t::Float64) = working_design(build_component_design(data), t)
working_design(patients::Vector{LabeledPatientMatrix}, t::Float64) = working_design(build_component_design(patients), t)

"""
    naive_average_matrix(x)
    naive_average_matrix(data)
    naive_average_matrix(patients)

Compute the patient-by-feature matrix obtained by averaging each patient's repeated
measurements without using cluster labels. This is the non-cluster-aware comparator used
in the CLUSSO-style baseline.
"""
function naive_average_matrix(x::Vector{<:AbstractMatrix{<:Real}})
    isempty(x) && throw(ArgumentError("At least one patient matrix is required."))
    p = size(x[1], 2)
    Xbar = zeros(length(x), p)
    for (i, Xi) in pairs(x)
        size(Xi, 2) == p || throw(ArgumentError("Patient $i has a different feature dimension."))
        Xbar[i, :] .= vec(mean(Xi, dims = 1))
    end
    return Xbar
end

naive_average_matrix(patients::Vector{LabeledPatientMatrix}) = naive_average_matrix([patient.x for patient in patients])
naive_average_matrix(data::ScalarOnMatrixData) = naive_average_matrix(data.patients)

"""
    RegressionConfig(; kwargs...)

Hyperparameters and proposal scales for the extracted spike-and-slab logistic regression
sampler.

The same configuration type is used for both the cluster-aware and naive baselines. In the
naive baseline, the `a_alpha`, `b_alpha`, and `u_step` fields are unused because there is
no learned mixing weight `t`.
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

One saved posterior draw from the cluster-aware SOMSOS regression sampler.
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

Posterior summaries averaged across saved cluster-aware draws.

`mean_active_beta` stores the posterior mean of `beta .* gamma`, which is the quantity used
for prediction.
"""
struct RegressionSummary
    mean_beta::Vector{Float64}
    mean_active_beta::Vector{Float64}
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
    NaiveRegressionSample

One saved posterior draw from the naive non-cluster-aware baseline.
"""
struct NaiveRegressionSample
    beta::Vector{Float64}
    gamma::Vector{Int}
    intercept::Float64
    omega::Float64
    tau2::Float64
    logposterior::Float64
end

"""
    NaiveRegressionSummary

Posterior summaries averaged across saved naive baseline draws.

`mean_active_beta` stores the posterior mean of `beta .* gamma`, which is the quantity used
for prediction.
"""
struct NaiveRegressionSummary
    mean_beta::Vector{Float64}
    mean_active_beta::Vector{Float64}
    pip::Vector{Float64}
    mean_intercept::Float64
    beta_acceptance::Float64
    intercept_acceptance::Float64
end

"""
    NaiveRegressionResult

Output from [`sample_naive_regression`](@ref).
"""
struct NaiveRegressionResult
    samples::Vector{NaiveRegressionSample}
    summary::NaiveRegressionSummary
    logposterior_trace::Vector{Float64}
    active_trace::Vector{Int}
    final_sample::NaiveRegressionSample
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
    mean_beta = zeros(p)
    mean_active_beta = zeros(p)
    pip = zeros(p)
    mean_t = 0.0
    mean_intercept = 0.0
    for sample in samples
        mean_beta .+= sample.beta
        mean_active_beta .+= sample.beta .* sample.gamma
        pip .+= sample.gamma
        mean_t += sample.t
        mean_intercept += sample.intercept
    end
    m = length(samples)
    return RegressionSummary(
        mean_beta ./ m,
        mean_active_beta ./ m,
        pip ./ m,
        mean_t / m,
        mean_intercept / m,
        beta_accept / n_iters,
        intercept_accept / n_iters,
        t_accept / n_iters,
    )
end

function _predict_probabilities(design::ComponentDesign, active_beta::Vector{Float64}, intercept::Float64, t::Float64)
    w = working_design(design, t)
    return sigmoid.(intercept .+ w * active_beta)
end

"""
    predict_probabilities(sample_or_summary_or_result, patients)
    predict_probabilities(sample_or_summary_or_result, data)
    predict_probabilities(sample_or_summary_or_result, design)
    predict_probabilities(sample_or_summary_or_result, x, z)

Predict binary-response probabilities from the cluster-aware SOMSOS model using either a
single draw, a posterior summary, or a full result object.
"""
function predict_probabilities(sample::RegressionSample, design::ComponentDesign)
    return _predict_probabilities(design, sample.beta .* sample.gamma, sample.intercept, sample.t)
end

function predict_probabilities(summary::RegressionSummary, design::ComponentDesign)
    return _predict_probabilities(design, summary.mean_active_beta, summary.mean_intercept, summary.mean_t)
end

predict_probabilities(result::RegressionResult, design::ComponentDesign) = predict_probabilities(result.summary, design)
predict_probabilities(sample::RegressionSample, patients::Vector{LabeledPatientMatrix}) = predict_probabilities(sample, build_component_design(patients))
predict_probabilities(summary::RegressionSummary, patients::Vector{LabeledPatientMatrix}) = predict_probabilities(summary, build_component_design(patients))
predict_probabilities(result::RegressionResult, patients::Vector{LabeledPatientMatrix}) = predict_probabilities(result.summary, build_component_design(patients))
predict_probabilities(sample::RegressionSample, data::ScalarOnMatrixData) = predict_probabilities(sample, data.patients)
predict_probabilities(summary::RegressionSummary, data::ScalarOnMatrixData) = predict_probabilities(summary, data.patients)
predict_probabilities(result::RegressionResult, data::ScalarOnMatrixData) = predict_probabilities(result.summary, data.patients)
predict_probabilities(sample::RegressionSample, x::Vector{<:AbstractMatrix{<:Real}}, z::Vector{<:AbstractVector{<:Integer}}) =
    predict_probabilities(sample, labeled_patients(x, z))
predict_probabilities(summary::RegressionSummary, x::Vector{<:AbstractMatrix{<:Real}}, z::Vector{<:AbstractVector{<:Integer}}) =
    predict_probabilities(summary, labeled_patients(x, z))
predict_probabilities(result::RegressionResult, x::Vector{<:AbstractMatrix{<:Real}}, z::Vector{<:AbstractVector{<:Integer}}) =
    predict_probabilities(result.summary, labeled_patients(x, z))

"""
    logposterior(data, sample, cfg)
    logposterior(design, y, sample, cfg)

Evaluate the joint log posterior for a saved cluster-aware SOMSOS draw.
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
    return _logposterior(design, _normalize_binary_vector(y), state, cfg)
end

function logposterior(
    data::ScalarOnMatrixData,
    sample::RegressionSample,
    cfg::RegressionConfig,
)
    return logposterior(build_component_design(data), data.y, sample, cfg)
end

function logposterior(
    x::Vector{<:AbstractMatrix{<:Real}},
    z::Vector{<:AbstractVector{<:Integer}},
    y::AbstractVector{<:Integer},
    sample::RegressionSample,
    cfg::RegressionConfig,
)
    return logposterior(build_component_design(x, z), y, sample, cfg)
end

"""
    sample_regression([rng], data, cfg, n_iters; burnin=0, thin=1)
    sample_regression([rng], x, z, y, cfg, n_iters; burnin=0, thin=1)

Run the cluster-aware spike-and-slab logistic regression sampler on patient-level labeled
matrices.
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

        push!(logpost_trace, _logposterior(design, data.y, state, cfg))
        push!(t_trace, sigmoid(state.u))
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

sample_regression(
    rng::AbstractRNG,
    x::Vector{<:AbstractMatrix{<:Real}},
    z::Vector{<:AbstractVector{<:Integer}},
    y::AbstractVector{<:Integer},
    cfg::RegressionConfig,
    n_iters::Integer;
    kwargs...,
) = sample_regression(rng, ScalarOnMatrixData(x, z, y), cfg, n_iters; kwargs...)

sample_regression(data::ScalarOnMatrixData, cfg::RegressionConfig, n_iters::Integer; kwargs...) =
    sample_regression(Random.default_rng(), data, cfg, n_iters; kwargs...)

sample_regression(
    x::Vector{<:AbstractMatrix{<:Real}},
    z::Vector{<:AbstractVector{<:Integer}},
    y::AbstractVector{<:Integer},
    cfg::RegressionConfig,
    n_iters::Integer;
    kwargs...,
) = sample_regression(Random.default_rng(), x, z, y, cfg, n_iters; kwargs...)

function _initialize_naive_state(rng::AbstractRNG, X::Matrix{Float64}, y::Vector{Int})
    p = size(X, 2)
    yc = Float64.(y) .- mean(y)
    scores = zeros(p)
    beta = zeros(p)
    for ell in 1:p
        x = X[:, ell]
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

function _naive_eta(X::Matrix{Float64}, state::RegressionState)
    return state.intercept .+ X * (state.beta .* state.gamma)
end

function _logposterior_beta_naive(
    X::Matrix{Float64},
    y::Vector{Int},
    state::RegressionState,
    beta::Vector{Float64},
)
    eta = state.intercept .+ X * (beta .* state.gamma)
    ll = logistic_loglik_sum(y, eta)
    grad = transpose(X .* reshape(Float64.(state.gamma), 1, :)) * (Float64.(y) .- sigmoid.(eta))
    grad = vec(grad) .- beta ./ state.tau2
    lp = ll - 0.5 * dot(beta, beta) / state.tau2
    return lp, grad
end

function _logposterior_intercept_naive(
    X::Matrix{Float64},
    y::Vector{Int},
    state::RegressionState,
    cfg::RegressionConfig,
    intercept::Float64,
)
    eta = intercept .+ X * (state.beta .* state.gamma)
    ll = logistic_loglik_sum(y, eta)
    grad = sum(Float64.(y) .- sigmoid.(eta)) - intercept / (cfg.sigma0^2)
    lp = ll - 0.5 * intercept^2 / (cfg.sigma0^2)
    return lp, grad
end

function _logposterior_naive(
    X::Matrix{Float64},
    y::Vector{Int},
    state::RegressionState,
    cfg::RegressionConfig,
)
    eta = _naive_eta(X, state)
    lp = logistic_loglik_sum(y, eta)
    lp += -0.5 * state.intercept^2 / (cfg.sigma0^2)
    lp += sum(logpdf.(Normal(0.0, sqrt(state.tau2)), state.beta))
    lp += logpdf(Beta(cfg.a_omega, cfg.b_omega), state.omega)
    for g in state.gamma
        prob = g == 1 ? state.omega : 1 - state.omega
        lp += log(max(prob, eps()))
    end
    lp += logpdf(InverseGamma(cfg.a_tau, cfg.b_tau), state.tau2)
    return lp
end

function _update_gamma_naive!(
    rng::AbstractRNG,
    X::Matrix{Float64},
    y::Vector{Int},
    state::RegressionState,
)
    eta = _naive_eta(X, state)
    for ell in eachindex(state.gamma)
        contrib = X[:, ell] .* state.beta[ell]
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

function _naive_sample_from_state(
    X::Matrix{Float64},
    y::Vector{Int},
    state::RegressionState,
    cfg::RegressionConfig,
)
    lp = _logposterior_naive(X, y, state, cfg)
    return NaiveRegressionSample(
        copy(state.beta),
        copy(state.gamma),
        state.intercept,
        state.omega,
        state.tau2,
        lp,
    )
end

function _summarize_naive_samples(
    samples::Vector{NaiveRegressionSample},
    beta_accept::Int,
    intercept_accept::Int,
    n_iters::Int,
)
    isempty(samples) && throw(ArgumentError("At least one saved sample is required to summarize the chain."))
    p = length(samples[1].beta)
    mean_beta = zeros(p)
    mean_active_beta = zeros(p)
    pip = zeros(p)
    mean_intercept = 0.0
    for sample in samples
        mean_beta .+= sample.beta
        mean_active_beta .+= sample.beta .* sample.gamma
        pip .+= sample.gamma
        mean_intercept += sample.intercept
    end
    m = length(samples)
    return NaiveRegressionSummary(
        mean_beta ./ m,
        mean_active_beta ./ m,
        pip ./ m,
        mean_intercept / m,
        beta_accept / n_iters,
        intercept_accept / n_iters,
    )
end

function _predict_probabilities_naive(X::Matrix{Float64}, active_beta::Vector{Float64}, intercept::Float64)
    return sigmoid.(intercept .+ X * active_beta)
end

"""
    predict_probabilities(sample_or_summary_or_result, x)
    predict_probabilities(sample_or_summary_or_result, data)

Predict binary-response probabilities from the naive non-cluster-aware baseline using a
single draw, a posterior summary, or a full result object.
"""
function predict_probabilities(sample::NaiveRegressionSample, X::Matrix{Float64})
    return _predict_probabilities_naive(X, sample.beta .* sample.gamma, sample.intercept)
end

function predict_probabilities(summary::NaiveRegressionSummary, X::Matrix{Float64})
    return _predict_probabilities_naive(X, summary.mean_active_beta, summary.mean_intercept)
end

predict_probabilities(result::NaiveRegressionResult, X::Matrix{Float64}) = predict_probabilities(result.summary, X)
predict_probabilities(sample::NaiveRegressionSample, x::Vector{<:AbstractMatrix{<:Real}}) = predict_probabilities(sample, naive_average_matrix(x))
predict_probabilities(summary::NaiveRegressionSummary, x::Vector{<:AbstractMatrix{<:Real}}) = predict_probabilities(summary, naive_average_matrix(x))
predict_probabilities(result::NaiveRegressionResult, x::Vector{<:AbstractMatrix{<:Real}}) = predict_probabilities(result.summary, naive_average_matrix(x))
predict_probabilities(sample::NaiveRegressionSample, data::ScalarOnMatrixData) = predict_probabilities(sample, naive_average_matrix(data))
predict_probabilities(summary::NaiveRegressionSummary, data::ScalarOnMatrixData) = predict_probabilities(summary, naive_average_matrix(data))
predict_probabilities(result::NaiveRegressionResult, data::ScalarOnMatrixData) = predict_probabilities(result.summary, naive_average_matrix(data))

"""
    logposterior(x, y, sample, cfg)
    logposterior(data, sample, cfg)

Evaluate the joint log posterior for a saved naive baseline draw.
"""
function logposterior(
    X::AbstractMatrix{<:Real},
    y::AbstractVector{<:Integer},
    sample::NaiveRegressionSample,
    cfg::RegressionConfig,
)
    state = RegressionState(copy(sample.beta), copy(sample.gamma), sample.intercept, 0.0, sample.omega, sample.tau2)
    return _logposterior_naive(Matrix{Float64}(X), _normalize_binary_vector(y), state, cfg)
end

function logposterior(
    x::Vector{<:AbstractMatrix{<:Real}},
    y::AbstractVector{<:Integer},
    sample::NaiveRegressionSample,
    cfg::RegressionConfig,
)
    return logposterior(naive_average_matrix(x), y, sample, cfg)
end

function logposterior(
    data::ScalarOnMatrixData,
    sample::NaiveRegressionSample,
    cfg::RegressionConfig,
)
    return logposterior(data.patients, data.y, sample, cfg)
end

function logposterior(
    patients::Vector{LabeledPatientMatrix},
    y::AbstractVector{<:Integer},
    sample::NaiveRegressionSample,
    cfg::RegressionConfig,
)
    return logposterior(naive_average_matrix(patients), y, sample, cfg)
end

"""
    sample_naive_regression([rng], x, y, cfg, n_iters; burnin=0, thin=1)
    sample_naive_regression([rng], data, cfg, n_iters; burnin=0, thin=1)

Run the naive non-cluster-aware spike-and-slab logistic baseline that averages each
patient's repeated measurements before fitting the regression.
"""
function sample_naive_regression(
    rng::AbstractRNG,
    x::Vector{<:AbstractMatrix{<:Real}},
    y::AbstractVector{<:Integer},
    cfg::RegressionConfig,
    n_iters::Integer;
    burnin::Integer = 0,
    thin::Integer = 1,
)
    n_iters > 0 || throw(ArgumentError("n_iters must be positive."))
    0 <= burnin < n_iters || throw(ArgumentError("burnin must satisfy 0 <= burnin < n_iters."))
    thin > 0 || throw(ArgumentError("thin must be positive."))

    X = naive_average_matrix(x)
    yvec = _normalize_binary_vector(y)
    size(X, 1) == length(yvec) || throw(ArgumentError("The response vector must match the number of patients."))

    p = size(X, 2)
    state = _initialize_naive_state(rng, X, yvec)
    samples = NaiveRegressionSample[]
    logpost_trace = Float64[]
    active_trace = Int[]
    beta_accept = 0
    intercept_accept = 0

    for iter in 1:n_iters
        _update_gamma_naive!(rng, X, yvec, state)
        state.omega = rand(rng, Beta(cfg.a_omega + sum(state.gamma), cfg.b_omega + p - sum(state.gamma)))
        state.tau2 = rand(rng, InverseGamma(cfg.a_tau + p / 2, cfg.b_tau + dot(state.beta, state.beta) / 2))

        beta_target = β -> _logposterior_beta_naive(X, yvec, state, β)
        beta_accept += mala_step_vector!(rng, state.beta, cfg.beta_step, beta_target) ? 1 : 0

        intercept_target = b0 -> _logposterior_intercept_naive(X, yvec, state, cfg, b0)
        state.intercept, accepted = mala_step_scalar(rng, state.intercept, cfg.intercept_step, intercept_target)
        intercept_accept += accepted ? 1 : 0

        push!(logpost_trace, _logposterior_naive(X, yvec, state, cfg))
        push!(active_trace, sum(state.gamma))

        if iter > burnin && ((iter - burnin) % thin == 0)
            push!(samples, _naive_sample_from_state(X, yvec, state, cfg))
        end
    end

    if isempty(samples)
        push!(samples, _naive_sample_from_state(X, yvec, state, cfg))
    end

    final_sample = _naive_sample_from_state(X, yvec, state, cfg)
    summary = _summarize_naive_samples(samples, beta_accept, intercept_accept, n_iters)
    return NaiveRegressionResult(samples, summary, logpost_trace, active_trace, final_sample)
end

sample_naive_regression(
    rng::AbstractRNG,
    patients::Vector{LabeledPatientMatrix},
    y::AbstractVector{<:Integer},
    cfg::RegressionConfig,
    n_iters::Integer;
    kwargs...,
) = sample_naive_regression(rng, [patient.x for patient in patients], y, cfg, n_iters; kwargs...)

sample_naive_regression(
    rng::AbstractRNG,
    data::ScalarOnMatrixData,
    cfg::RegressionConfig,
    n_iters::Integer;
    kwargs...,
) = sample_naive_regression(rng, data.patients, data.y, cfg, n_iters; kwargs...)

sample_naive_regression(
    x::Vector{<:AbstractMatrix{<:Real}},
    y::AbstractVector{<:Integer},
    cfg::RegressionConfig,
    n_iters::Integer;
    kwargs...,
) = sample_naive_regression(Random.default_rng(), x, y, cfg, n_iters; kwargs...)

sample_naive_regression(
    patients::Vector{LabeledPatientMatrix},
    y::AbstractVector{<:Integer},
    cfg::RegressionConfig,
    n_iters::Integer;
    kwargs...,
) = sample_naive_regression(Random.default_rng(), patients, y, cfg, n_iters; kwargs...)

sample_naive_regression(data::ScalarOnMatrixData, cfg::RegressionConfig, n_iters::Integer; kwargs...) =
    sample_naive_regression(Random.default_rng(), data, cfg, n_iters; kwargs...)

end
