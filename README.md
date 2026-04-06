# SOMSOS.jl

`SOMSOS.jl` is a Julia package for the scalar-on-matrix spike-and-slab logistic regression
stage extracted from the GLAM demo code in this workspace.

The package starts from a clear boundary:

- each patient has a repeated-measurement matrix `x_i`,
- each row of that matrix already has a two-class allocation label,
- each patient has a binary scalar response `y_i`.

From that input, `SOMSOS.jl` builds the exact patient-level design used in the original
repository and runs the extracted posterior sampler.

## What The Package Does

For each patient, the package reduces the labeled repeated-measurement matrix into two
component-specific patient summaries:

- `component1[i, :] = sum(rows assigned to component 1) / n_i`
- `component2[i, :] = sum(rows assigned to component 2) / n_i`

where `n_i` is the patient's total number of repeated measurements.

This detail matters: these are **not** within-component means. They are the same
component-specific sums divided by the total patient repeat count used by the original GLAM
regression code.

The regression then uses the working design

```math
W(t) = t \cdot component1 + (1 - t) \cdot component2
```

inside a logistic spike-and-slab regression:

```math
y_i \sim Bernoulli(logit^{-1}(intercept + W_i(t)^T (\beta \odot \gamma)))
```

with:

- Bernoulli inclusion indicators `gamma`,
- Gaussian slab coefficients `beta`,
- Beta-Bernoulli sparsity through `omega`,
- inverse-Gamma slab variance `tau2`,
- a Beta prior on `t`,
- MALA updates for `beta`, `intercept`, and the unconstrained `u = logit(t)`.

## What The Package Does Not Do

- It does not infer the latent row-level allocations.
- It does not fit the allocation-stage hierarchical mixture model.
- It does not implement arbitrary-`K` mixtures.
- It does not own benchmark or simulation code; those now live in `GLAM.jl`.
- It does not generalize beyond the binary logistic regression used in the original demo.

## Interplay With `HierarchicalMogSampler.jl`

This package is meant to consume the outputs of the earlier
[`HierarchicalMogSampler.jl`](../HierarchicalMogSampler.jl/README.md) package.

If you already have:

- a vector of patient matrices `x`,
- a saved allocation draw `sample.z` from `HierarchicalMogSampler`,
- binary patient outcomes `y`,

then the handoff is simply:

```julia
using SOMSOS

data = ScalarOnMatrixData(x, sample.z, y)
result = sample_regression(data, RegressionConfig(), 1_000; burnin = 400, thin = 10)
```

`ScalarOnMatrixData` accepts allocations in either `0/1` or `1/2` form and stores them as
`1/2` internally. That makes it compatible with both the original GLAM demo code and the
cleaner label convention used in `HierarchicalMogSampler.jl`.

## Installation

```julia
using Pkg
Pkg.develop(path="SOMSOS.jl")
```

## Quick Demo

The example below is mirrored in
[`examples/readme_demo.jl`](examples/readme_demo.jl). It builds a small labeled toy dataset
in-place, fits both the cluster-aware regression and the naive non-cluster-aware baseline,
and compares their posterior inclusion probabilities. For reproducible simulation regimes
and multi-replicate benchmarks, use `GLAM.jl`.

```julia
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
```

## Public API

### Core Input Types

- `LabeledPatientMatrix(x, z)`
  One patient's repeated-measurement matrix and row allocations.
- `ScalarOnMatrixData(patients, y)`
  Binary outcomes plus labeled patient matrices.
- `ScalarOnMatrixData(x, z, y)`
  Convenience constructor from raw matrices and allocation vectors.

### Derived Design

- `build_component_design(data)`
  Build the deterministic component-specific patient summary matrices.
- `working_design(design, t)` or `working_design(data, t)`
  Form the working regression design at a given `t`.
- `naive_average_matrix(data)` or `naive_average_matrix(x)`
  Build the non-cluster-aware patient-by-feature matrix used by the baseline comparator.

### Sampler API

- `sample_regression([rng], data, cfg, n_iters; burnin=0, thin=1)`
  Run the cluster-aware posterior sampler.
- `sample_naive_regression([rng], data_or_x, cfg, n_iters; burnin=0, thin=1)`
  Run the naive non-cluster-aware baseline on patient averages.
- `logposterior(data, sample, cfg)`
  Recompute the log posterior for a saved draw.
- `predict_probabilities(result_or_summary_or_sample, data_or_design)`
  Compute fitted response probabilities for either the cluster-aware or naive model.

### Result Types

- `RegressionSample`
  One saved posterior draw.
- `RegressionSummary`
  Mean coefficients, mean active coefficients, posterior inclusion probabilities, mean `t`,
  mean intercept, and acceptance rates.
- `RegressionResult`
  Saved draws, summary, traces, and the final state.
- `NaiveRegressionSummary`
  Mean coefficients, mean active coefficients, posterior inclusion probabilities, mean
  intercept, and acceptance rates for the naive baseline.
- `NaiveRegressionResult`
  Saved draws, summary, traces, and the final state for the naive baseline.

## Notes On Input Labels

The package accepts either:

- `0/1` allocations from the original GLAM demo code, or
- `1/2` allocations from `HierarchicalMogSampler.jl`.

Internally, SOMSOS always uses `1/2`.

## Repository Intent

This repository is intentionally narrow. It exists to provide a clean, documented package
for the scalar-on-matrix regression stage only, with an input boundary that makes it easy
to pair with a separate allocation sampler. `GLAM.jl` is the package that owns
reproducible simulation regimes and benchmark orchestration across `HierarchicalMogSampler`
and `SOMSOS`.
