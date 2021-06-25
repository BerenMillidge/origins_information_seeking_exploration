# Code to reproduce figure 1 numerical experiments for the paper "Understanding the Origins of Information-Seeking Exploratin in Probabilistic Objectives for Control"
# Essentially what we do here is demonstrate the difference between an evidence "maximizing" objective vs a divergence "probability matching" objective in a simple univariate scenario where we have two overlapping Gaussians
# as our "desire distribution". The evidence objective finds the peak of the distribution and concentrates all probability mass there. The divergence objective explicitly tries to match the desire distribution in all its modes.
# Even though both the desire and evidence objective can fit two Gaussians, the evidence objective nevertheless only utilizes one Gaussian and collapses the weight of the other to zero.

using Plots
using LinearAlgebra
using Statistics
using Distributions
using Optim

e = MathConstants.e


### Utility Function ###

function sigmoid(x)
    return 1 / (1 + exp(-x))
end
function Gaussian_cdf(x, mu, var)
    return (1 / sqrt(2 * pi * var)) * exp(-(x - mu)^2 / (2 * var))
end

function Gaussian_KL(mu1, mu2, var1, var2)
    log(sqrt(var2) / sqrt(var1)) + (var1 + (mu1 - mu2)^2) / (2 * var2) - 0.5
end

function Gaussian_entropy(mu, var)
    return log(sqrt(var * 2 * pi * e))
end

function approximate_KL(cdfs1, cdfs2)
    sum = 0
    for i in 1:length(cdfs1)
        sum += cdfs1[i] * log(cdfs1[i] / cdfs2[i])
    end
    return sum
end

function approximate_evidence(cdfs1, cdfs2)
    return sum(cdfs1 .* -log.(cdfs2))
end
function approximate_entropy(cdfs)
    return sum(cdfs .* log.(cdfs))
end

# Create our desire distribution cdf
function get_desired_cdf(mu_a=1, mu_b=4, var_1=1, var_b=0.4)
    xs = [i*0.01 for i in -500:1000]
    cdfs = (0.5 * Gaussian_cdf.(xs, mu_a, var_a)) + (0.5 * Gaussian_cdf.(xs, mu_b, var_b))
    return cdfs
end

### loss functions for evidence and divergence objective ###

function compute_divergence_loss(mu1, mu2, logvar1, logvar2, alpha,beta)
    lambda = 1e6
    desired_cdfs = get_desired_cdf()
    xs = [i*0.01 for i in -500:1000]
    cdfs = (sigmoid(alpha) .* Gaussian_cdf.(xs, mu1, exp(logvar1))) + (sigmoid(beta) .* Gaussian_cdf.(xs, mu2, exp(logvar2)))
    loss = approximate_KL(cdfs / sum(cdfs), desired_cdfs / sum(desired_cdfs))
    # ensure that constraint is respected
    loss += lambda * ((sigmoid(alpha) + sigmoid(beta)) - 1)^2
    return loss
end
function compute_evidence_loss(mu1, mu2, logvar1, logvar2, alpha,beta)
    lambda = 1e6
    desired_cdfs = get_desired_cdf()
    xs = [i*0.01 for i in -500:1000]
    cdfs = (sigmoid(alpha) .* Gaussian_cdf.(xs, mu1, exp(logvar1))) + (sigmoid(beta) .* Gaussian_cdf.(xs, mu2, exp(logvar2)))
    loss = approximate_evidence(cdfs / sum(cdfs), desired_cdfs / sum(desired_cdfs))
    # ensure that constraint is respected
    loss += lambda * ((sigmoid(alpha) + sigmoid(beta)) - 1)^2
    return loss
end

### Optimize using nonlinear optimization ###

function optimize_loss(loss_fn)
    init_params = zeros(6)
    init_params[1:4] = rand(4)
    print("$init_params \n")
    if loss_fn == "divergence"
        ll_func = compute_divergence_loss
    elseif loss_fn == "evidence"
        ll_func = compute_evidence_loss
    else
        error("Loss function not recognized")
    end
    opt = optimize(init_params -> ll_func(init_params...), init_params)
    output_params = Optim.minimizer(opt)
    final_loss = Optim.minimum(opt)
    return output_params,final_loss
end

### and plot ###

function plot_cdf(params,loss_fn)
    params, final_loss = optimize_loss(loss_fn)
    mu1, mu2, logvar1, logvar2, alpha,beta = params
    xs = [i*0.01 for i in -500:1000]
    desired_cdf = get_desired_cdf()
    cdfs = (sigmoid(alpha) .* Gaussian_cdf.(xs, mu1, exp(logvar1))) + (sigmoid(beta) .* Gaussian_cdf.(xs, mu2, exp(logvar2)))
    plot(xs ./ 10, cdfs/sum(cdfs), xaxis="X value", yaxis="Probability Density", label="Predicted Density")
    if loss_fn == "evidence"
        plot!(xs ./10, desired_cdf, title="Peak Finding with Evidence Objectives",linestyle=:dash,label="Desired Density")
    else
        plot!(xs ./10, desired_cdf / sum(desired_cdf),title="Probability Matching with Divergence objective",linestyle=:dash,label="Desired Density") # this is just for visibility
    end
    savefig("figures/$loss_fn")
    #return cdfs
end

plot_cdf(params,"evidence")
plot_cdf(params, "divergence")
