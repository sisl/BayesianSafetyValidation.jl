using Revise
using AbstractGPs
import AbstractGPs.KernelFunctions: ColVecs, SqExponentialKernel

# function AbstractGPs.posterior(fx::AbstractGPs.FiniteGP{<:AbstractGPs.PosteriorGP}, y::AbstractVector{<:Real})
#     m2 = mean(fx.f.prior, fx.x)
#     δ2 = y - m2
#     C12 = cov(fx.f.prior, fx.f.data.x, fx.x)
#     C22 = cov(fx.f.prior, fx.x) + fx.Σy
#     chol = AbstractGPs.update_chol(fx.f.data.C, C12, C22)
#     δ = vcat(fx.f.data.δ, δ2)
#     α = chol \ δ
#     # @show typeof(fx.f.data.x)
#     # @show typeof(fx.x)
#     x = vcat(fx.f.data.x, fx.x)
#     # @show typeof(x)
#     # @show which(vcat, [typeof(fx.f.data.x),typeof(fx.x)])
#     return AbstractGPs.PosteriorGP(fx.f.prior , (α=α, C=chol, x=x, δ=δ))
# end

Base.vcat(x1::ColVecs, x2::ColVecs) = ColVecs(hcat(x1.X, x2.X))

gp = GP(SqExponentialKernel())
n = 10

X = ColVecs(randn(2,n))
Y = rand(0:1, n)
# @show which(posterior, [typeof(gp(X)), typeof(Y)]) 
gp = posterior(gp(X), Y)

X = ColVecs(randn(2,1))
Y = rand(0:1, 1)
# @show which(posterior, [typeof(gp(X)), typeof(Y)]) 
gp = posterior(gp(X), Y)

X = ColVecs(randn(2,1))
Y = rand(0:1, 1)
# @show which(posterior, [typeof(gp(X)), typeof(Y)])
gp = posterior(gp(X), Y)

# Y′ = rand(0:1, 1)
# mean(gp′(X′))[1]
# sqrt(cov(gp′(X′))[1])

nothing # REPL