module ellipsoidal

import JuMP, PiecewiseLinearOpt, QuadGK

export getquestion, momentmatchingupdate

type BINQUADData
    prodvars::Dict{Tuple{JuMP.Variable,JuMP.Variable},JuMP.Variable}
    BINQUADData() = new(Dict{Tuple{JuMP.Variable,JuMP.Variable},JuMP.Variable}())
end

function initBINQUAD!(m::JuMP.Model)
    if !haskey(m.ext, :BINQUAD)
        m.ext[:BINQUAD] = BINQUADData()
    end
    nothing
end


function linquad(m::JuMP.Model,qexpr::JuMP.QuadExpr)
    initBINQUAD!(m)
    n = length(qexpr.qvars1)
    expr = JuMP.AffExpr()
    for i in 1:n
        if !haskey(m.ext[:BINQUAD].prodvars, (qexpr.qvars1[i],qexpr.qvars2[i]))
            prodvar=m.ext[:BINQUAD].prodvars[(qexpr.qvars1[i],qexpr.qvars2[i])]=JuMP.@variable(m,category = :Bin)
            JuMP.setname(prodvar,string("_",JuMP.getname(qexpr.qvars1[i]),"*",JuMP.getname(qexpr.qvars2[i]),"_"))
        else
            prodvar=m.ext[:BINQUAD].prodvars[(qexpr.qvars1[i],qexpr.qvars2[i])]
        end
        push!(expr,qexpr.qcoeffs[i],prodvar)
        JuMP.@constraint(m, prodvar <= qexpr.qvars1[i])
        JuMP.@constraint(m, prodvar <= qexpr.qvars2[i])
        JuMP.@constraint(m, prodvar >= qexpr.qvars1[i] + qexpr.qvars2[i] - 1)
    end
    expr
end

function getquestion(μ,Σ,mip_solver,k=3,variancefuction=qgk_deff)

  n = size(Σ,1)
  m = JuMP.Model(solver=mip_solver)

  # define variables for linearization
  JuMP.@variable(m, 0 <= x[1:n] <= 1, Int)
  JuMP.@variable(m, 0 <= y[1:n] <= 1, Int)
  # x ≠ y
  JuMP.@constraint(m, linquad(m,(x-y)⋅(x-y)) >= 1)
  # v = x-y, β ∼ 𝒩(μ,Σ), v⋅β ∼ 𝒩(μᵥ,σ²), μᵥ = μ⋅v, σ² = v'*Σ*v
  JuMP.@variable(m, μᵥ)
  JuMP.@constraint(m, μᵥ == μ⋅(x-y) )
  JuMP.@variable(m, σ² >=0)
  JuMP.@constraint(m, σ² == linquad(m,(x-y)⋅(Σ*(x-y))))
  # (x-y)'*Σ*(x-y) <= eigmax(Σ) ||x-y||₂ <= eigmax(Σ)*n
  σ̅² = eigmax(Σ)*n
  # (x-y)'*Σ*(x-y) >= eigmin(Σ) ||x-y||₂ >= eigmin(Σ)   ( x ≠ y )
  σ̲² = eigmin(Σ)
  μ̅ᵥ = norm(μ,1)

  μᵥnpoints = 2^k - 1
  μᵥpoints = []
  if μ̅ᵥ > 1e-6
      μᵥpoints = 0:μ̅ᵥ/μᵥnpoints:μ̅ᵥ+(μ̅ᵥ/μᵥnpoints)/2
  else
      μᵥpoints = 0:1e-6:1e-6 #0:0
  end
  σ²points = []
  σ²range = σ̅² - σ̲²
  σ²npoints = 2^k-1
  if σ²range > 1e-6
      σ²points = σ̲²:σ²range/σ²npoints:σ̅²+(σ²range/σ²npoints)/2
  else
      σ²points = σ̲²:1e-6:σ̲²+1e-6 #σ̲²: σ̲²
  end
  pwl = PiecewiseLinearOpt.BivariatePWLFunction(μᵥpoints, σ²points, (μᵥ,σ²) -> variancefuction(μᵥ,sqrt(σ²)); pattern=:UnionJack)

  obj = PiecewiseLinearOpt.piecewiselinear(m, μᵥ, σ², pwl; method=:Logarithmic)
  JuMP.@objective(m, Min,  obj )

  status = JuMP.solve(m)

  if status == :UserLimit
      if mapreduce(isnan,|,JuMP.getvalue(x)) || mapreduce(isnan,|,JuMP.getvalue(y))
          return []
      elseif norm(JuMP.getvalue(x)-JuMP.getvalue(y)) < 1e-6
          return []
      end
  elseif status != :Optimal
      return []
  end


  return [  round.(Int64,JuMP.getvalue(x)),  round.(Int64,JuMP.getvalue(y))]

end

function momentmatchingupdate(x,y,μ,Σ,updatefunction=qgk_update)

    # Σ =  Σ½ *  Σ½'
     Σ½ = ctranspose(chol(Σ))
    v = x - y
    n = length(v)
    # W is orthogonal matrix, W[:,1] = (1 / r) * Σ½'*v, σ² = v'*Σ*v = r²
    # Σ½ * W[:,1] = (1 / r) * Σ * v
    # v' * Σ½ * W = [(( 1 / r) * v' * Σ * v)  0 ⋯ 0 ] = [ σ * sign(r)  0 ⋯ 0 ]
    W,r = qr(Σ½'*v[:,:],;thin=false)
    # W is orthogonal matrix, W[:,1] = (1 / r) * Σ½'*v, σ² = v'*Σ*v = r²
    # r₊ = abs(r) ≥ 0, σ = r₊, W₋ = sign(r) * W
    # (Σ½ * W₋)[:,1] = Σ½ * W₋[:,1] = (1 / σ) * Σ * v
    # v' * Σ½ * W₋ = [((1 / σ) * v' * Σ * v)  0 ⋯ 0 ] = [ σ  0 ⋯ 0 ]
    W₋ = sign(r[]) * W
    r₊ = abs(r[])

    μᵥ = v⋅μ
    σ  = r₊
    μz, σ²z = updatefunction(μᵥ,σ)
    temp =  Σ½ * W

    return μ + (1 / σ) * Σ * v * μz, temp * [σ²z zeros(1,n-1); zeros(n-1,1) eye(n-1)] *temp'
end



function qgk_update(μᵥ,σ)


    C  = QuadGK.quadgk(x->(1+exp(-μᵥ-σ*x))^(-1)*exp(-x^2/2)/sqrt(2*pi),-Inf,Inf)[1]
    μz = QuadGK.quadgk(x->x*(1+exp(-μᵥ-σ*x))^(-1)*exp(-x^2/2)/sqrt(2*pi),-Inf,Inf)[1]/C
    σ²z = QuadGK.quadgk(x->x*x*(1+exp(-μᵥ-σ*x))^(-1)*exp(-x^2/2)/sqrt(2*pi),-Inf,Inf)[1]/C - μz^2

    μz, σ²z
end


function qgk_deff(μᵥ,σ,r=2)
    C  = QuadGK.quadgk(x->(1+exp(-μᵥ-σ*x))^(-1)*exp(-x^2/2)/sqrt(2*pi),-Inf,Inf)[1]
    if C < 1e-6
        C = 0
        σ²z1 = 1
    else
        μz1 = QuadGK.quadgk(x->x*(1+exp(-μᵥ-σ*x))^(-1)*exp(-x^2/2)/sqrt(2*pi),-Inf,Inf)[1]/C
        σ²z1 = QuadGK.quadgk(x->x*x*(1+exp(-μᵥ-σ*x))^(-1)*exp(-x^2/2)/sqrt(2*pi),-Inf,Inf)[1]/C - μz1^2
    end
    if 1-C < 1e-6
        C = 1
        σ²z2 = 1
    else
        μz2 = QuadGK.quadgk(x->x*(1-(1+exp(-μᵥ-σ*x))^(-1))*exp(-x^2/2)/sqrt(2*pi),-Inf,Inf)[1]/(1-C)
        σ²z2 = QuadGK.quadgk(x->x*x*(1-(1+exp(-μᵥ-σ*x))^(-1))*exp(-x^2/2)/sqrt(2*pi),-Inf,Inf)[1]/(1-C) - μz2^2
    end

    C*(σ²z1)^(1/r)+(1-C)*(σ²z2)^(1/r)
end
#
# function qgk_deff(μᵥ,σ,r=2)
#     C  = QuadGK.quadgk(x->(1+exp(-μᵥ-σ*x))^(-1)*exp(-x^2/2)/sqrt(2*pi),-Inf,Inf)[1]
#     if C >= 1e-4 && C <= 1-1e-4
#         μz1 = QuadGK.quadgk(x->x*(1+exp(-μᵥ-σ*x))^(-1)*exp(-x^2/2)/sqrt(2*pi),-Inf,Inf)[1]/C
#         σ²z1 = QuadGK.quadgk(x->x*x*(1+exp(-μᵥ-σ*x))^(-1)*exp(-x^2/2)/sqrt(2*pi),-Inf,Inf)[1]/C - μz1^2
#         μz2 = QuadGK.quadgk(x->x*(1-(1+exp(-μᵥ-σ*x))^(-1))*exp(-x^2/2)/sqrt(2*pi),-Inf,Inf)[1]/(1-C)
#         σ²z2 = QuadGK.quadgk(x->x*x*(1-(1+exp(-μᵥ-σ*x))^(-1))*exp(-x^2/2)/sqrt(2*pi),-Inf,Inf)[1]/(1-C) - μz2^2
#         return C*(σ²z1)^(1/r)+(1-C)*(σ²z2)^(1/r)
#     else
#         return 1
#     end
# end

function fisher_deff(μᵥ,σ,r=2)
    (σ^2*(1+exp(-μᵥ))^(-1)*(1+exp(μᵥ))^(-1)+1)^(-1/r)
end

function fisher_qgk_deff(μᵥ,σ,r=2)
    QuadGK.quadgk(x->(σ^2*(1+exp(-(μᵥ+σ*x)))^(-1)*(1+exp(μᵥ+σ*x))^(-1)+1)^(-1/r)*exp(-x^2/2)/sqrt(2*pi),-Inf,Inf)[1]
end


end
