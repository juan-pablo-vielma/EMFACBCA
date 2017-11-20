

module updateModule


include("ellipsoidal.jl")
export updateTraditionalEllipsoid, updateCorrectedEllipsoid, initializeEllipsoid, initializeExtendedCrossBox, updatePolyhedral, updateBayesApproximation

using Distributions
using JuMP
import .ellipsoidal


# Does not update anything. Used by probabilistic polyhedral and robust
#
# Parameters:
#
# - Q, c, A, b = description of uncertainty set (see runsimulation in geometric.jl)
# - X = array of all questions asked
# - answers = array of all answers
# - mu = mean of original prior distribution of beta
# - sigma = covariance matrix of original prior distribution of beta
# - precomp = precompuded data
# - parameters = list of parameters
#
# Returns
#
# - status of update always equal to "Normal"

function updateNull(Q,c,A,b,X,answers,mu,sigma,precomp,parameters)
	return "Normal"
end
## Polyhedral

# Updates the uncertainty set by adding the inequality
#
# x[i] <= x[answer] for all profiles i
#
# Parameters:
#
# - Q, c, A, b = description of uncertainty set (see runsimulation in geometric.jl)
# - X = array of all questions asked
# - answers = array of all answers
# - mu = mean of original prior distribution of beta
# - sigma = covariance matrix of original prior distribution of beta
# - precomp = precompuded data
# - parameters = list of parameters
#
# Returns
#
# - status of update always equal to "Normal"

function updatePolyhedral(Q,c,A,b,X,answers,mu,sigma,precomp,parameters)
	x = X[end]
	answer = answers[end]

	n = 0
	if length(A) > 0
		n = length(A[1])
	else
		n = length(c[1])
	end
	dimension = length(x[1])

	for i in setdiff(1:length(x),answer)
		push!(A,[ x[i] - x[answer]; zeros(n-dimension)])
		push!(b,0)
	end

	return "Normal"
end

## Ellispoidal

# Returns the "confidece" confidence level credibility ellipsoid
# for a gaussian with mean mu and covariate matrix sigma
#
# Parameters:
#
#   - mu = mean of original prior distribution of beta
#   - sigma = covariance matrix of original prior distribution of beta
#	- confidence = confidence level for the ellipsoid
#
# Returns
#
#	- ellipsoid (beta - c)'*Q^(-1)*(beta - c)

function initializeEllipsoid(mu,sigma,precomp)


	Q = Array{Float64,2}[]
	push!(Q,sigma*precomp["rhs"])
	c = Array{Float64,1}[]
	push!(c,mu)

	return Q,c,Array{Float64,1}[],Float64[]
end

# Assumes the uncertainty set is just one ellipsoid. If there are only two profiles it updates
# with the minimum volume ellipsoid that contains the original ellipsoid intersected with
#
# x["notanswer"] <= x[answer]
#
# If there are more than two profiles it iteratively repeats the update for each constraint
#
# x[i] <= x[answer]
#
# Parameters:
#
# - Q, c, A, b = description of uncertainty set (see runsimulation in geometric.jl)
# - X = array of all questions asked
# - answers = array of all answers
# - mu = mean of original prior distribution of beta
# - sigma = covariance matrix of original prior distribution of beta
# - precomp = precompuded data
# - parameters = list of parameters
#
# Returns
#
# - status of update. Is "Normal" for normal termination and "NoUpdate" if the minimum
#   volume ellipsoid is the original ellipsoid.

function updateTraditionalEllipsoid(Q,c,A,b,X,answers,mu,sigma,precomp,parameters)
	x = X[end]
	answer = answers[end]

	n = length(c[1])
	update = false
	for i in setdiff(1:length(x),answer)
		question = x[i] - x[answer]
		scale = sqrt(question'*Q[1]*question)[1]
		alpha = dot(question,c[1])/scale
		b = Q[1]*question/scale

		if alpha > -1/n
			update = true
			s1 = ((n^2)/(n^2-1))*(1-alpha^2)
			s2 = 2*(1+n*alpha)/((n+1)*(1+alpha))

			c[1] = c[1] - ((1+n*alpha)/(n+1))*b
			Q[1] = s1*( Q[1] - s2*b*b' )
		end

	end
	if update
		return "Normal"
	else
		return "NoUpdate"
	end

end

# Assumes the uncertainty set is just one ellipsoid. If there are only two profiles it updates
# with a correction of  minimum volume ellipsoid that contains the original ellipsoid intersected with
#
# x["notanswer"] <= x[answer]
#
# If there are more than two profiles it iteratively repeats the update for each constraint
#
# x[i] <= x[answer]
#
# Parameters:
#
# - Q, c, A, b = description of uncertainty set (see runsimulation in geometric.jl)
# - X = array of all questions asked
# - answers = array of all answers
# - mu = mean of original prior distribution of beta
# - sigma = covariance matrix of original prior distribution of beta
# - precomp = precompuded data
# - parameters = list of parameters
#
# Returns
#
# - status of update. Is "Normal" for normal termination and "NoUpdate" if the minimum
#   volume ellipsoid is the original ellipsoid.

function updateCorrectedEllipsoid(Q,c,A,b,X,answers,mu,sigma,precomp,parameters)
	x = X[end]
	answer = answers[end]

	n = length(c[1])
	update = false
	for i in setdiff(1:length(x),answer)
		question = x[i] - x[answer]
		scale = sqrt(question'*Q[1]*question)[1]
		alpha = dot(question,c[1])/scale
		b = Q[1]*question/scale

		if alpha > -1/n
			update = true
			s1 = ((n^2)/(n^2-1))*(1-alpha^2)
			s2 = 2*(1+n*alpha)/((n+1)*(1+alpha))

			c[1] = c[1] - ((1+n*alpha)/(n+1))*b
			Q[1] = Q[1] - (1+s1*s2-s1)*b*b'
		end

	end
	if update
		return "Normal"
	else
		return "NoUpdate"
	end

end

# Assumes the uncertainty set is just one ellipsoid, which is the credibility ellipsoid of a gaussian
# prior distribution. It updates the ellipsoid with the credibility ellipsoid of the posterior distribution
# obtained through an MCMC procedure that exploits the geometry of the update so that the dimension of the
# sample is only one less than the number of profiles
#
# Parameters:
#
# - Q, c, A, b = description of uncertainty set (see runsimulation in geometric.jl)
# - X = array of all questions asked
# - answers = array of all answers
# - mu = mean of original prior distribution of beta
# - sigma = covariance matrix of original prior distribution of beta
# - precomp = precompuded data
# - parameters = list of parameters
#
# Returns
#
# - status of update always equal to "Normal"

function updateBayesApproximation(Q,c,A,b,X,answers,mu,sigma,precomp,parameters)

	if answers[end] == 1
		x = vec(X[end][1])
		y = vec(X[end][2])
	else
		x = vec(X[end][2])
		y = vec(X[end][1])
	end

	cov = Q[1]/precomp["rhs"]
	center = c[1]


	newcenter, newcov = ellipsoidal.momentmatchingupdate(x,y,center,cov)
	c[1] = vec(newcenter)
	Q[1] = newcov*precomp["rhs"]

	return "Normal"

end

# function fisherupdate(x,y,μ,Σ)
#
# 	μ, inv(inv(Σ)+(x-y)*(x-y)'*(2+2*cosh(dot(x-y,μ)))^(-1))
#
# end




end
