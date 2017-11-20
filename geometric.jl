

module geometricModule


export runsimulation


include("questions.jl")
include("update.jl")
include("estimation.jl")
include("evaluation.jl")

using Distributions
using .questionModule
using .updateModule
using .estimationModule
using JuMP
using .evaluationModule



# Calculates answer to question according to the MNL model
#
# Parameters:
#
# - beta = real beta
# - x = array of profiles
#
# Returns:
#
# - answer = index of profile with maximum utility with errors
# - errorinanswer = 1 if the profile with the maximum utility with error does not have the maximum utility without error

function answerquestion(beta,x)

	numprofiles = length(x)
	d = Gumbel()
	errors = rand(d,numprofiles)

	realutilities = [dot(beta,x[i])  for i in 1:numprofiles]
	utilitieswitherror = realutilities + errors

	realmax = indmax(realutilities)
	answer  = indmax(utilitieswitherror)

	errorinanswer = 0
	ambiguousanswer = 0

	if realutilities[realmax] > realutilities[answer] + 1e-6
		errorinanswer = 1
	end

	return answer, errorinanswer
end



# Runs the simulation
#
# Parameters:
#
# - beta = real beta
# - mu = mean of prior
# - sigma = covariance  of prior
# - confidence = level for confidence ellipsoid/polyhedron
# - numquestions = number of questions asked
# - precomp = precompuded data
# - parameters = list of parameters
#
# Returs
#
# - stats = various statistics
#
# Data associated to run
#
# - At any stage the current uncertainty set where we believe beta to be contained is:
#
# 	(beta - c[i])'*Q[i]^(-1)*(beta - c[i])<= 1 for all i
# 	A[i]*beta <= b[i] for all i
#
# - center and cov are estimates of the shape of the uncertainty set so that
#
#      (beta - center)'*cov^(-1)*(beta - center) <= r
#
#   is an approximation of the uncertaintly set for an appropriately chosen r.
#   Alternative center and cov can be interpreted as approximaitons of the mean
#   and an appropriate scalling of the covariance matrix of the current posterior
#   distribution of beta

function runsimulation(beta,mu,sigma,confidence,numquestions,precomp,parameters)


	estimateidx = findfirst( s -> ismatch(r"estimate",s), parameters)
	estimate = getfield(estimationModule, parse(parameters[estimateidx]))

	questionidx = findfirst( s -> ismatch(r"question",s), parameters)
	getnextquestion = getfield(questionModule, parse(parameters[questionidx]))

	updateidx = findfirst( s -> ismatch(r"update",s), parameters)
	update = getfield(updateModule, parse(parameters[updateidx]))

	updateidx = findfirst( s -> ismatch(r"initialize",s), parameters)
	initialize = getfield(updateModule, parse(parameters[updateidx]))


	stats = Dict{AbstractString,Any}() #(String => Any)[]
	stats["questiontime"] = Float64[]
	stats["estimatetime"] = Float64[]
	stats["updatetime"] = Float64[]
	stats["x"] = (Array{Array{Int,1},1})[]
	stats["teststatus"] = "Normal"
	stats["numerrors"] = Int[]
	stats["answer"] = Int[]
	stats["Q"]=Array{Array{Float64,2},1}[]
	stats["c"]=Array{Array{Float64,1},1}[]
	stats["A"]=Array{Array{Float64,1},1}[]
	stats["b"]=Array{Float64,1}[]
	stats["center"]=Array{Float64,1}[]
	stats["cov"]=Array{Float64,2}[]

	Q,c,A,b = initialize(mu,sigma,precomp)

	push!(stats["Q"],deepcopy(Q))
	push!(stats["c"],deepcopy(c))
	push!(stats["A"],deepcopy(A))
	push!(stats["b"],deepcopy(b))


	answer = []
	errorinanswer = []
	dimension = length(beta)
	for i in 1:numquestions
		print(".")
		tic();
		center, cov, V, stats["teststatus"] = estimate(Q,c,A,b,stats["x"],stats["answer"],precomp,dimension,parameters)
		if stats["teststatus"] != "Normal"
			return stats
		end
		push!(stats["estimatetime"],toq());
		push!(stats["center"],center)
		push!(stats["cov"],cov)


		tic();
		x = getnextquestion(center,cov,V,Q,c,A,b,precomp,parameters)
		push!(stats["questiontime"],toq());
		if length(x) == 0
			stats["teststatus"] = "ZeroQuestion"
			return stats
		end
		push!(stats["x"],x);

		answer, errorinanswer  = answerquestion(beta,x)


		push!(stats["numerrors"],errorinanswer)
		push!(stats["answer"],answer)

		tic();
		stats["teststatus"] = update(Q,c,A,b,stats["x"],stats["answer"],mu,sigma,precomp,parameters)
		if stats["teststatus"] != "Normal"
			return stats
		end
		push!(stats["updatetime"],toq());
		push!(stats["Q"],deepcopy(Q))
		push!(stats["c"],deepcopy(c))
		push!(stats["A"],deepcopy(A))
		push!(stats["b"],deepcopy(b))
	end
	println("")
	tic();
	center, cov, V, stats["teststatus"] = estimate(Q,c,A,b,stats["x"],stats["answer"],precomp,dimension,parameters)
	if stats["teststatus"] != "Normal"
		return stats
	end
	push!(stats["estimatetime"],toq());
	push!(stats["center"],center)
	push!(stats["cov"],cov)

	return stats

end




end
