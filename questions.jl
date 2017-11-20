

module questionModule

include("evaluation.jl")
include("ellipsoidal.jl")
export questionMaxMin, questionDEffPWL, questionDEffPWLFisher, questionDEffPWLFisherInv

using JuMP
using CPLEX
using Cbc
using GLPKMathProgInterface
using GLPK
using Distributions
using .evaluationModule
import .ellipsoidal

# Selects a random question
#
#
# Parameters:
#
# - Q, c, A, b = description of the current uncertainty set (see runsimulation in geometric.jl)
# - center, cov = mean and covariance of the current normal (ellipsoidal) approximation of uncertainty set
# - V = "longest" axis of ellipsoid
# - precomp = precompuded data
#
# Returns
#
# - array of profile vectors for the question or an empty array if the procedure fails


function questionRand(center,cov,V,Q,c,A,b,precomp,parameters)

	q = rand(-1:1,length(center))

	[[ c>=0 ? 1 : 0 for c in q],[ c<=0 ? 1 : 0 for c in q]]
end

## Heuristic question selection

# Selects the next question with the Toubia heuristic
# (x_nx - c)' Q^-1 (x_nx -c), Ax <= b
# assumes either lenth(A) = 0 or length(Q) = 0
# in the second case cov = Q[1] and center = center[1]
#
# Parameters:
#
# - Q, c, A, b = description of the current uncertainty set (see runsimulation in geometric.jl)
# - V = "longest" axis of ellipsoid
# - center, cov = mean and covariance of the current normal (ellipsoidal) approximation of uncertainty set
# - precomp = precompuded data
#
# Returns
#
# - array of profile vectors for the question or an empty array if the procedure fails
function questionMaxMin(center,cov,V,Q,c,A,b,precomp,parameters)

	# get longest axis ends
	u = (Array{Float64,1})[]
	t1 = Inf
	t2 = Inf
	for j in 1:length(A)
		t1 = min(t1,abs((dot(A[j],center)-b[j])/dot(A[j],V)))
		t2 = min(t2,abs((b[j]-dot(A[j],center))/dot(A[j],V)))
	end
	for j in 1:length(Q)
		invQ = inv(cholfact(Q[j]))
		t = sqrt( 1 / dot(V,invQ*V) )
		t1 = min(t1,t)
		t2 = min(t2,t)
	end
	push!(u, center - t1*V)
	push!(u, center + t2*V)

	# get profiles
	rhsvector = rand(Uniform(0,50),30)
	xstar = (Array{Float64,1})[]
	dimension = length(center)
	for it in 1:length(rhsvector)
		for p in 1:2
			model = Model(solver=CplexSolver(CPX_PARAM_MIPDISPLAY=0,CPX_PARAM_SCRIND=0))
			@variable(model, 0 <= x[1:dimension] <= 1, Int)
			@constraint(model, sum( center[i]*x[i] for i=1:dimension ) <= rhsvector[it])

			@objective(model, Max, sum( u[p][i]*x[i] for i=1:dimension ))
			status = solve(model)
			push!(xstar,[round(getvalue(x[i])) for i in 1:dimension])

		end
		if norm(xstar[1]-xstar[2]) > 1e-6
			break
		else
			xstar = (Array{Float64,1})[]
		end
	end

	xstar

end

function questionDEffPWL(center,cov,V,Q,c,A,b,precomp,parameters)
	effPower = 2
	effindex =  findfirst( s -> ismatch(r"effDim",s), parameters)
	if  effindex > 0
		effPower = length(center)
	end
	questionDEffPWLABS(center,cov,V,Q,c,A,b,precomp,parameters,(m,s) -> ellipsoidal.qgk_deff(m,s,effPower))
end

function questionDEffPWLFisher(center,cov,V,Q,c,A,b,precomp,parameters)
	effPower = 2
	effindex =  findfirst( s -> ismatch(r"effDim",s), parameters)
	if  effindex > 0
		effPower = length(center)
	end
	questionDEffPWLABS(center,cov,V,Q,c,A,b,precomp,parameters,(m,s) -> ellipsoidal.fisher_deff(m,s,effPower))
end

function questionDEffPWLFisherQGK(center,cov,V,Q,c,A,b,precomp,parameters)
	effPower = 2
	effindex =  findfirst( s -> ismatch(r"effDim",s), parameters)
	if  effindex > 0
		effPower = length(center)
	end
	questionDEffPWLABS(center,cov,V,Q,c,A,b,precomp,parameters,(m,s) -> ellipsoidal.fisher_qgk_deff(m,s,effPower))
end

function questionDEffPWLABS(center,cov,V,Q,c,A,b,precomp,parameters,efffuction)

	sigma = cov/precomp["rhs"]
	n = size(sigma,1)

	# Set solver and time limit
	timeLimit = 1
	tilimidx =  findfirst( s -> ismatch(r"questionTimeLimit",s), parameters)
	if  tilimidx > 0
		timeLimit = float((parameters[tilimidx])[18:end])
	end
	solverID = "SolverCPLEX"
	selectedSolveridx =  findfirst( s -> ismatch(r"Solver",s), parameters)
	if selectedSolveridx > 0
		solverID = parameters[selectedSolveridx]
	end
	if  solverID == "SolverCPLEX"
		selectedSolver = CplexSolver(CPX_PARAM_TILIM=timeLimit,CPX_PARAM_MIPDISPLAY=0,CPX_PARAM_SCRIND=0)
	elseif solverID == "SolverGLPK"
		selectedSolver = GLPKSolverMIP(tm_lim=1000*timeLimit,msg_lev=GLPK.MSG_OFF)
	elseif solverID == "SolverCbc"
		selectedSolver = CbcSolver(seconds=timeLimit,logLevel=0)
	end
	mipk = 3
	mipkidx = findfirst( s -> ismatch(r"MIPK",s), parameters)
	if mipkidx > 0
			mipk = parse((parameters[mipkidx])[5:end])
	end

	ellipsoidal.getquestion(center,sigma,selectedSolver,mipk,efffuction)

end

## Normalized choice balance question selection

# Selects the next question as the one that minimizes the normalized question mean
#
# Parameters:
#
# - Q, c, A, b = description of the current uncertainty set (see runsimulation in geometric.jl)
# - center, cov = mean and covariance of the current normal (ellipsoidal) approximation of uncertainty set
# - V = "longest" axis of ellipsoid
# - precomp = precompuded data
#
# Returns
#
# - array of profile vectors for the question or an empty array if the procedure fails
function questionNormalizedCenterMIP(center,cov,V,Q,c,A,b,precomp,parameters)

	n = length(center)

	timeLimit = 3600
	model = Model(solver=CplexSolver(CPX_PARAM_TILIM=timeLimit,CPX_PARAM_MIPDISPLAY=0,CPX_PARAM_SCRIND=0))

	# define variables for linearization
	@variable(model, 0 <= x[1:2,1:n] <= 1, Int)
	@variable(model, 0 <= w[k=1:2,i=1:n,j=i+1:n] <= 1, Int)
	@variable(model, 0 <= w12[k1=1:2,k2=1:2,i=1:n,j=1:n] <= 1, Int)
	for k1 in 1:2
		# w = x^2
		for i in 1:n
			for j in (i+1):n
				@constraint(model, w[k1,i,j] >= x[k1,i] + x[k1,j] - 1)
				@constraint(model, w[k1,i,j] <= x[k1,i] )
				@constraint(model, w[k1,i,j] <= x[k1,j] )
			end
		end
		for k2 in 1:(k1-1)
			for i in 1:n
				for j in 1:n
					w12[k1,k2,i,j] = w12[k2,k1,i,j]
				end
			end
		end
		for i in 1:n
			for j in 1:(i-1)
				w12[k1,k1,i,j] = w[k1,j,i]
			end
			w12[k1,k1,i,i] = x[k1,i]
			for j in (i+1):n
				w12[k1,k1,i,j] = w[k1,i,j]
			end
		end
		for k2 in (k1+1):2
			# w12 = x1*x2
			for i in 1:n
				for j in 1:n
					@constraint(model, w12[k1,k2,i,j] >= x[k1,i] + x[k2,j] - 1)
					@constraint(model, w12[k1,k2,i,j] <= x[k1,i] )
					@constraint(model, w12[k1,k2,i,j] <= x[k2,j] )
				end
			end
		end
	end

	# forces products to be different
	for k1 in 1:2
		for k2 in (k1+1):2
			# (x1-x2).(x1-x2) >= 0
			@constraint(model, sum( x[k1,i] + x[k2,i] - 2*w12[k1,k2,i,i] for i=1:n ) >= 1)
		end
	end


	@variable(model, distance >=0)
	addcorrecteddistancetocenter(model,center,cov,n,distance,x,w,w12)




	@objective(model, Min,  distance )
	status = solve(model)


	[ [round(getvalue(x[k,i])) for i in 1:n] for k in 1:2]
end


# utility for questionNormalizedCenterMIP
function addcorrecteddistancetocenter(model,center,cov,n,distance,x,w,w12)

	# (x1-x2)'.cov.(x1-x2) <= eigmax(cov) ||x1-x2|| <= eigmax(cov)*n
	U = eigmax(cov)*n
	# (x1-x2)'.cov.(x1-x2) >= eigmin(cov) ||x1-x2|| >= eigmin(cov)   ( x1 <> x2 )
	L = eigmin(cov)

	@variable(model, 1/U <= f[k1=1:2,k2=k1+1:2] <= 1/L)
	@variable(model, 0 <= fx1[k1=1:2,k2=k1+1:2,1:n] <= 1/L)
	@variable(model, 0 <= fx2[k1=1:2,k2=k1+1:2,1:n] <= 1/L)
	@variable(model, 0 <= fw1[k1=1:2,k2=k1+1:2,i=1:n,j=i+1:n] <= 1/L)
	@variable(model, 0 <= fw2[k1=1:2,k2=k1+1:2,i=1:n,j=i+1:n] <= 1/L)
	@variable(model, 0 <= fw12[k1=1:2,k2=k1+1:2,i=1:n,j=1:n] <= 1/L)
	for k1 in 1:2
		for k2 in (k1+1):2
			FL = 1/U
			FU = 1/L
			# fx1 = f*x1, fx2 = f*x2
			for i in 1:n
				@constraint(model, fx1[k1,k2,i] <= FU*x[k1,i] )
				@constraint(model, fx1[k1,k2,i] <= f[k1,k2] + FL*x[k1,i] - FL )
				@constraint(model, fx1[k1,k2,i] >= f[k1,k2] + FU*x[k1,i] - FU )
				@constraint(model, fx1[k1,k2,i] >= FL*x[k1,i] )

				@constraint(model, fx2[k1,k2,i] <= FU*x[k2,i] )
				@constraint(model, fx2[k1,k2,i] <= f[k1,k2] + FL*x[k2,i] - FL )
				@constraint(model, fx2[k1,k2,i] >= f[k1,k2] + FU*x[k2,i] - FU )
				@constraint(model, fx2[k1,k2,i] >= FL*x[k2,i] )
			end
			# fw1 = f*w1, fw2 = f*w2
			for i in 1:n
				for j in (i+1):n
					@constraint(model, fw1[k1,k2,i,j] <= FU*w[k1,i,j] )
					@constraint(model, fw1[k1,k2,i,j] <= f[k1,k2] + FL*w[k1,i,j] - FL )
					@constraint(model, fw1[k1,k2,i,j] >= f[k1,k2] + FU*w[k1,i,j] - FU )
					@constraint(model, fw1[k1,k2,i,j] >= FL*w[k1,i,j] )

					@constraint(model, fw2[k1,k2,i,j] <= FU*w[k2,i,j] )
					@constraint(model, fw2[k1,k2,i,j] <= f[k1,k2] + FL*w[k2,i,j] - FL )
					@constraint(model, fw2[k1,k2,i,j] >= f[k1,k2] + FU*w[k2,i,j] - FU )
					@constraint(model, fw2[k1,k2,i,j] >= FL*w[k2,i,j] )
				end
			end
			# fw12 = f*w12
			for i in 1:n
				for j in 1:n
					@constraint(model, fw12[k1,k2,i,j] <= FU*w12[k1,k2,i,j] )
					@constraint(model, fw12[k1,k2,i,j] <= f[k1,k2] + FL*w12[k1,k2,i,j] - FL )
					@constraint(model, fw12[k1,k2,i,j] >= f[k1,k2] + FU*w12[k1,k2,i,j] - FU )
					@constraint(model, fw12[k1,k2,i,j] >= FL*w12[k1,k2,i,j] )
				end
			end

		end
	end
	for k1 in 1:2
		for k2 in (k1+1):2

			@constraint(model, 1 == sum( cov[i,i]*( fx1[k1,k2,i] + fx2[k1,k2,i] - 2*fw12[k1,k2,i,i] ) for i=1:n )
									   + sum( 2*cov[i,j]*( fw1[k1,k2,i,j] + fw2[k1,k2,i,j] - fw12[k1,k2,i,j] - fw12[k1,k2,j,i] ) for i=1:n,j=i+1:n ) )
			@constraint(model, distance >= sum( center[i]^2*( fx1[k1,k2,i] + fx2[k1,k2,i] - 2*fw12[k1,k2,i,i] ) for i=1:n )
													 + sum( 2*center[i]*center[j]*( fw1[k1,k2,i,j] + fw2[k1,k2,i,j] - fw12[k1,k2,i,j] - fw12[k1,k2,j,i] ) for i=1:n,j=i+1:n ) )

		end
	end
end

end
