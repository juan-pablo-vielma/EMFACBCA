module estimationModule

export estimateEllipsoid, estimateAnalyticCenter

using JuMP, CPLEX, Ipopt, Combinatorics

## Ellipsoidal method

# Assumes the uncertainty set is a single ellipsoid and
# hence the ellipsoidal approximation is the ellipsoid itself.
# If Q[1] is numerically non-symmetric it tries to fix it.
#
# Parameters:
#
# - Q, c, A, b = description of uncertainty set (see runsimulation in geometric.jl)
# - questions, answers = history of questions and answers
# - precomp = precompuded data
# - dimension = the dimension of beta
# - parameters = list of parameters
#
# Returns:
#
# - ellipsoidal approximation (beta - c[1])'*Q[1]^(-1)*(beta - c[1])
# - status, which is always "Normal"

function estimateEllipsoid(Q,c,A,b,questions,answers,precomp,dimension,parameters)
	if !issymmetric(Q[1])
		for i in 1:size(Q[1],1)
			for j in i+1:size(Q[1],1)
				if abs(Q[1][i,j]-Q[1][j,i]) > 1e-5
					return c[1], Q[1], "UnsymmetricMatrix"
				else
					Q[1][i,j]=Q[1][j,i]
				end
			end
		end
	end

	L,V = eig(Q[1])
	order = sortperm(L,rev=true)

	return c[1], Q[1],V[:,order[1]], "Normal"
end


## Polyhedral method

# Computes the analytic center of the uncertainty set
#
# The analytic center is xstar[1:dimension] and Minv[1:dimension,1:dimension] is the matrix such that
# (x - xstar[1:dimension])'*(Minv[1:dimension,1:dimension])^(-1)*(x - xstar[1:dimension]) <= 1 and
# (x - xstar[1:dimension])'*(Minv[1:dimension,1:dimension])^(-1)*(x - xstar[1:dimension]) <= n(n-1)
# are the inner and outer ellipsoidal approximations of the projection of the uncertainty set
# onto the first dimension variables
#
# Parameters:
#
# - Q, c, A, b = description of uncertainty set (see runsimulation in geometric.jl)
# - questions, answers = history of questions and answers
# - precomp = precompuded data
# - dimension = the dimension of beta
# - parameters = list of parameters
#
# Returns:
#
# - ellipsoidal approximation (beta - xstar[1:dimension])'*Minv[1:dimension,1:dimension]^(-1)*(beta - xstar[1:dimension])
# - status, which can be "Normal" for normal termination, "EmptyInterior" if the uncertainty set has empty interior,
#	or "IpoptError" if Ipopt has trouble calculating the analytic center

function estimateAnalyticCenter(Q,c,A,b,questions,answers,precomp,dimension,parameters)

	# First correct questions so we have an interior point

	strictTolerance = 1e-6

	model = Model(solver=CplexSolver(CPX_PARAM_SCRIND=0))
	m = length(A)
	n = 0
	if m > 0
		n = length(A[1])
	else
		n = length(c[1])
	end
	@variable(model, t >= 0)
	@variable(model, x[1:n])
	@constraint(model,slacks[k=1:m],  sum( A[k][i]*x[i] for i=1:n ) + strictTolerance <= b[k] + t)
	qm = length(Q)
	invQ = Array{Float64,2}[]
	for k in 1:qm
		push!(invQ,inv(cholfact(Q[k])))
	end
	@constraint(model,qslacks[k=1:qm], sum( invQ[k][i,j]*(x[i]*x[j]-c[k][i]*x[j]-c[k][j]*x[i]+c[k][i]*c[k][j]) for i=1:dimension, j=1:dimension ) + strictTolerance <= 1 )
	@objective(model, Min, t)
	status = solve(model)
	if status != :Optimal
		return [],[],[], "AnalyticCenterErrorDelta"
	end
	delta = getvalue(t)
	# First get an interior point to initialize IPOPT
	model = Model(solver=CplexSolver(CPX_PARAM_SCRIND=0))
	m = length(A)
	n = 0
	if m > 0
		n = length(A[1])
	else
		n = length(c[1])
	end
	@variable(model, t)
	@variable(model, x[1:n])
	@variable(model, s[1:m] <=  - strictTolerance )
	@constraint(model,[k=1:m],  sum( A[k][i]*x[i] for i=1:n ) -  b[k] - delta <= s[k])
	@constraint(model,[k=1:m],  s[k] <= t)
	qm = length(Q)
	@variable(model, sq[1:qm] <=  - strictTolerance )
	invQ = Array{Float64,2}[]
	for k in 1:qm
		push!(invQ,inv(cholfact(Q[k])))
	end
	@constraint(model,[k=1:qm], sum( invQ[k][i,j]*(x[i]*x[j]-c[k][i]*x[j]-c[k][j]*x[i]+c[k][i]*c[k][j]) for i=1:dimension, j=1:dimension ) - 1 <= sq[k])
	@constraint(model,[k=1:qm], sq[k] <= t)
	@objective(model, Min, t)
	status = solve(model)
	if status != :Optimal
		return [],[],[], "AnalyticCenterErrorEmptyInterior"
	end
	xstart = getvalue(x)
	sstart = getvalue(s)
	sqstart = getvalue(sq)

	# Now get the analytic center
	model = Model(solver=IpoptSolver(print_level=0))
	@variable(model, x[i=1:n], start = xstart[i])
	@variable(model, s[i=1:m] <= - strictTolerance,  start = sstart[i])
	@variable(model, sq[i=1:qm] <= - strictTolerance,  start = sqstart[i])
	@constraint(model,slacks[k=1:m],  sum( A[k][i]*x[i] for i=1:n ) -  b[k] - delta <= s[k])
	@constraint(model,qslacks[k=1:qm], sum( invQ[k][i,j]*(x[i]*x[j]-c[k][i]*x[j]-c[k][j]*x[i]+c[k][i]*c[k][j]) for i=1:dimension, j=1:dimension ) - 1 <= sq[k])

	@NLobjective(model, Min, - sum( log(-s[k]) for k=1:m) - sum( log(-sq[k]) for k=1:qm))
	status = solve(model)
	if status != :Optimal
		return [],[],[], "AnalyticCenterErrorIpoptError"
	end
	xstar = getvalue(x)
	Minv = []
	hessian = []
	try
		hessian = hessianofLogBarrier(invQ,c,A,b,xstar,delta)
		Minv = inv(cholfact(hessian))
	catch exception
		println("EXEPTION")
		println(exception)
		println(minimum(eigvals(hessian)))
	end

	center = xstar[1:dimension]
	cov = Minv[1:dimension,1:dimension]
	L,V = eig(cov)
 	order = sortperm(L,rev=true)

	center, cov, V[:,order[1]], "Normal"
end


# Returns hessian of sum_k - Log(1-(x-c[k])'*invQ[k]*(x-c[k])) + sum_k - Log(b[k]-A[k]*x)
function hessianofLogBarrier(invQ,c,A,b,xstar,delta)

	m = length(A)
	qm = length(invQ)
	n = 0
	if m > 0
		n = length(A[1])
	else
		n = length(c[1])
	end
	M = zeros(n,n)
	for k in 1:m
		for i in 1:n
			for j in 1:n
				M[i,j] += (A[k][i]*A[k][j])/( b[k] + delta - dot(A[k],xstar) )^2
			end
		end
	end
	for k in 1:qm
		dimension = length(c[k])
		qx = invQ[k]*(xstar[1:dimension]-c[k])
		denominator = (1 - (xstar[1:dimension]-c[k])'*invQ[k]*(xstar[1:dimension]-c[k]))[1]
		for i in 1:dimension
			for j in 1:dimension
				M[i,j] += 2 * invQ[k][i,j]/denominator  + 4 * dot(qx[i],qx[j])/(denominator^2)
			end
		end
	end

	M
end

## Probabilistic Polyhedral  method

# Auxiliary function for  estimateProbPoly

function getAb(questions,answers,reverse)

	tempA = Array{Float64,1}[]
	tempb = Float64[]

	for q in 1:length(questions)
		if reverse[q] > 0
			push!(tempA, questions[q][answers[q]] - questions[q][3-answers[q]])
		else
			push!(tempA, questions[q][3-answers[q]] - questions[q][answers[q]])
		end
		push!(tempb,0)
	end

	tempA, tempb
end


#
# Parameters:
#
# - Q, c, A, b = description of uncertainty set (see runsimulation in geometric.jl)
# - questions, answers = history of questions and answers
# - precomp = precompuded data
# - dimension = the dimension of beta
# - parameters = list of parameters
#
# Returns:
#

function estimateProbPoly(Q,c,A,b,questions,answers,precomp,dimension,parameters)
	count = 0
	totalweight = 0
	alpha = precomp["alpha"]
	numquestions = length(questions)
	reverse = zeros(Int,numquestions)
	tempA,tempb = getAb(questions,answers,reverse)
	center, cov, nullV, status = estimateAnalyticCenter(Q,c,tempA,tempb,questions,answers,precomp,dimension,[])
	if status != "Normal"
		return [],[],[],status
	end

	Vs = Array{Float64,1}[]
	pis = Float64[]
	L,V = eig(cov)
	order = sortperm(L,rev=true)
	push!(Vs,V[:,order[1]])
	push!(pis,alpha^dimension)
	center *= alpha^dimension
	cov *= alpha^dimension
	totalweight += alpha^dimension
	count += 1
	for z in 1:numquestions-1
		stop = false
		for op in combinations(1:numquestions,z)
			reverse = zeros(numquestions)
			for l in op
				reverse[l] = 1
			end
			tempA,tempb = getAb(questions,answers,reverse)
			tempcenter, tempcov, nullV, status = estimateAnalyticCenter(Q,c,tempA,tempb,questions,answers,precomp,dimension,[])
			if status != "Normal"
				return [],[],[],status
			end
			L,V = eig(tempcov)
			order = sortperm(L,rev=true)
			push!(Vs,V[:,order[1]])
			push!(pis,alpha^(dimension-z)*(1-alpha)^z)
			center += alpha^(dimension-z)*(1-alpha)^z*tempcenter
			cov += alpha^(dimension-z)*(1-alpha)^z*tempcov
			totalweight += alpha^(dimension-z)*(1-alpha)^z
			count += 1
			if count > 32
				stop = true
				break
			end
		end
		if stop
			break
		end
	end
	center /= totalweight
	cov /= totalweight
	Pi = diagm(pis)/totalweight
	VV = zeros(dimension,count)
	for l in 1:count
		VV[:,l] = Vs[l]
	end

	AA = VV*Pi*VV'
	if !issymmetric(AA)
		for i in 1:size(AA,1)
			for j in i+1:size(AA,1)
				if abs(AA[i,j]-AA[j,i]) > 1e-5
					return [],[],[], "UnsymmetricMatrix"
				else
					AA[i,j]=AA[j,i]
				end
			end
		end
	end

	L,V = eig(AA)
	order = sortperm(L,rev=true)

	center, cov, V[:,order[1]], status

end

## Robust

#
# Parameters:
#
# - Q, c, A, b = description of uncertainty set (see runsimulation in geometric.jl)
# - questions, answers = history of questions and answers
# - precomp = precompuded data
# - dimension = the dimension of beta
# - parameters = list of parameters
#
# Returns:
#


function estimateAnalyticCenterRobust(Q,c,A,b,questions,answers,precomp,dimension,parameters)

	if length(questions) == 0
		return c[1], eye(dimension), [], "Normal"
	end

	K = ceil(Int,(1-precomp["alpha"])*length(questions))
	A = Array{Float64,1}[]

	for q in 1:length(questions)
		push!(A, questions[q][3-answers[q]] - questions[q][answers[q]])
	end

	model = Model(solver=CplexSolver(CPX_PARAM_SCRIND=0))
	m = length(A)
	n = dimension
	@variable(model, x[1:n])
	qm = length(Q)
	invQ = Array{Float64,2}[]
	for k in 1:qm
		push!(invQ,inv(cholfact(Q[k])))
	end
	@constraint(model,qslacks[k=1:qm], sum( invQ[k][i,j]*(x[i]*x[j]-c[k][i]*x[j]-c[k][j]*x[i]+c[k][i]*c[k][j]) for i=1:dimension, j=1:dimension ) <= 1 )
	lbounds = Float64[]
	ubounds = Float64[]
	for k in 1:m
		@objective(model, Max, sum( A[k][i]*x[i] for i=1:n ))
		status = solve(model)
		if status != :Optimal
			return [], [], [], "EmptyInterior"
		end
		push!(ubounds,getobjectivevalue(model))
		@objective(model, Min, sum( A[k][i]*x[i] for i=1:n ))
		status = solve(model)
		if status != :Optimal
			return [], [], [], "EmptyInterior"
		end
		push!(lbounds,getobjectivevalue(model))
	end

	model = Model(solver=CplexSolver(CPX_PARAM_SCRIND=0))
	@variable(model, x[i=1:n])
	@variable(model, 0<=y[i=1:m]<=1,Int)
	@variable(model, s1[i=1:m] <= - 1e-6)
	@variable(model, ts1[i=1:m])
	@variable(model, s2[i=1:m] <= - 1e-6)
	@variable(model, ts2[i=1:m] )
	@variable(model, sq[i=1:qm] <= - 1e-6)
	@variable(model, tsq[i=1:qm])
	@constraint(model,[k=1:m],  sum( A[k][i]*x[i] for i=1:n ) - y[k]*ubounds[k]  <= s1[k])
	for i in 1:3
		rr = -100/(10^i)
		fp = -1/rr
		@constraint(model,[k=1:m],  -log(-rr)+ fp*(s1[k]-rr) <= ts1[k])
	end
	@constraint(model,[k=1:m],  (1-y[k])*lbounds[k] - sum( A[k][i]*x[i] for i=1:n )  <= s2[k])
	for i in 1:3
		rr = -100/(10^i)
		fp = -1/rr
		@constraint(model,[k=1:m],  -log(-rr)+ fp*(s2[k]-rr) <= ts2[k])
	end

	@constraint(model,[k=1:qm], sum( invQ[k][i,j]*(x[i]*x[j]-c[k][i]*x[j]-c[k][j]*x[i]+c[k][i]*c[k][j]) for i=1:dimension, j=1:dimension ) - 1 <= sq[k])
	for i in 1:3
		rr = -100/(10^i)
		fp = -1/rr
		@constraint(model,[k=1:qm],  -log(-rr)+ fp*(sq[k]-rr) <= tsq[k])
	end
	@constraint(model,sum(y[i] for i=1:m) <= K)
	@objective(model, Min, sum( ts1[k] + ts2[k] for k=1:m) + sum( tsq[k] for k=1:qm))
	try
		status = solve(model)
	catch
		return [], [], [], "CPLEXError"
	end
	if status != :Optimal
		return [], [], [], "EmptyInterior"
	end

	xstart = getvalue(x)
	s1start = getvalue(s1)
	s2start = getvalue(s1)
	ystart = getvalue(y)
	sqstart = getvalue(sq)


	# Now get the analytic center
	model = Model(solver=IpoptSolver(print_level=0))

	@variable(model, x[i=1:n], start = xstart[i])
	@variable(model, s1[i=1:m] <= - 1e-6, start = s1start[i])
	@variable(model, s2[i=1:m] <= - 1e-6, start = s2start[i])
	@variable(model, sq[i=1:qm] <= - 1e-6, start = sqstart[i])
	@constraint(model,[k=1:m],  sum( A[k][i]*x[i] for i=1:n ) - ystart[k]*ubounds[k]  <= s1[k])
	@constraint(model,[k=1:m],  (1-ystart[k])*lbounds[k] - sum( A[k][i]*x[i] for i=1:n )  <= s2[k])

	@constraint(model,[k=1:qm], sum( invQ[k][i,j]*(x[i]*x[j]-c[k][i]*x[j]-c[k][j]*x[i]+c[k][i]*c[k][j]) for i=1:dimension, j=1:dimension ) - 1.01 <= sq[k])
	@NLobjective(model, Min, - sum( log(-s1[k]) +log(-s2[k]) for k=1:m) - sum( log(-sq[k]) for k=1:qm))


	status = solve(model)
	if status != :Optimal
		return [], [], [], "IpoptError"
	end

	xstar = getvalue(x)
	xstar[1:dimension], eye(dimension), [],  "Normal"
end




end
