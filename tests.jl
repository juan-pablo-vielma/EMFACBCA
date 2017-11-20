include("geometric.jl")

using .geometricModule
using Distributions
using HDF5

# Estimates the probability of answering correctly given that the partworth vector is mu
#
# Parameters:
#
# - mu = partworth vector
#
# Returns:
#
# - alpha = the probability estimate

function getAlpha(mu)
	dimension = length(mu)
	alpha = 0
	for i in 1:100000
		prof1 = zeros(dimension)
		prof2 = zeros(dimension)
		while norm(prof1-prof2) < 1e-6
			prof1 = rand(0:1,dimension)
			prof2 = rand(0:1,dimension)
		end
		alpha += max(exp(dot(mu,prof1)),exp(dot(mu,prof2)))/(exp(dot(mu,prof1))+exp(dot(mu,prof2)))
	end
	return alpha/100000
end



# Generates precomputed data
#
# Parameters:
#
# - mu = mean of prior
# - sigma = covariance  of prior
# - confidence = level for confidence ellipsoid/polyhedron
#
# Returns
#
# - precomputed data
function precompute(mu,sigma,confidence)
	precomp = Dict{AbstractString,Any}()

	distro = Chisq(length(mu))
	precomp["rhs"] = quantile(distro, confidence)
	precomp["alpha"] = getAlpha(mu)

	precomp
end

function savedata(filename,results,methods)

	file = open(string(filename,"_time.csv"),"w")
	for m in methods
		for key in keys(results[m])
			println(file,m,",",key,",",mean(results[m][key]),",",maximum(results[m][key]))
		end
	end
	close(file)
end

function test(logfile,filename,iterations,mu,truemu,sigma,confidence,numquestions,methods)


	dimension = length(mu)

	results = Dict()
	for m in methods
		results[m]=Dict(
					  "updatetime"  => Float64[],
					  "questiontime"  => Float64[]
					  )
	end
	csigma = ctranspose(chol(sigma))
	precomp = precompute(mu,sigma,confidence)
	filename = string(filename,norm(mu-truemu)<0.0001 ? "":"wrongmu")
	serializefile = open(string(filename,"serial.dat"),"a")
	betafilename = string(filename,"beta.h5")

	betas=[]
	if !isfile(betafilename)
		h5open(betafilename, "w") do file
    	write(file, "betas", hcat([truemu + csigma*randn(dimension)  for i in 1:iterations]...))
		end
	end
	betas=h5read(betafilename,"betas")
	for i in 1:iterations
		println("Iteration: ",i,"/",iterations)
		beta = betas[:,i]
		runseed = rand(1:100000000)
		allstats = Dict{AbstractString,Any}()
		allstats["beta"]=beta
		allstats["alpha"]=precomp["alpha"]
		allstats["mu"]=mu
		allstats["truemu"]=truemu
		allstats["sigma"]=sigma
		allstats["results"]=Dict{AbstractString,Any}[]
		for m in methods
			print(m)
			parameters = split(m,"_")
			stats = []
			srand(runseed)
			for tries in 1:100
				stats = runsimulation(beta,mu,sigma,confidence,numquestions,precomp,parameters)
				if stats["teststatus"] != "Normal"
					println(logfile,"Error: Repeating iteration for ",m)
				else
					break
				end
			end
			if stats["teststatus"] != "Normal"
				error("Too many iteration drops")
			end
			stats["method"]=m
			push!(allstats["results"],stats)
		end
		serialize(serializefile,allstats)
		for m in methods
			push!(results[m]["updatetime"],allstats["results"][find(x -> x["method"] == m,allstats["results"])[1]]["updatetime"]...)
			push!(results[m]["questiontime"],allstats["results"][find(x -> x["method"] == m,allstats["results"])[1]]["questiontime"]...)
		end
	end
	close(serializefile)
	savedata(filename,results,methods)
end

function runall(iterations)
	dimension = 12
	confidence = 0.9
	methods = [	"estimateEllipsoid_updateBayesApproximation_initializeEllipsoid_questionDEffPWL_effDim",
  						"estimateEllipsoid_updateBayesApproximation_initializeEllipsoid_questionDEffPWLFisher_effDim",
							"estimateEllipsoid_updateBayesApproximation_initializeEllipsoid_questionDEffPWLFisherQGK_effDim",
 							"estimateEllipsoid_updateBayesApproximation_initializeEllipsoid_questionMaxMin",
  						"estimateAnalyticCenter_updatePolyhedral_initializeEllipsoid_questionMaxMin",
							"estimateProbPoly_updateNull_initializeEllipsoid_questionMaxMin",
							"estimateAnalyticCenterRobust_updateNull_initializeEllipsoid_questionNormalizedCenterMIP"
							]
	numquestions=16
	logfile = open("logfile.txt","w")
	for c in [0.5,1.5]
		println(logfile,"c = ",c)
		for varscale in [0.5,2.0]
			println(logfile,"varscale = ",varscale)
			basename = string("results/",dimension,"all",c,"c",varscale,"s")
			mu = c*ones(dimension)
			sigma = c*varscale*eye(dimension)
			println(basename)
			println("\n true :\n")
			println(logfile,"\n true :\n")
			test(logfile,basename,iterations,mu,mu,sigma,confidence,numquestions,methods)

			tempgaussian = randn(dimension)
			scale1=c*varscale*sqrt(2)*(gamma((dimension+1)/2)/gamma((dimension)/2))
			distro = Chisq(length(mu))
			truemu=mu+scale1*(tempgaussian/norm(tempgaussian))
 			println("\n false :\n")
 			println(logfile,"\n false :\n")
 			test(logfile,basename,iterations,mu,truemu,sigma,confidence,numquestions,methods)

		end
	end
	close(logfile)

end
