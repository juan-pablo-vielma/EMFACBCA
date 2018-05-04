include("evaluation.jl")
include("HB_estimation.jl")

using .evaluationModule
using .HB_estimationModule
using HypothesisTests

function computefirstpassstats(filename)
	file = open(filename,"r")
	println("Computing First Pass Statistics...")
	data = Dict()
	data["truebetas"] = Array{Float64}[]
	data["allmethods"] = Set(AbstractString[])
	globalcount=1
	while !eof(file)
		println("Customer ",globalcount,": ")
		globalcount+=1
		allstats = deserialize(file)
		push!(data["truebetas"],allstats["beta"])
		nfeatures=data["nfeatures"]=length(allstats["beta"])
		mu = data["mu"] = allstats["mu"]
		truemu = data["truemu"] = allstats["truemu"]
		sigma = data["sigma"] = allstats["sigma"]

		for stats in allstats["results"]
			push!(data["allmethods"],stats["method"])
			nquestions=data["nquestions"]=length(stats["x"])
			if !haskey(data, stats["method"])
				data[stats["method"]]=Dict()
				data[stats["method"]]["x"]=Array{Array{Int64}}[]
				data[stats["method"]]["y"]=Array{Array{Int64}}[]
				data[stats["method"]]["answer"]=Array{Array{Int64}}[]
				data[stats["method"]]["numerrors"]=zeros(length(stats["numerrors"]))
				data[stats["method"]]["Mbetas"]=[Array{Float64}[] for i in 1:nquestions]
				data[stats["method"]]["MSigmas"]=[Array{Float64,2}[] for i in 1:nquestions]
			end
			push!(data[stats["method"]]["x"],[stats["x"][i][1] for i in  1:nquestions])
			push!(data[stats["method"]]["y"],[stats["x"][i][2] for i in  1:nquestions])
			push!(data[stats["method"]]["answer"],[stats["x"][i][stats["answer"][i]] for i in  1:nquestions])
			for i in 1:nquestions
				data[stats["method"]]["numerrors"][i]+=stats["numerrors"][i]
			end
			print("+")
			for i in 1:nquestions
				print(".")
				push!(data[stats["method"]]["Mbetas"][i],stats["center"][i+1])
				rhs=stats["cov"][1][1,1]/sigma[1,1]
				push!(data[stats["method"]]["MSigmas"][i],stats["cov"][i+1]/rhs)
			end
		end
		println("")
	end
	data["ncustomers"]=length(data[collect(data["allmethods"])[1]]["x"])
	close(file)
	data
end

betadistance(meanbetas,truebetas,truemu,ncustomers) = [norm(meanbetas[i]-truebetas[i]) / norm(truebetas[i]-truemu) for i in 1:ncustomers]

betascaleddistance(meanbetas,truebetas,truemu,ncustomers) = betadistance(meanbetas ./ norm.(meanbetas),truebetas ./ norm.(truebetas),truemu / norm(truemu),ncustomers)

relativebetadistance(meanbetas1,meanbetas2,truebetas,truemu,ncustomers) = [norm(meanbetas1[i]-meanbetas2[i]) / norm(truebetas[i]-truemu) for i in 1:ncustomers]
relativebetascaleddistance(meanbetas1,meanbetas2,truebetas,truemu,ncustomers) = relativebetadistance(meanbetas1 ./ norm.(meanbetas1),meanbetas2 ./ norm.(meanbetas2),truebetas ./ norm.(truebetas),truemu / norm(truemu),ncustomers)


RMSE(meanbetas,truebetas,truemu,ncustomers) = vec((vcat(meanbetas...) - vcat(truebetas...)).^2)


scaledRMSE(meanbetas,truebetas,truemu,ncustomers) = RMSE(length(truemu)*meanbetas ./ norm.(meanbetas,1),length(truemu)*truebetas ./ norm.(truebetas,1),truemu / norm(truemu),ncustomers)

Deff(varbetas,m,sigma) = det.(varbetas).^(1/m)

betaDeff2(varbetas,sigma) = Deff(varbetas,2,sigma)

betaDeffdim(varbetas,sigma) = Deff(varbetas,size(sigma,1),sigma)

hitrates(truebetas, estimatedbetas, questions) = [hitrate(truebetas[i], estimatedbetas[i], questions) for i in 1:length(truebetas)]
hitratessample(truebetas, estimatedbetas, questions) = [hitratesample(truebetas[i], estimatedbetas[i], questions) for i in 1:length(truebetas)]
hitratesformula(truebetas, estimatedbetas, questions) = [hitrateformula(truebetas[i], estimatedbetas[i], questions) for i in 1:length(truebetas)]

function computemeasures(data,filename)
	println("Evaluating Quality Measures")
	nfeatures = data["nfeatures"]
	nquestions = data["nquestions"]
	ncustomers = data["ncustomers"]


	hitratequestions = Array{Int64}[]
	allquestionsasked = Set(Array{Int64}[])
	for method in data["allmethods"]
		for customer in 1:ncustomers
			for question in 1:nquestions
				push!(allquestionsasked,data[method]["x"][customer][question]-data[method]["y"][customer][question])
			end
		end
	end
	srand(123)
	while length(hitratequestions) < 1000
		x=rand(-1:1,nfeatures)
		if norm(x)>0 && !in(x,allquestionsasked) && !in(x,hitratequestions) && !in(-x,allquestionsasked) && !in(-x,hitratequestions)
			push!(hitratequestions,x)
		 end
	end
	truemu = data["truemu"]
	truebetas = data["truebetas"]
	sigma = data["sigma"]
	for method in data["allmethods"]
		data[method]["estimatorquality"] = Dict()
		for estimator in keys(data[method]["estimates"])
			currentquality=data[method]["estimatorquality"][estimator]=Dict()
			currentestimates = data[method]["estimates"][estimator]
			betameanmeasures = [betascaleddistance,scaledRMSE]
			betavarmeasures = [betaDeffdim]



			for measure in [betameanmeasures;betavarmeasures;"quality";"hitrate";"hitratesample";"hitrateformula";"marketshare";"marketsharesample";"marketsharehuber";"marketshareformula";"fisherDeffdim";"mfisherDeffdim";"BFisherdistance";"BMFisherdistance"]
				currentquality[string(measure)]=Array{Float64}[]
			end
			for i in 1:(nquestions+1)
				meanbetas = currentestimates["meanbetas"][i]
				varbetas = currentestimates["varbetas"][i]
				quality = currentestimates["quality"][i]
				for measure in betameanmeasures
					push!(currentquality[string(measure)],measure(meanbetas,truebetas,truemu,ncustomers))
				end
				for measure in betavarmeasures
					push!(currentquality[string(measure)],measure(varbetas,sigma))
				end
				push!(currentquality["quality"],[quality])
				push!(currentquality["hitrate"], hitrates(truebetas, meanbetas, hitratequestions))
				push!(currentquality["hitratesample"], hitratessample(truebetas, meanbetas, hitratequestions))
				push!(currentquality["hitrateformula"], hitratesformula(truebetas, meanbetas, hitratequestions))
				push!(currentquality["marketshare"],marketshare(truebetas, meanbetas, hitratequestions))
				push!(currentquality["marketsharesample"],marketsharesample(truebetas, meanbetas, hitratequestions))
				push!(currentquality["marketsharehuber"],marketsharehuber(truebetas, meanbetas, hitratequestions))
				push!(currentquality["marketshareformula"],marketshareformula(truebetas, meanbetas, hitratequestions))
				push!(currentquality["fisherDeffdim"],Deff(currentestimates["fishermatrix"][i],size(sigma,1),sigma))
				push!(currentquality["mfisherDeffdim"],Deff(currentestimates["mfishermatrix"][i],size(sigma,1),sigma))
				push!(currentquality["BFisherdistance"],abs.(currentquality["fisherDeffdim"][end]-currentquality["betaDeffdim"][end]))
				push!(currentquality["BMFisherdistance"],abs.(currentquality["mfisherDeffdim"][end]-currentquality["betaDeffdim"][end]))
			end
		end
	end
	tablemethods = [("Polyhedral","estimateAnalyticCenter_updatePolyhedral_initializeEllipsoid_questionMaxMin"),
								 ("Prob. Poly.","estimateProbPoly_updateNull_initializeEllipsoid_questionMaxMin"),("Robust","estimateAnalyticCenterRobust_updateNull_initializeEllipsoid_questionNormalizedCenterMIP")]
	fishertable = ("Fisher","estimateEllipsoid_updateBayesApproximation_initializeEllipsoid_questionDEffPWLFisher_effDim")
	expectedfishertable = ("Expected Fisher","estimateEllipsoid_updateBayesApproximation_initializeEllipsoid_questionDEffPWLFisherQGK_effDim")
	bayestable = ("Ellipsoidal","estimateEllipsoid_updateBayesApproximation_initializeEllipsoid_questionDEffPWL_effDim")
	savetables(filename,"results/tables/bayescovtables",[bayestable, tablemethods...] )
	savetables(filename,"results/tables/allellipsoid",[fishertable, expectedfishertable, bayestable,("MaxMin","estimateEllipsoid_updateBayesApproximation_initializeEllipsoid_questionMaxMin")])
	savetablesother(filename,"results/tables/bayescovtablesupdate",[bayestable, tablemethods...] )
	savetablesHR(filename,"results/tables/hrtables",[bayestable, tablemethods...])
	savetablesMS(filename,"results/tables/mstables",[bayestable, tablemethods...])

	truemu = data["truemu"]
	truebetas = data["truebetas"]
	nfeatures = data["nfeatures"]
	nquestions = data["nquestions"]
	ncustomers = data["ncustomers"]
	if !onlymethod
			filename = string("results/tables/alldistances.csv")
			if !isfile(filename)
					file = open(filename,"w")
					allstats = ["HBFisher","HBFisherM","BFisher","BFisherM","MFisher","MFisherM"]
					tmp=hcat([[st*" Avg",st*" Max"] for st in allstats]...)
					header2 = string(join(tmp[1,:],","),",",join(tmp[2,:],","))
					println(file,repeat("4,",2*length(allstats)),repeat("8,",2*length(allstats)),repeat("16,",2*length(allstats)))
					println(file,header2,",",header2,",",header2)
			else
				file = open(filename,"a")
			end
			for question in [5,9,17]
				HBFisher=Float64[]
				HBFisherM=Float64[]
				BFisher=Float64[]
				BFisherM=Float64[]
				MFisher=Float64[]
				MFisherM=Float64[]
				for method in data["allmethods"]
					mfdeff = data[method]["estimatorquality"]["Method"]["fisherDeffdim"][question]
					bfdeff = data[method]["estimatorquality"]["STANB"]["fisherDeffdim"][question]
					hbfdeff = data[method]["estimatorquality"]["HB"]["fisherDeffdim"][question]
					mmfdeff = data[method]["estimatorquality"]["Method"]["mfisherDeffdim"][question]
					bmfdeff = data[method]["estimatorquality"]["STANB"]["mfisherDeffdim"][question]
					hbmfdeff = data[method]["estimatorquality"]["HB"]["mfisherDeffdim"][question]
					mdeff = data[method]["estimatorquality"]["Method"]["betaDeffdim"][question]
					bdeff = data[method]["estimatorquality"]["STANB"]["betaDeffdim"][question]
					hbdeff = data[method]["estimatorquality"]["HB"]["betaDeffdim"][question]
					append!(HBFisher,abs.(hbfdeff-hbdeff))
					append!(HBFisherM,abs.(hbmfdeff-hbdeff))
					append!(BFisher,abs.(bfdeff-bdeff))
					append!(BFisherM,abs.(bmfdeff-bdeff))
					if contains(method,"updateBayesApproximation")
						append!(MFisher,abs.(mfdeff-mdeff))
						append!(MFisherM,abs.(mmfdeff-mdeff))
					end
				end
				print(file,mean(HBFisher),",")
				print(file,mean(HBFisherM),",")
				print(file,mean(BFisher),",")
				print(file,mean(BFisherM),",")
				print(file,mean(MFisher),",")
				print(file,mean(MFisherM),",")
				print(file,maximum(HBFisher),",")
				print(file,maximum(HBFisherM),",")
				print(file,maximum(BFisher),",")
				print(file,maximum(BFisherM),",")
				print(file,maximum(MFisher),",")
				print(file,maximum(MFisherM),",")
		end
		close(file)
	end
	if !onlymethod
		for method in data["allmethods"]
			filename = string("results/tables/",method,"_alldistances.csv")
			if !isfile(filename)
					file = open(filename,"w")
					allstats = ["M-BRMSE","M-HBRMSE"]
					header2 = join(hcat([[st*" Avg",st*" Max"] for st in allstats]...),",")
					println(file,header2)
			else
				file = open(filename,"a")
			end
			mb=[]
			mhb=[]
			for question in 2:17
				mest = data[method]["estimates"]["Method"]["meanbetas"][question]
				best = data[method]["estimates"]["STANB"]["meanbetas"][question]
				hbest = data[method]["estimates"]["HB"]["meanbetas"][question]
				mb = [mb;scaledRMSE(mest,best,truemu,ncustomers)]
				mhb = [mhb;scaledRMSE(mest,hbest,truemu,ncustomers)]
			end
			printavamax(file,mb)
			printavamax(file,mhb)
			println(file,"")
			close(file)
		end
	end
end

function printavamax(file,array)
	print(file,mean(array),",",maximum(array),",")
end

function savetablesother(filename,tablename,tablemethods)
	if onlymethod
		return
	end
	file = open(tablename,"a")
	println(file,filename)
	truemu = data["truemu"]
	truebetas = data["truebetas"]
	nfeatures = data["nfeatures"]
	nquestions = data["nquestions"]
	ncustomers = data["ncustomers"]
	for method in tablemethods
		print(file,method[1])

		for question in [5,9,17]
			mest = data[method[2]]["estimates"]["Method"]["meanbetas"][question]
			best = data[method[2]]["estimates"]["STANB"]["meanbetas"][question]
			hbest = data[method[2]]["estimates"]["HB"]["meanbetas"][question]
			print(file,"&")
			print(file,@sprintf("%.2f",sqrt(mean(scaledRMSE(mest,best,truemu,ncustomers)))))
		end
		for question in [5,9,17]
			mest = data[method[2]]["estimates"]["Method"]["meanbetas"][question]
			best = data[method[2]]["estimates"]["STANB"]["meanbetas"][question]
			hbest = data[method[2]]["estimates"]["HB"]["meanbetas"][question]
			print(file,"&")
			print(file,@sprintf("%.2f",sqrt(mean(scaledRMSE(mest,hbest,truemu,ncustomers)))))
		end
		println(file,"\\\\")
	end
	close(file)
end

function savetablesHR(filename,tablename,tablemethods)
	file = open(tablename,"a")
	println(file,filename)


	allestimates = onlymethod ? ["Method"] : ["Method","STANB","HB"]
	for estimator in allestimates
		println(file,estimator,"\\\\")
		for method in tablemethods
			print(file,method[1])
			for question in [5,9,17]
				significant = htestmin([-data[method[2]]["estimatorquality"][estimator]["hitrate"][question], [ -data[m[2]]["estimatorquality"][estimator]["hitrate"][question] for m in setdiff(tablemethods,[method])]...]) # pval<= 0.05
				print(file,"&")
				if significant
					print(file,"\\tablehighlight{")
				end
				print(file,@sprintf("%.2f",mean(data[method[2]]["estimatorquality"][estimator]["hitrate"][question])))
				if  significant
					print(file,"}")
				end
			end
			for question in [5,9,17]
				significant = htestmin([-data[method[2]]["estimatorquality"][estimator]["hitratesample"][question], [ -data[m[2]]["estimatorquality"][estimator]["hitratesample"][question] for m in setdiff(tablemethods,[method])]...]) # pval<= 0.05
				print(file,"&")
				if significant
					print(file,"\\tablehighlight{")
				end
				print(file,@sprintf("%.2f",mean(data[method[2]]["estimatorquality"][estimator]["hitratesample"][question])))
				if  significant
					print(file,"}")
				end
			end
			for question in [5,9,17]
				significant = htestmin([-data[method[2]]["estimatorquality"][estimator]["hitrateformula"][question], [ -data[m[2]]["estimatorquality"][estimator]["hitrateformula"][question] for m in setdiff(tablemethods,[method])]...]) # pval<= 0.05
				print(file,"&")
				if significant
					print(file,"\\tablehighlight{")
				end
				print(file,@sprintf("%.2f",mean(data[method[2]]["estimatorquality"][estimator]["hitrateformula"][question])))
				if  significant
					print(file,"}")
				end
			end
			println(file,"\\\\")
		end
	end
	close(file)
end

function savetablesMS(filename,tablename,tablemethods)
	file = open(tablename,"a")
	println(file,filename)


	allestimates = onlymethod ? ["Method"] : ["Method","STANB","HB"]
	for estimator in allestimates
		println(file,estimator,"\\\\")
		for method in tablemethods
			print(file,method[1])
			for question in [5,9,17]
				significant = htestmin([data[method[2]]["estimatorquality"][estimator]["marketshare"][question], [ data[m[2]]["estimatorquality"][estimator]["marketshare"][question] for m in setdiff(tablemethods,[method])]...]) # pval<= 0.05
				print(file,"&")
				if significant
					print(file,"\\tablehighlight{")
				end
				print(file,@sprintf("%.2f",mean(data[method[2]]["estimatorquality"][estimator]["marketshare"][question])))
				if  significant
					print(file,"}")
				end
			end
			for question in [5,9,17]
				significant = htestmin([data[method[2]]["estimatorquality"][estimator]["marketsharesample"][question], [ data[m[2]]["estimatorquality"][estimator]["marketsharesample"][question] for m in setdiff(tablemethods,[method])]...]) # pval<= 0.05
				print(file,"&")
				if significant
					print(file,"\\tablehighlight{")
				end
				print(file,@sprintf("%.2f",mean(data[method[2]]["estimatorquality"][estimator]["marketsharesample"][question])))
				if  significant
					print(file,"}")
				end
			end
			for question in [5,9,17]
				significant = htestmin([data[method[2]]["estimatorquality"][estimator]["marketsharehuber"][question], [ data[m[2]]["estimatorquality"][estimator]["marketsharehuber"][question] for m in setdiff(tablemethods,[method])]...]) # pval<= 0.05
				print(file,"&")
				if significant
					print(file,"\\tablehighlight{")
				end
				print(file,@sprintf("%.2f",mean(data[method[2]]["estimatorquality"][estimator]["marketsharehuber"][question])))
				if  significant
					print(file,"}")
				end
			end
			for question in [5,9,17]
				significant = htestmin([data[method[2]]["estimatorquality"][estimator]["marketshareformula"][question], [ data[m[2]]["estimatorquality"][estimator]["marketshareformula"][question] for m in setdiff(tablemethods,[method])]...]) # pval<= 0.05
				print(file,"&")
				if significant
					print(file,"\\tablehighlight{")
				end
				print(file,@sprintf("%.2f",mean(data[method[2]]["estimatorquality"][estimator]["marketshareformula"][question])))
				if  significant
					print(file,"}")
				end
			end
			println(file,"\\\\")
		end
	end
	close(file)
end

function savetables(filename,tablename,tablemethods,mfisher=false)



	allestimates = onlymethod ? ["Method"] : ["Method","STANB","HB"]
	for estimator in allestimates
		file = open(string(tablename,"_",estimator,".txt"),"a")
		println(file,filename)
		for method in tablemethods
			print(file,method[1])
			if estimator == "Method"
				for question in [5,9,17]
					if mfisher
						significant = htestmin([data[method[2]]["estimatorquality"][estimator]["mfisherDeffdim"][question], [ data[m[2]]["estimatorquality"][estimator]["mfisherDeffdim"][question] for m in setdiff(tablemethods,[method])]...])
					else
						significant = htestmin([data[method[2]]["estimatorquality"][estimator]["fisherDeffdim"][question], [ data[m[2]]["estimatorquality"][estimator]["fisherDeffdim"][question] for m in setdiff(tablemethods,[method])]...])
					end
					print(file,"&")
					if significant
						print(file,"\\tablehighlight{")
					end
					if mfisher
						print(file,@sprintf("%.2f",mean(data[method[2]]["estimatorquality"][estimator]["mfisherDeffdim"][question])))
					else
						print(file,@sprintf("%.2f",mean(data[method[2]]["estimatorquality"][estimator]["fisherDeffdim"][question])))
					end
					if significant
						print(file,"}")
					end
				end
			else
				for question in [5,9,17]
					significant = htestmin([data[method[2]]["estimatorquality"][estimator]["betaDeffdim"][question], [ data[m[2]]["estimatorquality"][estimator]["betaDeffdim"][question] for m in setdiff(tablemethods,[method])]...])
					print(file,"&")
					if significant
						print(file,"\\tablehighlight{")
					end
					print(file,@sprintf("%.2f",mean(data[method[2]]["estimatorquality"][estimator]["betaDeffdim"][question])))
					if significant
						print(file,"}")
					end
				end
			end
			for question in [5,9,17]
				significant = htestmin([data[method[2]]["estimatorquality"][estimator]["scaledRMSE"][question], [ data[m[2]]["estimatorquality"][estimator]["scaledRMSE"][question] for m in setdiff(tablemethods,[method])]...]) # pval<= 0.05
				print(file,"&")
				if significant
					print(file,"\\tablehighlight{")
				end
				print(file,@sprintf("%.2f",sqrt(mean(data[method[2]]["estimatorquality"][estimator]["scaledRMSE"][question]))))
				if  significant
					print(file,"}")
				end
			end
			for question in [5,9,17]
				significant = htestmin([-data[method[2]]["estimatorquality"][estimator]["hitratesample"][question], [ -data[m[2]]["estimatorquality"][estimator]["hitratesample"][question] for m in setdiff(tablemethods,[method])]...]) # pval<= 0.05
				print(file,"&")
				if significant
					print(file,"\\tablehighlight{")
				end
				print(file,@sprintf("%.2f",mean(data[method[2]]["estimatorquality"][estimator]["hitratesample"][question])))
				if  significant
					print(file,"}")
				end
			end
			for question in [5,9,17]
				significant = htestmin([data[method[2]]["estimatorquality"][estimator]["marketsharehuber"][question], [ data[m[2]]["estimatorquality"][estimator]["marketsharehuber"][question] for m in setdiff(tablemethods,[method])]...]) # pval<= 0.05
				print(file,"&")
				if significant
					print(file,"\\tablehighlight{")
				end
				print(file,@sprintf("%.2f",mean(data[method[2]]["estimatorquality"][estimator]["marketsharehuber"][question])))
				if  significant
					print(file,"}")
				end
			end
			println(file,"\\\\")
		end
		close(file)
	end
end


# Correcting for multiple comparisons with Holm–Bonferroni method
function htestmin(data)
	pvalues = Float64[]
	for i in 1:length(data)
		for j in (i+1):length(data)
			try
				push!(pvalues,pvalue(UnequalVarianceTTest(data[i],data[j]);tail=:left))
			catch
				println("PVAL=",data)
				return false
			end
		end
	end
	p=sortperm(pvalues)
	k=maximum(findin(p,collect(1:(length(data)-1))))
	bin = length(data)*(length(data)-1)/2
	maximum(pvalues[1:(length(data)-1)]) <= 0.05 / (bin+1-k)
end

function getMethodEstimates(question,X,Y,A,mu,sigma,nfeatures,ncustomers,Mbetas,MSigmas)
	estimatorname = string("Method")

	meanbetas = Mbetas[question]
	varbetas = MSigmas[question]
	quality = 1

	estimatorname,  meanbetas, varbetas, quality
end

function getHBStanEstimates(question,X,Y,A,mu,sigma,nfeatures,ncustomers,Mbetas,MSigmas)
	o_Sigma, o_mu, o_beta, quality, totaltime = HBposteriorSTANFactorized(X[:,1:question,:],Y[:,1:question,:],A[:,1:question,:],mu,sigma,1,nfeatures +2)
	meanbetas = [vec(mean(o_beta[:,:,customer],2)) for customer in 1:ncustomers]
	varbetas = [cov(o_beta[:,:,customer],2) for customer in 1:ncustomers]

	estimatorname = "HB"
	estimatorname, meanbetas, varbetas, quality
end

function getBStanEstimates(question,X,Y,A,mu,sigma,nfeatures,ncustomers,Mbetas,MSigmas)
	o_beta, quality, totaltime = BposteriorSTAN(X[:,1:question,:],Y[:,1:question,:],A[:,1:question,:],mu,sigma)
	meanbetas = [vec(mean(o_beta[:,:,customer],2)) for customer in 1:ncustomers]
	varbetas = [cov(o_beta[:,:,customer],2) for customer in 1:ncustomers]

	estimatorname = "STANB"
	estimatorname, meanbetas, varbetas, quality
end


function fisherMatrix(Sigma,x,y,mu)
	inv(inv(Sigma) + sum.(((2+2*cosh(dot(x[i]-y[i],mu)))^(-1))*(x[i]-y[i])*(x[i]-y[i])' for i in 1:length(x)))
end

function computeestimators(data)
	println("\n Calculating estimates:")
	nfeatures = data["nfeatures"]
	nquestions = data["nquestions"]
	ncustomers = data["ncustomers"]
	mu = data["mu"]
	Sigma = data["sigma"]

	allestimates = onlymethod ? [getMethodEstimates] : [getMethodEstimates,getBStanEstimates,getHBStanEstimates]
	for method in data["allmethods"]
		print(method,":")
		data[method]["estimates"] = Dict()
		X=cat(3,[hcat(ee...) for ee in data[method]["x"]]...)
		Y=cat(3,[hcat(ee...) for ee in data[method]["y"]]...)
		A=cat(3,[hcat(ee...) for ee in data[method]["answer"]]...)
		for getEstimates in allestimates
			print("+")
			for question in 1:nquestions
				print(".")
				estimatorname, meanbetas, varbetas, quality = getEstimates(question,X,Y,A,mu,Sigma,nfeatures,ncustomers,data[method]["Mbetas"],data[method]["MSigmas"])
				if !haskey(data[method]["estimates"], estimatorname)
					data[method]["estimates"][estimatorname] = Dict("meanbetas" => Array{Array{Float64,1}}[], "varbetas" => Array{Array{Float64,2}}[], "quality" => Float64[], "fishermatrix" =>Array{Array{Float64,2}}[], "mfishermatrix" =>Array{Array{Float64,2}}[])
					push!(data[method]["estimates"][estimatorname]["meanbetas"],[mu for i in 1:ncustomers])
					push!(data[method]["estimates"][estimatorname]["varbetas"],[Sigma for i in 1:ncustomers])
					push!(data[method]["estimates"][estimatorname]["fishermatrix"],[Sigma for i in 1:ncustomers])
					push!(data[method]["estimates"][estimatorname]["mfishermatrix"],[Sigma for i in 1:ncustomers])
					push!(data[method]["estimates"][estimatorname]["quality"],1)
				end
				push!(data[method]["estimates"][estimatorname]["meanbetas"],meanbetas)
				push!(data[method]["estimates"][estimatorname]["varbetas"],varbetas)
				push!(data[method]["estimates"][estimatorname]["quality"],quality)
				push!(data[method]["estimates"][estimatorname]["fishermatrix"],[fisherMatrix(Sigma,data[method]["x"][i][1:question],data[method]["y"][i][1:question],meanbetas[i]) for i in 1:ncustomers])
				push!(data[method]["estimates"][estimatorname]["mfishermatrix"],[fisherMatrix(Sigma,data[method]["x"][i][1:question],data[method]["y"][i][1:question],mu) for i in 1:ncustomers])
			end
		end
		println("")
	end
end

filename = ARGS[1]

const global onlymethod = length(ARGS) > 1 ? true : false

println("\nStatistics for ",filename,"\n")

data = []
hitdatafilename = string("results/tmp/",basename(filename),"_hr.dat")
hbdatafilename = string("results/tmp/",basename(filename),"_hb.dat")
fpdatafilename = string("results/tmp/",basename(filename),"_fp.dat")
if isfile(hitdatafilename)
	file = open(hitdatafilename,"r")
	data = deserialize(file)
	close(file)
elseif isfile(hbdatafilename)
	file = open(hbdatafilename,"r")
	data = deserialize(file)
	close(file)
	computemeasures(data,filename)
	file = open(hitdatafilename,"w")
	serialize(file,data)
	close(file)
elseif isfile(fpdatafilename)
	file = open(fpdatafilename,"r")
	data = deserialize(file)
	close(file)
	computeestimators(data)
	file = open(hbdatafilename,"w")
	serialize(file,data)
	close(file)
	computemeasures(data,filename)
	file = open(hitdatafilename,"w")
	serialize(file,data)
	close(file)
else
	data = computefirstpassstats(filename)
	file = open(fpdatafilename,"w")
	serialize(file,data)
	close(file)
	computeestimators(data)
	file = open(hbdatafilename,"w")
	serialize(file,data)
	close(file)
	computemeasures(data,filename)
	file = open(hitdatafilename,"w")
	serialize(file,data)
	close(file)
end

println("Writting files...")


methods=collect(data["allmethods"])
estimators = collect(keys(data[methods[1]]["estimatorquality"]))
measures=keys(data[methods[1]]["estimatorquality"][estimators[1]])
for measure in measures
	for estimator in estimators
		for level in [0.05;0.10;0.15;0.20;0.25;0.30;0.35;0.4]
			file = open(string("results/csv/",basename(filename),"_timeplot_",estimator,"_",measure,"_",level,".csv"),"w")
			for k in data["allmethods"]

				print(file,k)
				for i in 1:length(data[k]["estimatorquality"][estimator][measure])
					print(file,",",quantile(data[k]["estimatorquality"][estimator][measure][i],level))
				end
				println(file,"")
				print(file,k)
				for i in 1:length(data[k]["estimatorquality"][estimator][measure])
					print(file,",",median(data[k]["estimatorquality"][estimator][measure][i]))
				end
				println(file,"")
				print(file,k)
				for i in 1:length(data[k]["estimatorquality"][estimator][measure])
					print(file,",",quantile(data[k]["estimatorquality"][estimator][measure][i],1-level))
				end
				println(file,"")
			end
			close(file)
		end
	end
end


for stat in ["numerrors"]
		file = open(string("results/csv/",basename(filename),"_timeplot_",stat,".csv"),"w")
		for k in data["allmethods"]
			print(file,k)
			for i in 1:length(data[k][stat])
				print(file,",",data[k][stat][i])
			end
			println(file,"")
		end
		close(file)
end



println("Done.")
