module evaluationModule



export  hitrate, marketshare


# Calculares the MAE of the estiamted marketshare
#
# Parameters:
#
# - truebetas = array with vector of true partworths for each customer
# - estimatedbetas = array with vector of estimated partworths for each customer
# - questions = pairwise questions to estimate marketshare
#
# Returns
#
# - MAE of marketshare

function marketshare(truebetas, estimatedbetas, questions)
	abserrors = Float64[]
	for q in questions
		trueshare = 0
		for b in truebetas
			if dot(b,q) >=0
				trueshare+=1
			end
		end
		trueshare /= length(truebetas)
		estimatedshare = 0
		for b in estimatedbetas
			if dot(vec(b),q) >=0
				estimatedshare+=1
			end
		end
		estimatedshare /= length(estimatedbetas)
		push!(abserrors,abs(estimatedshare-trueshare))
	end
	return abserrors
end

# Calculares the hitrate for an estimated partworth vector
#
# Parameters:
#
# - truebetas = vector of true partworths
# - estimatedbetas = vector of estimated partworths
# - questions = pairwise questions to calculate hitrate
#
# Returns
#
# - hitrate

function hitrate(truebeta, estimatedbeta, questions)
	hits=0
	for q in questions
		if dot(truebeta,q)*dot(vec(estimatedbeta),q) >=0
			hits+=1
		end
	end
	return hits/length(questions)
end

end
