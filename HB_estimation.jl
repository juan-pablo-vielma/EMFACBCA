module HB_estimationModule

export HBposteriorSTANFactorized, BposteriorSTAN

using Distributions, Stan, Mamba

function BposteriorSTAN(x,y,alpha,mu_0,Sigma_0)


  tic()
  N           = 5000       # total iterations to be recorded in the MCMC procedure
  numchains = 4
  n_questions = size(x,2)   # number of questions per respondents
  n_customers = size(x,3)   # number of respondents
  n_features = size(x,1)


  o_beta  = zeros(n_features,N*numchains,n_customers)


  const hb_stan_model ="
      data {
        int n_obs;                       // number of observations
        int n_features;                  // number of attributes
        int n_questions;                // number of questions
        int n_customers;                // number of respondents
        int y[n_obs];                   // respondents answers
        int id[n_obs];                  // index respondent
        matrix[n_obs, n_features] x;    // questions asked
        row_vector[n_features] mu_0;                             // prior on mu
        cov_matrix[n_features] Sigma_0;                      // prior on Sigma
        cholesky_factor_cov[n_features] rootSigma_0;                      // prior on Sigma
      }
      parameters {
        row_vector[n_features] mu;                               // population mean
        matrix[n_features, n_customers] z;
      }
      transformed parameters {
        matrix[n_customers, n_features] mbeta;
        mbeta = rep_matrix(mu_0, n_customers) + (rootSigma_0 * z)';
      }
      model {
        to_vector(z) ~ normal(0, 1);
        y ~ bernoulli_logit(rows_dot_product(mbeta[id] , x));              // each answer from a logit
      }
    "
  x_data = zeros(n_questions*n_customers,n_features);
  z_data = zeros(n_questions*n_customers,1);
  for t in 1:n_customers
    x_data[(t-1)*n_questions+1:t*n_questions,:] = (x[:,:,t] - y[:,:,t])';
    z_data[(t-1)*n_questions+1:t*n_questions]   = prod(alpha[:,:,t].==x[:,:,t],1)';
  end
  id     = cumsum(repmat([1 ; zeros(n_questions-1,1)],n_customers,1));
  z_data = squeeze(z_data,2); id = squeeze(id,2); mu_0 = vec(mu_0);
  const HB_data= [
  Dict("n_obs"=> size(z_data,1),
         "id" => id,
         "n_features" => n_features,
         "n_questions" => n_questions,
         "n_customers" => n_customers,
         "y" => z_data,
         "x" => x_data,
         "mu_0" => mu_0,
         "Sigma_0" => Sigma_0,
         "rootSigma_0" => ctranspose(chol(Sigma_0))
    )
  ]

  global stanmodel, rc, sim


  
  originalSTDOUT = STDOUT
  originalSTDERR = STDERR
  errfile = open("STAN_errorfile.txt","w")
  redirect_stderr(errfile)
  outfile = open("STAN_out.txt","w")
  redirect_stdout(outfile)
  stanmodel = Stanmodel(num_samples = N, nchains =numchains, thin = 1, name ="STANestimation", model = hb_stan_model);
  rc, sim = stan(stanmodel, HB_data, CmdStanDir = CMDSTAN_HOME;summary=false)
  diag = gelmandiag(sim, mpsrf=true, transform=true)


  redirect_stderr(originalSTDERR)
  redirect_stdout(originalSTDOUT)
  close(errfile)
  close(outfile)
  mpsrf = 0
  if rc == 0
    for i in 1:n_features
      for h in 1:n_customers
          o_beta[i,:,h]  = sim[:,string("mbeta.",h,".",i),1:numchains].value[:]
          mpsrf = 0
          mpsrf = max(mpsrf,diag.value[indexin([string("mbeta.",h,".",i)],diag.rownames),2][1])
      end
    end
  end
  totaltime = toq();
  return o_beta, mpsrf, totaltime
end


function HBposteriorSTANFactorized(x,y,alpha,mu_0,Lambda_0,kappa_0,nu_0)


  tic()
  N           = 5000       # total iterations to be recorded in the MCMC procedure
  numchains = 4
  n_questions = size(x,2)   # number of questions per respondents
  n_customers = size(x,3)   # number of respondents
  n_features = size(x,1)

  o_Sigma = zeros(n_features,n_features,N*numchains)
  o_mu    = zeros(n_features,N*numchains)
  o_beta  = zeros(n_features,N*numchains,n_customers)

  const hb_stan_model ="
    data {
      int n_obs;                       // number of observations
      int n_features;                  // number of attributes
      int n_questions;                // number of questions
      int n_customers;                // number of respondents
      int y[n_obs];                   // respondents answers
      int id[n_obs];                  // index respondent
      matrix[n_obs, n_features] x;    // questions asked
      real kappa;                     // prior parameter for mu
      vector[n_features] mu_0;                             // prior on mu
      cov_matrix[n_features] Sigma_0;                      // prior on Sigma
    }
    parameters {
      row_vector[n_features] mu;                               // population mean
      matrix[n_features, n_customers] z;
    }
    transformed parameters {
      matrix[n_customers, n_features] mbeta;
      mbeta = rep_matrix(mu, n_customers) + (sqrt(Sigma_0[1,1]) * z)';
    }
    model {
      mu ~ multi_normal(mu_0, Sigma_0/kappa);
      to_vector(z) ~ normal(0, 1);
      y ~ bernoulli_logit(rows_dot_product(mbeta[id] , x));              // each answer from a logit
    }
  generated quantities {
    matrix[n_features,n_features] Sigma;
    Sigma = Sigma_0;
    }
    "
  x_data = zeros(n_questions*n_customers,n_features);
  z_data = zeros(n_questions*n_customers,1);
  for t in 1:n_customers
    x_data[(t-1)*n_questions+1:t*n_questions,:] = (x[:,:,t] - y[:,:,t])';
    z_data[(t-1)*n_questions+1:t*n_questions]   = prod(alpha[:,:,t].==x[:,:,t],1)';
  end
  id     = cumsum(repmat([1 ; zeros(n_questions-1,1)],n_customers,1));
  z_data = squeeze(z_data,2); id = squeeze(id,2); mu_0 = vec(mu_0);
  const HB_data= [
  Dict("n_obs"=> size(z_data,1),
         "id" => id,
         "n_features" => n_features,
         "n_questions" => n_questions,
         "n_customers" => n_customers,
         "y" => z_data,
         "x" => x_data,
         "mu_0" => mu_0,
         "Sigma_0" => Lambda_0,
         "kappa" => kappa_0
    )
  ]

  global stanmodel, rc, sim


  
  originalSTDOUT = STDOUT
  originalSTDERR = STDERR
  errfile = open("STAN_errorfile.txt","w")
  redirect_stderr(errfile)
  outfile = open("STAN_out.txt","w")
  redirect_stdout(outfile)
  stanmodel = Stanmodel(num_samples = N, nchains =numchains, thin = 1, name ="STANestimation", model = hb_stan_model);
  rc, sim = stan(stanmodel, HB_data, CmdStanDir = CMDSTAN_HOME;summary=false)
  diag = gelmandiag(sim, mpsrf=true, transform=true)

  redirect_stderr(originalSTDERR)
  redirect_stdout(originalSTDOUT)
  close(errfile)
  close(outfile)
  Sigma_mpsrf = 0
  mu_mpsrf = 0
  beta_mpsrf = 0
  if rc == 0
    for i in 1:n_features
      o_mu[i,:]    = sim[:,string("mu.",i),1:numchains].value[:]
      mu_mpsrf = max(mu_mpsrf,diag.value[indexin([string("mu.",i)],diag.rownames),2][1])
      for j in 1:n_features
        o_Sigma[i,j,:] = sim[:,string("Sigma.",i,".",j),1:numchains].value[:]
        Sigma_mpsrf = max(Sigma_mpsrf,diag.value[indexin([string("Sigma.",i,".",j)],diag.rownames),2][1])
      end
      for h in 1:n_customers
          o_beta[i,:,h]  = sim[:,string("mbeta.",h,".",i),1:numchains].value[:]
          beta_mpsrf = max(beta_mpsrf,diag.value[indexin([string("mbeta.",h,".",i)],diag.rownames),2][1])
      end
    end
  end
  totaltime = toq();
  mpsrf = max(mu_mpsrf,beta_mpsrf)
  return o_Sigma, o_mu, o_beta, mpsrf, totaltime
end


end
