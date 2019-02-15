function [best_train_accuracy, best_transmat_train, best_obsmat_train, best_train_error, test_table] = EM_test_script_v4(tstat, test, n_trials)
  %% Main function to be able to call and run the HMM test for a single theromostat
  %% inputs: thermostat, test number and how many trials to consider
  %% outputs: result table and number of best metrics
  prediction_length = 6;

  [Mn, Hn, Wn, O, Q] = initalize();

  fname_test = sprintf('../demo_test/HMM/data/test/test_%s_%s.csv',test,tstat);
  test_table = readtable(fname_test);

  fname_train = sprintf('../demo_test/HMM/data/train/train_%s_%s.csv',test,tstat);
  train_table = readtable(fname_train);

  test_array = test_table.encoded;
  test_array = test_array.';
  train_array = train_table.encoded;
  train_array = train_array.';

  train_array = [];
  H_train = [];
  W_train = [];
  M_train = [];

  %% -- Training --
  for row = 1:(size(train_table,1)-28)
      temp_table = train_table(row:row+28,:);
      time_delta = temp_table.Var1(size(temp_table,1)) - temp_table.Var1(1);
      if time_delta == hours(14.0)
          train_array = [train_array; temp_table.encoded.'];
          H_train = [H_train, temp_table.H_t.'];
          W_train = [W_train, temp_table.W_t.'];
          M_train = [M_train, temp_table.M_t.'];
      end
  end



  for trial = 1:n_trials
    
    tic
    % initial guess of parameters
    prior1 = normalise(rand(Q,1)); % leaves you with a normalized column vector
    trans_mat1 = mk_stochastic(rand(Q,Q));
    obs_mat1 = mk_stochastic(rand(Q,O));

    % improve guess of parameters using EM
    [LL, prior2, trans_mat2, obs_mat2] = dhmm_em(train_array, prior1, trans_mat1, obs_mat1, 'max_iter', 100);

    % use model to compute log likelihood; as per toolbox
    loglik = dhmm_logprob(train_array, prior2, trans_mat2, obs_mat2);
    % log lik is slightly different than LL(end), since it is computed after the final M step


    % Viterbi Algorithm applied on Training Data
    obs_lik = multinomial_prob(train_array, obs_mat2);
    result_seq = viterbi_path(prior2, trans_mat2, obs_lik)-1;
    % Observation liklikhood

    % obs_seq is the most likely M readings in the observations
    obs_seq = zeros(1,size(result_seq,2));
    for i = 1:size(result_seq,2)
        % Since (S,H,W) in (M,S,H,W) are actually known in sequence, we
        % can compare the probability of (0,S,H,W) with (1,S,H,W) in
        % the Emission Probability Matrix obsmat2; the one with higher
        % probability tells us if M = 0 or 1
        index1 = W_train(i) + Wn*H_train(i) + Wn*Hn*0 + 1;
        index2 = W_train(i) + Wn*H_train(i) + Wn*Hn*1 + 1;

        if obs_mat2(result_seq(i)+1,index1) > obs_mat2(result_seq(i)+1, index2)
            obs_seq(i) = 0;
        else
            obs_seq(i) = 1;
        end
    end
    fprintf('elapsed training for a single iteration %f \n', toc)
    % Accuracy
    % obs_seq is the returned M readings from Viterbi
    % error is where obs_seq and actual M readings do not match
    error = abs(obs_seq - M_train);
    train_accuracy = 0;
    % calc the current accuracy in this trial
    train_accuracy = 1 - (sum(error)/size(obs_seq,2));

    if trial == 1
      best_train_accuracy = train_accuracy;
      best_prior_train = prior2;
      best_transmat_train = trans_mat2;
      best_obsmat_train = obs_mat2;
      best_obs_lik_train = obs_lik;
      best_train_error = error;
    elseif train_accuracy > best_train_accuracy
      best_train_accuracy = train_accuracy;
      best_prior_train = prior2;
      best_transmat_train = trans_mat2;
      best_obsmat_train = obs_mat2;
      best_obs_lik_train = obs_lik;
      best_train_error = error;
    end

  end

  %% Testing
  date_times = [];
  val_lists = [];
  test_obs = [];

  A = best_transmat_train;
  B = best_obsmat_train;

  for row = 1:(size(test_table,1)-28)
      temp_table = test_table(row:row+28,:);
      time_delta = temp_table.Var1(size(temp_table,1)) - temp_table.Var1(1);
      if time_delta == hours(14.0)
          date_times = [date_times;temp_table.Var1(24)];

          test_array = temp_table.encoded(1:23);
          test_array = test_array.';

          H = temp_table.H_t(24:end);
          H = H.';

          W = temp_table.W_t(24:end);
          W = W.';

          M = temp_table.M_t(24:end);
          M = M.';

          test_obs = [test_obs; M(1)];

          test_obs_lik = multinomial_prob(test_array, best_obsmat_train);

          % avoid all zero entries, due to rounding
          for row1=1:size(test_obs_lik,1)
              for col=1:size(test_obs_lik,2)
                  if test_obs_lik(row1,col) == 0
                       test_obs_lik(row1,col) = 1e-23;% add a very small number
                  end
              end
          end

          % using Viterbi Algorithm
          tic
          state_seq = viterbi_path(best_prior_train, best_transmat_train, test_obs_lik)-1; %minus 1 just to make ones and zeros
          
          % use forward back to get the posterior for states
          % gamma is the smooth posterior marginal
          % aplha is the forwared pass, beta the back
          [alpha, beta, gamma, loglik, xi, gamma2] = fwdback(best_prior_train, best_transmat_train, test_obs_lik,'fwd_only',1,'scaled',1);
          toc

          %% Infer the Most Likely Observations
          % obs_seq is the most likely M readings in the observations
           obs_seq = zeros(1,prediction_length);
           belief_state = alpha(:,end);

           for i = 1:prediction_length

             belief_state = sum(A.*belief_state,2);

             temp_val = B.*belief_state;
             temp_val = sum(temp_val,1);

             % Since (S,H,W) in (M,S,H,W) are actually known in sequence, we
             % can compare the probability of (0,H,W) with (1,H,W) in
             % the Emission Probability Matrix obsmat2; the one with higher
             % probability tells us if M = 0 or 1

             index1 = W(i) + Wn*H(i) + Wn*Hn*0 + 1;
             index2 = W(i) + Wn*H(i) + Wn*Hn*1 + 1;
             emission.matrix = temp_val(:,[index1 index2]);
             index_num = argmax(emission.matrix);

             if index_num == 1
                 obs_seq(i) = 0;
             else
                 obs_seq(i) = 1;
             end

           end

           val_lists = [val_lists; obs_seq];

      end
  end

  test_table = array2table(val_lists,'VariableNames',{'M_t0' 'M_t1' 'M_t2' 'M_t3' 'M_t4' 'M_t5'});
  test_table.M_t = test_obs;
  test_table.date_time = date_times;
end
