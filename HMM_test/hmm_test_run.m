clear all; clc;

addpath(genpath('~/Documents/HMMall')); %location of tool package
addpath(genpath('~/Documents/PhD_Work/occupancy_paper_final/HMM_test')); %locations of these tools

dirData = dir('~/Documents/PhD_Work/occupancy_paper_final/demo_test/HMM/data/train/*.csv'); %location of data

for f_num = 1:size(dirData,1)
  
  file = dirData(f_num);
  tstat = split(file.name,'_');
  tstat = split(tstat(4),'.');
  tstat = tstat(1);
  tstat = tstat{1};

  test = split(file.name,'_');
  test = strcat(test{2},'_',test{3});

  tic;
  [best_train_accuracy, best_transmat_train, best_obsmat_train, best_train_error, test_table] = EM_test_script_v4(tstat, test, 30);
  toc;
  save_fname =  sprintf('../demo_test/HMM/test_em_finish_%s_%s.csv',test,tstat); % a csv for each thermostat.
  writetable(test_table,save_fname,'Delimiter',',');

end
