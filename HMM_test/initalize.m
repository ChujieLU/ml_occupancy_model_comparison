function [Mn, Hn, Wn, O, Q] = initalize()
  %% Initialize
  % Mn: # of M values, boolean, {0,1}, Mn = 2
  % Hn: # of timesteps in a day, 30-min increment, {0,1,...,47},Hn = 48
  % Wn: # of W values, boolean, {0,1}, Wn = 2
  % O: size of observation set = # (M,H,W) combinations; as per toolbox
  % Q: size of state set; as per toolbox
  Mn = 2;
  Hn = 48;
  Wn = 2;
  O = Mn * Hn * Wn;
  Q = 2; %2 or 3 state;
end
