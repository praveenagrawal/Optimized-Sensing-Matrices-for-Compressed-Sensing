function [P,G,coherence] = find_P(patch_size,f,t,gamma,no_iter,mode,rand_meas)
%% Objective: Minimize mu_t{PD} wrt P
%% Input
% patch_size = 8;
n = patch_size*patch_size;
% f = 0.8;
p = ceil(f*n); % number of measurements
% t = 0.2; % coherence threshold
% 2-D dct matrix as dictionary
% D = kron(dctmtx(patch_size), dctmtx(patch_size)'); % n x n
D = orth(randn(n,n));
coherence = zeros(no_iter,1);
% gamma = 0.5; % down-scaling factor
% no_iter = 50;

%% Initialization
P = rand_meas(1:p,:); % p x n

%% Loop
for l = 1:no_iter
    % Effective Dictionary
    unnormalized = P*D; % p x n
    temp1 = sqrt(sum(unnormalized.*unnormalized));
    temp2 = repmat(temp1,p,1);
    % normalized
    D_n = unnormalized./temp2; % p x n

    % Gram matrix
    G = D_n'*D_n; % n x n
    % Set threshold
    if mode==0 % fixed percent of elements for coherence
        thresh = ceil(t*n*(n-1)); % percent of off-diagonal elements
        % Set t such that thresh number of elements are > t;
        G_temp1 = reshape(G,n*n,1);
        G_temp2 = sort(G_temp1); % sorts elements of G in ascending order
        t_set = (G_temp2(n*n-n-thresh)+G_temp2(n*n-n-thresh-1))/2;
    elseif mode==1 % fixed lower_bound
        t_set = t;
    end
   
    coherence(l) = find_coherence(G,t_set);
    
    % Shrink and update Gram matrix
    for i = 1:n
        for j = 1:n
            if (abs(G(i,j)) >= t_set)
                G(i,j) = gamma*G(i,j);
            elseif (abs(G(i,j)) < t_set && abs(G(i,j)) >= gamma*t_set)
                G(i,j) = gamma*t_set*sign(G(i,j));
            else
                G(i,j) = G(i,j);
            end
        end
    end

    % Reduce Rank
    [U_G,S_G,V_G] = svd(G);
    G_rankp = U_G(:,1:p)*S_G(1:p,1:p)*(V_G(:,1:p))';

    % Square root
    [G_ev,G_d] = eig(G_rankp);
    S = G_ev(:,1:p)*sqrt(abs(G_d(1:p,1:p))); % G = G_ev*G_d*G_ev' % n x p
    S = S'; % p x n

    % Update P
    P = S/D; % S*inv(D) % p x n
end
figure()
plot(coherence);
end
