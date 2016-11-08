% testing with sparse vectors rather than images
% function [MSPE,MSIE] =  sparse_vectors(mode)
clear;
clc;
% check data
% f = 0.1;
% N = 10;
% no_iter = 2;
f = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05]; f = flip(f);
% f = [0.05, 0.1, 0.5, 0.8]; % Fraction of measurements
N = 100; % Number of test signals
no_iter = 500; % For measurement matrix
% k = 120; % Length of sparse vectors
T = 8; % sparsity
% n = k;
mode = 0; % 0: t%  ||  1: t as threshold 
% Dictionary 1 (DCT)
k = 64; % Length of sparse vectors
rows_dict = k;
cols_dict = k;
dict = kron(dctmtx(8)', dctmtx(8)'); % 2-D dct matrix
n = k;

% rows_dict = 80;
% cols_dict = k;
% dict = randn(rows_dict,cols_dict);
t = 0.2;
t_set = 0.2;
gamma = 0.5; % Shrinking factor
alpha = zeros(cols_dict,N); % sparse co-efficients
x = zeros(rows_dict,N); % actual measurements
estimatedX = zeros(size(x));
MSIE_paper = zeros(length(f),1);
MSIE_original = zeros(length(f),1);
rand_meas = randn(cols_dict,rows_dict); % 80 x 120
% x = D*alpha
% y = P*D*alpha
%% Generating sparse measurements
for i = 1:N
    loc = ceil((k-1).*rand(T,1) + 1); % location of non-zero values \in (1,k+1)
    value = randn(T,1); % values in that location
    for j = 1:T
        alpha(loc(j),i) = value(j);
    end
    x(:,i) = dict*alpha(:,i); % rows_dict x 1 ||  actual measurements
end
    
for itr = 1:length(f)
    tic;
    no_meas = ceil(f(itr)*cols_dict);
    p = no_meas
    P = rand_meas(1:no_meas,:); % random initialization for measurement matrix
    coherence = zeros(no_iter,1);

    %% Measurement matrix
    for l = 1:no_iter
        % Effective Dictionary
        unnormalized = P*dict; % no_meas x n
        temp1 = sqrt(sum(unnormalized.*unnormalized));
        temp2 = repmat(temp1,p,1);
        D_n = unnormalized./temp2; % no_meas x n

        % Gram matrix
        G = D_n'*D_n; % n x n

        % Set threshold
        if mode == 0
            thresh = ceil(t*n*(n-1)); % percent of off-diagonal elements
            % Set t such that thresh number of elements are > t;
            G_temp1 = reshape(G,n*n,1);
            G_temp2 = sort(G_temp1); % sorts elements of G in ascending order
            t_set = (G_temp2(n*n-n-thresh)+G_temp2(n*n-n-thresh-1))/2;
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
        P = S/dict; % S*pinv(D) % p x n
    end
    %% Compressed Measurements
    y = P*dict*alpha; % no_meas x N

    %% Reconstruction using low coherence measurement matrix
    % Applying OMP
    A = P*dict; % no_meas x n
    A_modified = zeros(no_meas,n);
    for j = 1:n
        A_modified(:,j) = A(:,j)./(norm(A(:,j),2));
    end

    for i = 1:N
        r = y(:,i);
        s = 0;
        j = 0;
        m = 1;
        epsilon = 0.01;
        supportSet = [];
        while norm(r,2)>epsilon && m<=no_meas
            [~,index] = max(abs(A_modified'*r)); % 1 x n
            supportSet = [supportSet index];
            Atemp = A(:,supportSet);
            s = pinv(Atemp)*y(:,i);
            r = y(:,i) - Atemp*s;
            m = m+1;
        end
        estimatedTheta = zeros(n,1);
        estimatedTheta(supportSet) = s;
        estimatedX(:,i) = dict*estimatedTheta;
    end
%     MSIE_paper(itr) = sum(sum(abs(x-estimatedX)))/(N*rows_dict);
    MSIE_paper(itr) = RRMSE(x,estimatedX);
    
    %% Reconstruction using random measurement matrix
    % Applying OMP
    P = rand_meas(1:no_meas,:);;
    A = P*dict; % no_meas x rows_dict
    
    % Compressed Measurements
    y = P*dict*alpha; % no_meas x N
    
    % finding coherence
    unnormalized = P*dict; 
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
    coherence_orig = find_coherence(G,t_set);
    
    A_modified = zeros(no_meas,n);
    for j = 1:n
        A_modified(:,j) = A(:,j)./(norm(A(:,j),2));
    end

    for i = 1:N
        r = y(:,i);
        s = 0;
        j = 0;
        m = 1;
        epsilon = 0.01;
        supportSet = [];
        while norm(r,2)>epsilon && m<=no_meas
            [~,index] = max(abs(A_modified'*r)); % 1 x n
            supportSet = [supportSet index];
            Atemp = A(:,supportSet);
            s = pinv(Atemp)*y(:,i);
            r = y(:,i) - Atemp*s;
            m = m+1;
        end
        estimatedTheta = zeros(n,1);
        estimatedTheta(supportSet) = s;
        estimatedX(:,i) = dict*estimatedTheta;
    end
%     MSIE_original(itr) = sum(sum(abs(x-estimatedX)))/(N*rows_dict);
    MSIE_original(itr) = RRMSE(x,estimatedX);
    toc;
    
%     xaxis = 1:1:no_iter;
%     figure;
%     plot(xaxis,coherence,xaxis,coherence_orig*(ones(no_iter,1)));
%     legend('Low coherence matrix','Random gaussian matrix');
%     title(strcat('Coherence-meas=',int2str(no_meas)),  'FontWeight','bold');
%     xlabel('number of iterations');
%     ylabel('Coherence');
%     temp = strcat('coherence_synthetic_comparison_no-meas=',int2str(no_meas),'.png');
%     saveas(gcf,temp);
end


figure;
plot(ceil(f*n), MSIE_original, ceil(f*n), MSIE_paper);
legend('MSE-original','MSE-paper');
title('MSE vs no-meas',  'FontWeight','bold');
xlabel('number of measurements');
ylabel('MSE');
temp = strcat('MSE_synthetic data_sparsity=',int2str(T),'_gamma=',int2str(gamma),'.png');
saveas(gcf,temp);


% end