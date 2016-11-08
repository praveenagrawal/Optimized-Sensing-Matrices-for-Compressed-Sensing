clear;
clc;
img = imread('../Data/Original.png');
[rows,cols,~] = size(img);
imgInput = rgb2gray(img);
% [rows,cols,~] = size(img);

% imgInput = rgb2gray(img);
patchSize = 8; % patchsize
n = patchSize*patchSize;
t = 0.2;
mode = 0;
f = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]; f = flip(f);
% f = [0.05, 0.1, 0.3, 0.5, 0.7, 0.9];
% f = [0.7,0.4,0.3]; f = flip(f);
MSIE_paper = zeros(length(f),1);
MSIE_orig = zeros(length(f),1);
gamma = 0.5;
no_iter = 100;
rand_meas = randn(n,n);
coherence = zeros(length(f),no_iter);
for i = 1:length(f)
    i
disp 'Finding phi ...';
[phi,G,coherence(i,:)] = find_P(patchSize,f(i),t,gamma,no_iter,mode,rand_meas); % m x n
disp 'phi_found';
disp 'Running OMP';
MSIE_paper(i) = OMP(imgInput,patchSize,f(i),phi,'lowCoherence');
MSIE_orig(i) = OMP(imgInput,patchSize,f(i),rand_meas(1:ceil(f(i)*n),:),'randomGaussian');
end

% y = 1:1:no_iter;
% figure;
% plot(y,coherence(1,:),y,coherence(2,:),y,coherence(3,:),y,coherence(4,:),y,coherence(5,:),y,coherence(6,:));
% legend('f = 0.05','f = 0.1', 'f = 0.3', 'f = 0.5', 'f = 0.7', 'f = 0.9')
% title('Coherence vs iteration number',  'FontWeight','bold');
% xlabel('Iteration number');
% ylabel('Coherence');
% temp = strcat('coherence_',int2str(mode),'.png');
% saveas(gcf,temp);

figure;
plot(ceil(f*n), MSIE_paper,ceil(f*n), MSIE_orig)
legend('Low coherence','Random')
% plot(ceil(f*n), MSIE_paper,'r');
% plot(ceil(f*n), MSIE_orig,'b');
title('RRMSE vs m',  'FontWeight','bold');
xlabel('m');
ylabel('RRMSE');
temp = 'RRMSE_comparison.png';
saveas(gcf,temp);

%% Testing on synthetic data
clear;
T = [4,8,12,16];
for j = 1:length(T)
    j
    sparse_vectors_DCT(1,T(j));
    sparse_vectors_DCT(2,T(j));
    sparse_vectors_DCT1(1,T(j));
    sparse_vectors_DCT1(2,T(j));
end

T = [4,8,12,24,30];
for j = 1:length(T)
    j
    sparse_vectors_random(T(j),1);
    sparse_vectors_random(T(j),2);
end
% [MSPE1,MSIE1] = sparse_vectors(1);
% [MSPE0,MSIE0] = sparse_vectors(0);
% 
% 
% figure;
% plot(ceil(f*n), MSIE0,ceil(f*n), MSPE0,ceil(f*n), MSIE1,ceil(f*n), MSIE1)
% legend('Total mean square error (orig)','Average of vector-wise error(orig)','Total mean square error (paper)','Average of vector-wise error(paper)')
% title('MSE vs no_meas',  'FontWeight','bold');
% xlabel('number of measurements');
% ylabel('MSE');
% temp = 'MSE_synthetic data.png';
% saveas(gcf,temp);
