function MSIE = OMP(imgInput,patchSize,F,phiM,temp_string)
% imgInput = imread('../Data/barbara256.png');
% Initializations
% patchSize = 8; % patchsize
n = patchSize*patchSize; % actual number of measurements
% F = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05];
MSPE = zeros(1, length(F));    % Mean Square Patch Error
MSIE = zeros(1, length(F));    % Mean Square Image Error
for itr = 1:length(F)
    f = F(itr);
    m = ceil(f*n); % compressed number of measurements
    U = kron(dctmtx(patchSize)', dctmtx(patchSize)'); % 2-D dct matrix
    [rows,columns] = size(imgInput);
    numOfPatches = length(1:rows-patchSize+1)*length(1:columns-patchSize+1);
    y = zeros(m,numOfPatches); % compressed measurement matrix
    x = zeros(n,numOfPatches); % actual measurement matrix
    % Generate matrix phi (64 x 64) ~ N(0,1)
%     phi = randn(n);
%     phiM = phi(1:m,:);
    A = phiM*U;  % m x 64
    % Dividing into overlapping patches and generating measurements
    k = 1;
    for i = 1:rows-patchSize+1
        for j = 1:columns-patchSize+1
            tempPatch = imgInput(i:i+patchSize-1,j:j+patchSize-1);
            x(:,k) = reshape(tempPatch,n,1);
            y(:,k) = phiM*x(:,k);
%             y(:,k) = y(:,k) + 0.05*mean(abs(y(:,k)))*randn(m,1);
            k = k+1;
        end
    end

    % Applying OMP
    A_modified = zeros(m,n);
    for j = 1:n
        A_modified(:,j) = A(:,j)./(norm(A(:,j),2));
    end

    estimatedX = zeros(size(x));
    for i = 1:numOfPatches
        r = y(:,i);
        s = 0;
        j = 0;
        k = 1;
        epsilon = 0.1;
        supportSet = [];
        while norm(r,2)>epsilon && k<=m
            [~,index] = max(abs(A_modified'*r)); % 1 x n
            supportSet = [supportSet index];
            Atemp = A(:,supportSet);
            s = pinv(Atemp)*y(:,i);
            r = y(:,i) - Atemp*s;
            k = k+1;
        end
        estimatedTheta = zeros(n,1);
        estimatedTheta(supportSet) = s;
        estimatedX(:,i) = U*estimatedTheta;
        MSPE(itr) = MSPE(itr) + sum((x(:,i)-estimatedX(:,i)).^2)/n;
    end
    MSPE(itr) = MSPE(itr)/numOfPatches;
    % Reconstruct Image
    reconstructedImg = zeros(rows, columns);
    weights = zeros(rows, columns);
    k = 1;
    for i = 1:rows-patchSize+1
        for j = 1:columns-patchSize+1
            reconstructedImg(i:i+patchSize-1,j:j+patchSize-1) = ...
                reconstructedImg(i:i+patchSize-1,j:j+patchSize-1) + ...
                reshape(estimatedX(:,k),patchSize, patchSize);
            weights(i:i+patchSize-1,j:j+patchSize-1) = weights(i:i+patchSize-1,j:j+patchSize-1) + 1;
            k = k + 1;
         end
    end
    reconstructedImg = reconstructedImg ./ weights;
%     MSIE(itr) = sum(sum((double(imgInput)-reconstructedImg).^2))/(rows*columns);
    MSIE(itr) = RRMSE(double(imgInput),reconstructedImg);
    figure;
    imshow(uint8(reconstructedImg));
    title(strcat('Reconstructed Image for f =  ', num2str(f)),  'FontWeight','bold');
    temp = strcat(int2str(m),temp_string,'.png');
    saveas(gcf, temp);
end
% figure;
% plot(ceil(F*n), MSPE);
% title('MSPE vs m',  'FontWeight','bold');
% xlabel('m');
% ylabel('MSPE');
% 
% figure;
% plot(ceil(F*n), MSIE);
% title('MSIE vs m',  'FontWeight','bold');
% xlabel('m');
% ylabel('MSIE');
end