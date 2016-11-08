function [ rrmse ] = RRMSE(X, Y)
%UNTITLED Summary of this function goes here
%   Returns the relative root mean squared error
% X is the original matrix, Y is the new matrix

rrmse = sqrt(sum(sum((abs(X)-abs(Y)).^2)))/sqrt(sum(sum(abs(X).^2)));

end

