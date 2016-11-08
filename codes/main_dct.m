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