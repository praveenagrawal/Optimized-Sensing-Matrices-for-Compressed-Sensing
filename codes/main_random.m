MSIE_original = zeros(50,11);
MSIE_paper = zeros(50,11);
for j = 1:50
    j
    [MSIE_original(j,:),MSIE_paper(j,:)] =  sparse_vectors_random();
end

save('original_algo_random','MSIE_original');
save('updated_algo_random','MSIE_paper');