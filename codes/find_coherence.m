function coherence = find_coherence(G,t)
% Caclulate mutual coherence
sum_coherence = 0;
count = 0;
[n,m] = size(G);
for i = 1:n
    for j = 1:m
        if (i~=j)
            if abs(G(i,j))>=t
                sum_coherence = sum_coherence + abs(G(i,j));
                count = count+1;
            end
        end
    end
end
coherence = sum_coherence/count;
end