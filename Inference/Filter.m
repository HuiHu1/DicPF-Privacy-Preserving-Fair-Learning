function [NEW_P,E] = Filter(P,E)
%filter too small values
threshold = 0.001;
[r1,c1]=size(E);
for j =1:r1
    if(E(j)==0 || E(j)==1 || isnan(E(j)))
        E(j)=Inf;
        P(j,:)=Inf;
    end
end
P(any(isinf(P),2),:) = [];
E(any(isinf(E),2),:) = [];
NEW_P = P;
end

