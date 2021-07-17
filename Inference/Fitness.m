function [E]=Fitness(pre,P)
[x1 y1]=size(P);
E=zeros(x1,1);
for rr=1:x1
    infer_s=P(rr,:);
    minority = size(find(infer_s(1,:)==1));
    majority = size(find(infer_s(1,:)==0));
    count1 = 0;
    count2 = 0;
    for i =1:y1
        if((pre(i,1)==1) && infer_s(1,i)==1)
            count1 = count1+1;
        end
        if((pre(i,1)==1) && infer_s(1,i)==0)
            count2 = count2+1;
        end
    end
    ratio_Minority =  count1/minority(1,2);
    ratio_Majority =  count2/majority(1,2);
    E(rr,1) = abs(ratio_Minority-ratio_Majority);
end
end