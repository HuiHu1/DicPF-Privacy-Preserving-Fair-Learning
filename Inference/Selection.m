function [YY1,YY2] = Selection(P,F)
[x y]=size(P);
[value,idx] = sort(F); %rank
p_num = round(0.8*x);
YY1 = P(idx(1:p_num),:);
YY2 = value(idx(1:p_num));