function Y = Mutation(P,m)
% P = population
% n = chromosomes to be mutated
[x1 y1]=size(P);
n = round(x1*m);
for i=1:n
    r1=randi(x1);
    A1=P(r1,:);  
    r2=randi(y1);
    if A1(1,r2)==1
        A1(1,r2)=0;
    else
        A1(1,r2)=1;
    end
    P(i,:)=A1;
end
Y=P;
end

