function Y=Crossover(P,c)
[x1 y1] = size(P);
n = round(x1*c);
for i =1:n
    r1=randi(x1,1,2);
    while r1(1)==r1(2)
        r1=randi(x1,1,2);
    end
    A1=P(r1(1),:); 
    A2=P(r1(2),:); 
    r2=1+randi(y1-1);  
    B1=A1(1,r2:y1);
    A1(1,r2:y1)=A2(1,r2:y1);
    A2(1,r2:y1)=B1;
    P(r1(1),:)= A1;
    P(r1(2),:)=A2;
end
Y=P;
end

