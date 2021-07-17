clear
clc;
tic
t1=toc;
%import data
DataSample = csvread('data/communitycrime/crimecommunity.csv');
infer_sample_size=1993; 
non_sensitive = DataSample(1:infer_sample_size,2:100);
true_label = DataSample(1:infer_sample_size,102);
true_sensitive = DataSample(1:infer_sample_size,1);
%GA
p=200; %[200,500]
c=0.1; %[0.1,0.3]
m=0.01;  
tg=500; %hyperparameter(to converge)
iteration = 20;  
P = Population(p,infer_sample_size);
[x1 y1]=size(P);
P_best_i = zeros(tg,infer_sample_size); % 
PL_GA =zeros(iteration,1);
for loop =1:iteration
    pre = Sparsemodel(DataSample(1:infer_sample_size,:));  
    for i=1:tg
        P = Crossover(P,c);
        P = Mutation(P,m);
        [E] = Fitness(pre,P);
        [P,E] = Filter(P,E);
        [P,S] = Selection(P,E);
        [row column]=size(P);
        K(i,1)=sum(S)/row; %average
        [value,index] = min(S);
        K(i,2)=value; %best
        P_best_i(i,:)=P(index,:);
    end
    [best_fitness_value,best_index]= min(transpose(K(:,2)));
    P2 = P_best_i(best_index,:); 
    P_loss=0;
    for i =1:infer_sample_size
        P_loss = P_loss+(P2(1,i)-true_sensitive(i,1))^2;
    end
    P_loss = 1-(1/infer_sample_size)*P_loss;  
    PL_GA(loop,1)=P_loss;
end
t2=toc;
display(strcat('Excution time is:',num2str(t2),'s'));
