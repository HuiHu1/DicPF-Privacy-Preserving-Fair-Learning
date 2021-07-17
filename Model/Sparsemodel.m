clear
clc;
tic
t1=toc;
%import data
DataSample = csvread('data/communitycrime/crimecommunity.csv');
randomset = csvread('data/communitycrime/crimecommunity_index.csv');
dic = csvread('data/communitycrime/fair_accurate_dic.csv');
%model hyperparameters
iter=50;
featureNum=99;
num_iter=1500;
%observe the process
for i=1:iter
    data1(i,num_iter)=Inf;
    SP(i,num_iter)=Inf;
    data_minority(i,num_iter)=Inf;
    data_majority(i,num_iter)=Inf;
    num_min(i,num_iter)=Inf;
    num_ma(i,num_iter)=Inf;
    positive_num(i,num_iter)=0;
end
%record the final results
for i=1:iter
    data1_final(i,1)=Inf;
    SP_final(i,1)=Inf;
    positive(i,1)=Inf;
end
W_fair = dic;
% model part
for loop=1:iter
    normal_data = DataSample(:,2:100);
    trainset = randomset(1:1493,loop);
    testset = randomset(1494:1993,loop);
    train_x = normal_data(trainset(:,1),:);
    train_label = DataSample(trainset(:,1),102);
    test_x = normal_data(testset(:,1),:);
    test_label = DataSample(testset(:,1),102);
    [row, column]=size(W_fair);
    beta = rand(column,1);
    lamda = 50; %grid search in [1,1000]
    Inner_loop=0;
    while(Inner_loop<num_iter)
        Inner_loop = Inner_loop+1;
        j = randi(column,1,1);
        W_fair_bank = W_fair(:,[1:j-1 j+1:end]);
        beta_bank = beta([1:j-1,j+1:end]);
        A_j = train_x*W_fair_bank*beta_bank-train_label;
        Q_j = 2*transpose(train_x*W_fair(:,j))*A_j;
        R_j = 2*transpose(train_x*W_fair(:,j))*(train_x*W_fair(:,j));
        if(Q_j<-2*lamda)
            beta(j,1)=(-2*lamda-Q_j)/R_j;
        end
        if(Q_j>2*lamda)
            beta(j,1)=(2*lamda-Q_j)/R_j;
        end
        if(abs(Q_j)<=2*lamda)
            beta(j,1) = 0;
        end
        prediction_label = test_x*(W_fair*beta);
        [row column]=size(prediction_label);
        for i = 1:row
            if(abs(prediction_label(i,1))>=0.5)
                prediction_label(i,1)=1;
            else
                prediction_label(i,1)=0;
            end
        end
        positive_num(loop,Inner_loop)=sum(prediction_label);
        [trow,tcolumn]=size(test_x);
        count =0;
        for j=1:trow
            if(abs(prediction_label(j,1))~=test_label(j,1))
                count =count+1;
            end
        end
        error = count/trow;
        [Minority_num,Majority_num,sub_error_min,sub_error_ma,Fair_SP] = Calcualte(prediction_label,test_label,loop,DataSample,randomset);
        data_minority(loop,Inner_loop)=sub_error_min;
        data_majority(loop,Inner_loop)=sub_error_ma;
        num_min(loop,Inner_loop)=Minority_num;
        num_ma(loop,Inner_loop)=Majority_num;
        data1(loop,Inner_loop) = error;
        SP(loop,Inner_loop)=Fair_SP;
    end
    if(num_min(loop,num_iter)==0 || num_ma(loop,num_iter)==0)
        disp("No positive predictions!!");
        break;
    else
        temp_positive = positive_num(loop,num_iter);
        temp_data1 = data1(loop,num_iter);
        temp_sp= SP(loop,num_iter);
    end
    if(data1_final(loop,1)>temp_data1)
        data1_final(loop,1) = temp_data1;
        SP_final(loop,1) = temp_sp;
        positive(loop,1)=temp_positive;
    end
end
data1_final(any(isinf(data1_final),2),:) = [];
SP_final(any(isinf(SP_final),2),:) = [];
positive(any(isinf(positive),2),:) = [];
diviation_SP = std(SP_final);
diviation_Error = std(data1_final);
t2=toc;
display(strcat('The running time is: ',num2str(t2),'s'));