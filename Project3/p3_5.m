% EE219 UCLA
% PROJECT 3
% PART 5:
% Recommendation, L.
% BY YI & ZIWEN
% WINTER 2017

clear;

% Parameters determined from previous parts
% Threshold for like/dislike
TH_train=3;
TH=0.96;
% NMF
k=10;
lambda=0.01;

% import 100k data
data = importdata('u.data');
user = data(:,1);
item = data(:,2);
rating = data(:,3);

% setup indices for 10-fold cross-validation
indices = crossvalind('Kfold', 100000, 10);

% Matrix for Top 20 movies for each user
% Approximate matrix UV for 10 folds
Top20=zeros(max(user), max(item), 10); % 10 folds
UV=Top20;

% 10-fold cross validation
for cvi = 1:10
    % convert dataset to matrix R
    % generate weight matrix W
    R=zeros(max(user),max(item));
    W=R;
    for i = 1:size(rating)
    % training data
        if indices(i) ~= cvi
            % swapped R and W
            W(user(i),item(i))=rating(i);
            R(user(i),item(i))=1;
        end
    end      
    [U,V,numIter,tElapsed,finalResidual] = l2wnmfrule(R,k,lambda);
    % predicted 
    UV(:,:,cvi)=U*V;
    % sort the predicted matrix for each user in descending order
    [~, I]=sort(UV(:,:,cvi), 2, 'descend');
    % mark the top 20 movies with its rank for each user in each fold
    % e.g. the 5th movie is marked as 5
    for u=1:max(user)
        for j=1:20
            Top20(u, I(u,j), cvi)=j;
        end
    end
end

% convert dataset to matrix R
R=zeros(max(user),max(item));
for i=1:size(rating)
    R(user(i),item(i))=rating(i);
end

% calculate average Precision for L=5
L=5;
precision_fold=zeros(1,10);
for i=1:10
    % the top L known items for fold i
    top_items = ((Top20(:,:,i)~=0) & (Top20(:,:,i)<=L) & R~=0); %top L and known for fold i
    UVi=UV(:,:,i);
    precision_fold(i)=length(find(R(top_items)>TH_train & UVi(top_items)>TH))/length(find(UVi(top_items)>TH));
end
average_precision=mean(precision_fold);
    
% calculate average hit rate and false alarm rate
% L=1:20
average_hit_rate=zeros(1,20);
average_FA_rate=zeros(1,20);
for L=1:20
    Hit_rate_fold=zeros(1,10);
    FA_rate_fold=zeros(1,10);
    for i=1:10
        % the top L known items for fold i
        top_items = ((Top20(:,:,i)~=0) & (Top20(:,:,i)<=L) & (R~=0)); %top L and known for fold i
        UVi=UV(:,:,i);
        Hit_rate_fold(i)=length(find(UVi(top_items)>TH & R(top_items)>TH_train))/length(find(R(top_items)>TH_train));
        FA_rate_fold(i)=length(find(UVi(top_items)>TH & R(top_items)<=TH_train))/length(find(R(top_items)<=TH_train));
    end
    average_hit_rate(L)=mean(Hit_rate_fold);
    average_FA_rate(L)=mean(FA_rate_fold);
end

% print precision for L=5
fprintf('Avg Precision(L=5) = %.2f\n', average_precision);
% plot hit rate vs. false alarm rate
plot(average_FA_rate, average_hit_rate);
xlabel('False Alarm Rate');
ylabel('Hit Rate');
title('Hit Rate vs. False Alarm Rate for L=1-20');