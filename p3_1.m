% EE219 UCLA
% PROJECT 3
% PART 1:
% NMF, K=10, 50, AND 100.
% BY YI & ZIWEN
% WINTER 2017

clear;

% import 100k data
data = importdata('u.data');
user = data(:,1);
item = data(:,2);
rating = data(:,3);

% convert dataset to matrix R
% generate weight matrix W
R=zeros(max(user),max(item));
W=R;
for i = 1:size(rating)
   R(user(i),item(i))=rating(i);
   W(user(i),item(i))=1;
end

% perform NMF and calculate the least square error
[U,V,numIter,tElapsed,finalResidual] =wnmfrule(R,10);
error = W.*(R-(U*V)).^2;
sum1 =sum(error(:));
[U2,V2,numIter2,tElapsed2,finalResidual2] =wnmfrule(R,50);
error2 = W.*(R-(U2*V2)).^2;
sum2 =sum(error2(:));
[U3,V3,numIter3,tElapsed3,finalResidual3] =wnmfrule(R,100);
error3 = W.*(R-(U3*V3)).^2;
sum3 =sum(error3(:));

% print the results
fprintf('Least Square Error = %f (k=10),%f (k=50),%f (k=100)',sum1,sum2,sum3);