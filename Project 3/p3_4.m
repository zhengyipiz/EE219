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
