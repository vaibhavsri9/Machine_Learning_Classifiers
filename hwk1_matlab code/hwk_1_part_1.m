%Code for Hwk1 Minimum Expected Risk Classifier
clear all;
%% Creation of Samples
n = 2; %no. of feature dimensions
N = 10000; % no. of iid samples
mu(:,1) = [-0.1;0]; mu(:,2) = [0.1,0];
Sigma(:,:,1) = [1 -0.9;-0.9 1]; Sigma(:,:,2) = [1 0.9;0.9 1];
p = [0.8,0.2]; % class priors for labels 0 and 1
label = rand(1,N) >= p(1);
Nc = [length(find(label==0)),length(find(label==1))]; % number of samples from each class
x = zeros(n,N); % save up space
% Draw samples from each class pdf
for l = 0:1
    %x(:,label==l) = randGaussian(Nc(l+1),mu(:,l+1),Sigma(:,:,l+1));
    x(:,label==l) = mvnrnd(mu(:,l+1),Sigma(:,:,l+1),Nc(l+1))';
end
figure(1), clf,
plot(x(1,label==0),x(2,label==0),'o'), hold on,
plot(x(1,label==1),x(2,label==1),'+'), axis equal,
legend('Class 0','Class 1'), 
title('Data and their true labels'),
xlabel('x_1'), ylabel('x_2'), 

%% Loss Incorporation 
%lambda = [0 1;1 0]; % loss values
%gamma = (lambda(2,1)-lambda(1,1))/(lambda(1,2)-lambda(2,2)) * p(1)/p(2); %threshold
discriminantScore = log(evalGaussian(x,mu(:,2),Sigma(:,:,2)))-log(evalGaussian(x,mu(:,1),Sigma(:,:,1)));% - log(gamma);
p_detection = zeros(1000,1);
p_false_alarm = zeros(1000,1);
w = 1;
for gamma = 0:0.5:250
   
    decision = (discriminantScore >= log(gamma));
    ind00 = find(decision==0 & label==0); p00 = length(ind00)/Nc(1); % probability of true negative
    ind10 = find(decision==1 & label==0); p10 = length(ind10)/Nc(1); % probability of false positive
    ind01 = find(decision==0 & label==1); p01 = length(ind01)/Nc(2); % probability of false negative
    ind11 = find(decision==1 & label==1); p11 = length(ind11)/Nc(2); % probability of true positive
    p_detection(w) = p11;
    p_false_alarm(w) = p10;
    w= w+1;
end
figure(2)
plot(p_false_alarm,p_detection);hold on;
lambda = [0 1;1 0]; % loss values
gamma = (lambda(2,1)-lambda(1,1))/(lambda(1,2)-lambda(2,2)) * p(1)/p(2); %threshold
discriminantScore = log(evalGaussian(x,mu(:,2),Sigma(:,:,2)))-log(evalGaussian(x,mu(:,1),Sigma(:,:,1)))
decision = (discriminantScore >= log(gamma));
    ind00 = find(decision==0 & label==0); p00 = length(ind00)/Nc(1);
    ind10 = find(decision==1 & label==0); p10 = length(ind10)/Nc(1);
    ind01 = find(decision==0 & label==1); p01 = length(ind01)/Nc(2); 
    ind11 = find(decision==1 & label==1); p11 = length(ind11)/Nc(2); 
    p_error = [p10,p01]*Nc'/N;
figure(2)
scatter(p10,p11,'r*');
title('ROC Curve and marking the minimum error possible point')
xlabel('Probability of false detection'),ylabel('Probability of detection');
%% Plotting the correct and incorrect 
figure(3), % class 0 circle, class 1 +, correct green, incorrect red
plot(x(1,ind00),x(2,ind00),'og'); hold on,
plot(x(1,ind10),x(2,ind10),'or'); hold on,
plot(x(1,ind01),x(2,ind01),'+r'); hold on,
plot(x(1,ind11),x(2,ind11),'+g'); hold on,
axis equal,

%% Draw the decision boundary
horizontalGrid = linspace(floor(min(x(1,:))),ceil(max(x(1,:))),101);
verticalGrid = linspace(floor(min(x(2,:))),ceil(max(x(2,:))),91);
[h,v] = meshgrid(horizontalGrid,verticalGrid);
discriminantScoreGridValues = log(evalGaussian([h(:)';v(:)'],mu(:,2),Sigma(:,:,2)))-log(evalGaussian([h(:)';v(:)'],mu(:,1),Sigma(:,:,1))) - log(gamma);
minDSGV = min(discriminantScoreGridValues);
maxDSGV = max(discriminantScoreGridValues);
discriminantScoreGrid = reshape(discriminantScoreGridValues,91,101);
% Check for values
figure(3), contour(horizontalGrid,verticalGrid,discriminantScoreGrid,[minDSGV*[0.9,0.6,0.3],0,[0.3,0.6,0.9]*maxDSGV]); % plot equilevel contours of the discriminant function 
% including the contour at level 0 which is the decision boundary
legend('Correct decisions for data from Class 0','Wrong decisions for data from Class 0','Wrong decisions for data from Class 1','Correct decisions for data from Class 1','Equilevel contours of the discriminant function' ), 
title('Data and their classifier decisions versus true labels'),
xlabel('x_1'), ylabel('x_2'), 

