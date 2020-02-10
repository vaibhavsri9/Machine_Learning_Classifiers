% Fisher LDA trial
clear all;
%% Creation of samples
n = 2; %no. of feature dimensions
N = 10000; % no. of iid samples
mu(:,1) = [-0.1;0]; mu(:,2) = [0.1,0];
Sigma(:,:,1) = [1 -0.9;-0.9 1]; Sigma(:,:,2) = [1 0.9;0.9 1];
p = [0.8,0.2]; % class priors for labels 0 and 1
label = rand(1,N) >= p(1);
Nc = [length(find(label==0)),length(find(label==1))]; % number of samples from each class
x = zeros(n,N); % save up space
% Draw samples
for l = 0:1
    %x(:,label==l) = randGaussian(Nc(l+1),mu(:,l+1),Sigma(:,:,l+1));
    x(:,label==l) = mvnrnd(mu(:,l+1),Sigma(:,:,l+1),Nc(l+1))';
end
data_0 = [];
data_1 = [];
for i = 1 : N
    if label(i) == 0
        data_0 = [data_0; x(:, i)'];
    end
    if label(i) == 1
        data_1 = [data_1; x(:, i)'];
    end
end

%% Calcualtion of sample based estimated mean and covariance
mu1h = mean(data_0, 1)';
mu2h = mean(data_1, 1)';
S1h = cov(data_0);
S2h = cov(data_1);

%% Calculation of between/within-class scatter matrices 
Sb = (mu1h-mu2h)*(mu1h-mu2h)';
Sw = S1h + S2h;

%% Fisher LDA proj vector
[V, D] = eig(inv(Sw)*Sb);
% Arranging the diagnol elements
[~,ind] = sort(diag(D),'descend');
w_LDA = V(:,ind(1));

%% Projecting from both classes
data_0 = data_0';
data_1 = data_1';
result_0 = w_LDA'*data_0;
% projection of 1st vector and 2nd vector
result_1 = w_LDA'*data_1;

%% Plotting data 
figure(1),
subplot(2,1,1), plot(data_0(1,:),data_0(2,:),'r+'); hold on;
plot(data_1(1,:),data_1(2,:),'bo'); axis equal, 
title('Sample Data')
subplot(2,1,2), plot(result_0(1,:),zeros(1,Nc(1)),'r*'); hold on;
plot(result_1(1,:),zeros(1,Nc(2)),'bo'); axis equal,
title('Sample Data projection')

%% Classifier Design
projected_mean_d0 = w_LDA'*mean(data_0,2);
projected_mean_d1 = w_LDA'*mean(data_1,2);

TP = [];
FP = [];

minm_err = [];
error_list = [];
g_list = [];
tpr_minm_err = [];
fpr_minm_err = [];
threshold_minm_err = 0;

if projected_mean_d0 < projected_mean_d1
    
    for threshold = (min([result_0,result_1])-2):0.5:(max([result_0,result_1])+2)
        piid_00=[];
        piid_01=[];
        piid_10=[];
        piid_11=[];
        
        for q=1:size(result_0, 2)
            if result_0(q) > threshold
                piid_01 = [piid_01, result_0(q)];
            else
                piid_00 = [piid_00, result_0(q)];
            end
        end
        
        for q=1:size(result_1, 2)
            if result_1(q) > threshold
                piid_10 = [piid_10, result_1(q)];
            else
                piid_11 = [piid_11, result_1(q)];
            end
        end
    
        p_1_given_0 = size(piid_01,2)/size(result_0,2);
        p_1_given_1 = size(piid_11,2)/size(result_1,2);
        
        error = (size(piid_01,2) + size(piid_10,2))/N ;
        error_list = [error_list error];
        g_list = [g_list threshold];
        
       % Replace this logic
    if(isempty(minm_err))
        minm_error = error;
        threshold_minm_err = threshold;
        tpr_minm_err = p_1_given_1;
        fpr_minm_err = p_1_given_0;
    elseif(error < minm_err)
        minm_error = error;
        tpr_minm_err = p_1_given_1;
        fpr_minm_err = p_1_given_0;
        threshold_minm_err = threshold;
    end
    
    TP = [TP; p_1_given_1];
    FP = [FP; p_1_given_0];
    
    
    end
end

figure(2);
plot( FP', TP');
axis equal;
ylabel('True Positve Probability'), xlabel('False Positve Probability');
title('ROC');
hold on;
plot( fpr_minm_err, tpr_minm_err, 'r*');
title('ROC Curve')



