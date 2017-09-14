function [ net_Q ] = Q_nn_init_DQN_from_simulation( WeightsPerLayer )

%   import weights and biases for a 100X100X40 Q neural network for the DQN algorithm, pre-trained in python using Tensorflow


%********************************define and init Q nn********************
net_Q=feedforwardnet([WeightsPerLayer,WeightsPerLayer,40]);

X=[1,0.5,-1,-0.5,1;1,-1,1,0.5,1;1,0.5,-1,-0.5,1;1,0.5,-1,-0.5,1];   %X is the input and determines the input dimension of the nn (here output is 4X1) - numbers are only for determining dimensions and have no meaning/impact
T=[1,0,1,0,1;0,1,0,1,0;1,0,1,0,1;1,0,1,0,1;1,0,1,0,1];  %T is the target and determines the output dimensions of the nn (here output is 5X1) - numbers are only for determining dimensions and have no meaning/impact

net_Q.trainFcn = 'traingdm';          %Gradient Descent Backpropagation
net_Q.trainParam.epochs = 1;           %1 epoch becuase we are batch-training
net_Q.trainParam.lr = 0.0001;
net_Q.trainParam.mc = 0.99;
net_Q.trainParam.showWindow=false;    %show GUI for training results
net_Q.divideFcn='dividetrain';       %assign all examples to training (and not to validation/testing)
net_Q.performFcn='mse';             %mean square error performance function
%net_Q.performFcn='crossentropy';  
init(net_Q);                        %initialize net
net_Q=train(net_Q,X,T);



%%  LOAD WEIGHTS AND BIASES FROM TENSORFLOW

%% weights 
%% input layer
A = load('net.IW{1}(all,1).mat');
net_Q.IW{1}(:,1) = A.arr';

A = load('net.IW{1}(all,2).mat');
net_Q.IW{1}(:,2) = A.arr';

A = load('net.IW{1}(all,3).mat');
net_Q.IW{1}(:,3) = A.arr';

A = load('net.IW{1}(all,4).mat');
net_Q.IW{1}(:,4) = A.arr';

%%  middle and out layers

for i=1:100
    address = strcat('net.LW{2,1}(all,',int2str(i),').mat');
    A = load(address);
    net_Q.LW{2,1}(:,i) = A.arr';
end

for i=1:100
    address = strcat('net.LW{3,2}(all,',int2str(i),').mat');
    A = load(address);
    net_Q.LW{3,2}(:,i) = A.arr';
end

for i=1:40
    address = strcat('net.LW{4,3}(all,',int2str(i),').mat');
    A = load(address);
    net_Q.LW{4,3}(:,i) = A.arr';
end


%% biases

A = load('net.b{1}.mat');
net_Q.b{1}(:) = A.arr';

A = load('net.b{2}.mat');
net_Q.b{2}(:) = A.arr';

A = load('net.b{3}.mat');
net_Q.b{3}(:) = A.arr';

A = load('net.b{4}.mat');
net_Q.b{4}(:) = A.arr';


%load values to network:
%this example is for a 4 input, 3 output,hidden layers: 10X10X40 
%% input layer:
% net1.IW

% ans =
% 
%   4ª1 cell array
% 
%     [5ª4 double]
%     []
%     []
%     []
% 
%     
% go to net1.IW{1}
% and then we can call it like
% net1.Iw{row,column}
% and assign the values we want

%%  middle and out layers
% net1.LW
% 
% net1.LW
% 
% ans =
% 
%   4ª4 cell array
% 
%               []               []               []    []
%     [5ª5 double]               []               []    []
%               []    [40ª5 double]               []    []
%               []               []    [3ª40 double]    []

%go to net1.LW{2,1} for example
%and then assign values like this: net1.LW{2,1}(1,1)=3

% for the out layer write net1.LW{4,3}



%% biases (for a 100X100X40 example) network
% net.b
% 
% ans =
% 
%   4ª1 cell array
% 
%     [100ª1 double]
%     [100ª1 double]
%     [ 40ª1 double]
%     [  5ª1 double]

% net.b{4} is the content of [  5ª1 double]

end

