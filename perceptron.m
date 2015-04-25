%perceptron
%Ejemplo de funcionamiento
%input=[0 0; 1 0; 0 1; 1 1];
%t=[0 0 0 1]' % para el caso de AND
%w=perceptron(input,t)
function w=perceptron(input, t)
%----------------------------------
input=[ones(size(input,1),1), input];
%---------------------------------- W_0 random
w=rand(1,size(input,2))'*2-1;
tError=1;
steps=0;
lr=0.2;
while(tError~=0)
    y=input*w;
    y(y<=0)=0;
    y(y>0)=1;
    error=t-y;
    tError=sum(abs(error))
    for i=1:size(input,1)
        %-------------------------- Updating weights
        w=w+lr*input(i,:)'.*error(i);
    end
    steps=steps+1
end
end
