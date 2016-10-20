
% This code is for doing segmentation of images using Gaussian Mixture Models and 
% Expectation - Maximization algorithm

% In this, I am doing a three class segmentation
clc;
clear all;
close all;

% For loading the file i.e image 
[fname,path]=uigetfile('*.jpg')
fname=strcat(path,fname);
% Reading an image and converting in to double i.e pixel intensity value lies between 0 and 1
im=imread(fname);
im=im2double(im);

im_1=im(:,:,1)/255;   %r component
im_2=im(:,:,2)/255;   %g component
im_3=im(:,:,3)/255;   %b component

% Initializing means of the three classes
m1=[120;120;120]/255;
m2=[12;12;12]/255;
m3=[180;180;180]/255;

% Initializing co-variances for three classes
COV1=eye(3);
COV2=eye(3);
COV3=eye(3);

%Initializing to store rgb values and responsibilities of every pixel
im1=zeros(3,154401);
gama=zeros(3,154401);

% calling a function to calculate probability
z=@normaldist;
k=1;
for i=1:321
    for j=1:481       
        im1(:,k)=[im_1(i,j);im_2(i,j);im_3(i,j)];
        k = k+1;
    end
end
% Initialization of counter for iterations
counter = 0;
% Initializing priors
N(1)=1/3;
N(2)=1/3;
N(3)=1/3;


while(counter <60)
    counter = counter +1;
%responsibility calculations
C = zeros(3,321*481);
for k=1:154401

a=z(im1(:,k),m1,COV1)*N(1);
b=z(im1(:,k),m2,COV2)*N(2);
c=z(im1(:,k),m3,COV3)*N(3);
s = a+b+c;
gama(:,k)=[a/s;b/s;c/s];
[ll,ind] = max(gama(:,k));
 if(ind == 1)
    C(1,k) = 1;
 elseif(ind == 2)
    C(2,k) = 1;
 else
    C(3,k) = 1;
 end
end
t = 154401;
%Sum of responsibilites
N1 = sum(gama(1,:));
N2 = sum(gama(2,:));
N3 = sum(gama(3,:));
%Prior Calculation
N = [N1 N2 N3]/t ;
fav1=zeros(3,1);
fav2=zeros(3,1);
fav3=zeros(3,1);
for i=1:154401
fav1 = fav1 +im1(:,i)*gama(1,i);
fav2 = fav2 +im1(:,i)*gama(2,i);
fav3 = fav3 +im1(:,i)*gama(3,i);
end
%Mean calculation
m1 = fav1/N1
m2 = fav2/N2
m3 = fav3/N3
cov1=zeros(3,3);
cov2=zeros(3,3);
cov3=zeros(3,3);
for i=1:154401
 cov1=cov1+(im1(:,i)-m1)*(im1(:,i)-m1)'*gama(1,i);
 cov2=cov2+(im1(:,i)-m2)*(im1(:,i)-m2)'*gama(2,i);
 cov3=cov3+(im1(:,i)-m3)*(im1(:,i)-m3)'*gama(3,i);
end
%Covariance calculation
COV1=cov1/N1;
COV2=cov2/N2;
COV3=cov3/N3;
%loglikelihood estimation
L1=0;

for i=1:154401
    L1=log((N1/t)*z(im1(:,i),m1,COV1)+(N2/t)*z(im1(:,i),m2,COV2)+(N1/t)*z(im1(:,i),m3,COV3))+L1;
end
end
%Displaying the segmented output
f = zeros(321,481,3);
f(:,:,1) = reshape(C(1,:),321,481);
f(:,:,2) = reshape(C(2,:),321,481);
f(:,:,3) = reshape(C(3,:),321,481);
figure
image(f)
