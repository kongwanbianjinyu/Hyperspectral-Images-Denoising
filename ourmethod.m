load pompoms.mat
sigma = 0.3;
noisy_msi = msi+sigma*randn(size(msi));
%imshow(noisy_msi(:,:,10));
%[psnr, ssim, fsim, ergas, msam] = MSIQA(255*noisy_msi, 255*msi)
Hs = noisy_msi;

Hsize = size(Hs); W = Hsize(1); H = Hsize(2); S = Hsize(3);
  dw = 8; dh = 8;
rownum = (W-dw)/4+1;
colnum = (H-dh)/4+1;
 n = rownum*colnum;

mu = 3; lambda = 1;
%initialize the variables
params.peak_value = 1;
params.nsigma = sigma;
params.spect_dim = 3;
params.dic_num = 1;
[clean_img, basis, idx, temp, info, CDIC] = TensorDL(Hs, params);
Dw = CDIC.U1; %fix this later: step 1
Dh = CDIC.U2;
Ds = CDIC.U3; 
[a,mw] = size(Dw); [b,mh] = size(Dh); [c,ms] = size(Ds);
D = combine(Dw, Dh, Ds);

Y = ones(n, dw, dh, S);
Z = ones(n, mw, mh, ms); C = Z; L = Z; 

k = 1;
for row = 1:4:W-dw+1
    for col = 1:4:H-dh+1
        Y(k,:,:,:) = Hs(row:row+dw-1, col:col+dh-1, :); %kth patch of the HSI
        Yk = squeeze(Y(k,:,:,:));
        vecZ = Yk(:)\D; %step 2
        Z(k,:,:,:) = reshape(vecZ, mw, mh, ms);
        Zk = squeeze(Z(k,:,:,:));
        C(k,:,:,:) = Zk; %step 3
        L(k,:,:,:) = ones(size(Zk)); %step 4
        k = k + 1;
    end
    row
end

converge = 0;
lastobj = 10000000;
iterations = 0;
%ADMM to iteratively optimize each variable
while converge == 0
    iterations = iterations + 1
    for k = 1:n
        Yk = squeeze(Y(k,:,:,:));
        Zk = squeeze(Z(k,:,:,:));
        Ck = squeeze(C(k,:,:,:));
        Lk = squeeze(L(k,:,:,:));

        Zk = updateZ(D, Lk, Ck, Yk, mu); %step 1
        Ck = updateC(lambda, mu, Zk, Lk); %step 2
        Lk = updateL(Lk, Ck, Zk, mu); %step 4

        Z(k,:,:,:) = Zk; C(k,:,:,:) = Ck; L(k,:,:,:) = Lk;
        if round(k/1000) == k/1000
            k
        end
    end
         %Dw = updateD(Z, Dh, Ds, Y, Dw, 1);
         %Dh = updateD(Z, Dw, Ds, Y, Dh, 2);
         Ds = updateD(Z, Dh, Dw, Y, Ds, 3);
         D = combine(Dw, Dh, Ds);

    %evaluate the objective function
    mysum = 0;
    for k = 1:n
        Yk = squeeze(Y(k,:,:,:));
        Zk = squeeze(Z(k,:,:,:));
        Ck = squeeze(C(k,:,:,:));
        Lk = squeeze(L(k,:,:,:));
        firstprod = tensorprod(Zk,Dw, 1);
        secondprod = tensorprod(firstprod, Dh, 2);
        thirdprod = tensorprod(secondprod,Ds,3);
        %(norm(script1(Yk-thirdprod,1), 'fro'))^2
        %lambda*l0norm(Ck)
        %Lk(:)'*(Ck(:)-Zk(:))
        %mu/2*(norm(script1(Ck-Zk,1), 'fro'))^2
        mysum = mysum + (norm(script1(Yk-thirdprod,1), 'fro'))^2 + lambda*l0norm(Ck) + Lk(:)'*(Ck(:)-Zk(:)) + mu/2*(norm(script1(Ck-Zk,1), 'fro'))^2;
    end
    if abs((mysum-lastobj)/lastobj) < 1e-3
        converge = 1;
    end
    lastobj = mysum
end

%average over Z times D to get the denoised image
finalimage = zeros(W,H,S);
k = 1;
for row = 1:4:W-dw+1
    for col = 1:4:H-dh+1
        Zk = squeeze(Z(k,:,:,:));
        first = tensorprod(Zk, Dw, 1);
        second = tensorprod(first, Dh, 2);
        third = tensorprod(second, Ds, 3);
        finalimage(row:row+dw-1, col:col+dh-1,:) = finalimage(row:row+dw-1, col:col+dh-1,:) + third;
        k = k + 1;
    end
end
finalimage = finalimage./getaverage(W,H,S,dw,dh);
imshow(finalimage(:,:,10));
[psnr, ssim, fsim, ergas, msam] = MSIQA(255*finalimage, 255*msi)

function answer = updateC(lambda, mu, Z, L)
    answer = zeros(size(Z));
    [a,b,c] = size(Z);
    for i = 1:a
        for j = 1:b
            for k = 1:c
                answer(i,j,k) = hardthresh(lambda/mu, Z(i,j,k)-L(i,j,k)/mu);
            end
        end
    end
end

function x = l0norm(M)
    [a,b,c] = size(M);
    x = 0;
    for i = 1:a
        for j = 1:b
            for k = 1:c
                if M(i,j,k) > 1e-7
                    x = x+1;
                end
            end
        end
    end
end

function answer = getaverage(W, H, S, dw, dh)
    %computes the pointwise averaging matrix
    answer = zeros(W,H,S);
    for row = 1:4:W-dw+1
        for col = 1:4:H-dh+1
            answer(row:row+dw-1, col:col+dh-1, :) = answer(row:row+dw-1, col:col+dh-1, :) + ones(dw,dh,S);
        end
    end
end

function Z = updateZ(D, L, C, Y, mu)
    vecZ = inv(D'*D+mu*eye(size(D'*D)))*(L(:) + D'*Y(:) + mu*C(:));
    Z = reshape(vecZ, size(C));
end

function answer = hardthresh(const, mat)
    if abs(mat) < sqrt(2*const)
        answer = 0;
    elseif abs(mat) > sqrt(2*const)
        answer = mat;
    else
        answer = 0;
    end
end

function answer = tensorprod(A, B, i)
    answer = nmodeproduct(A,B,i);
end

function Dnew = updateD(Z, firstD, secD, Y, oldD, m)
    %Z is the array of tensors Z(k), Y is array of tensors Y(k)
    [n, ~, ~, ~] = size(Z); 
    [u,v] = size(oldD); 
    sum1 = zeros(u*v, u*v);
    sum2 = zeros(u*v, 1);
    for k = 1:n
        Zk = squeeze(Z(k,:,:,:));
        Yk = squeeze(Y(k,:,:,:));
        firstthing = 0; secondthing = 0;
        if m == 1
            firstthing = 2 ;secondthing = 3;
        elseif m == 2
            firstthing = 1; secondthing = 3;
        else
            firstthing = 1; secondthing = 2;
        end
        firstprod = tensorprod(Zk, firstD, firstthing);
        fullprod = tensorprod(firstprod, secD, secondthing);
        [a,b,c,d] = size(Y);
        if m == 1 || m == 2
            idsize = b;
        else
            idsize = d;
        end
        A = kron((script1(fullprod, m))', eye(idsize)); 
        sum1 = sum1 + A'*A;
        asdf = script1(Yk,m);
        sum2 = sum2 + A'*asdf(:); 
    end
    Dnew = inv(sum1)*sum2;
    Dnew = reshape(Dnew, size(oldD));
end

function D = combine(Dw, Dh, Ds)
    temp = kron(Ds, Dh);
    D = kron(temp, Dw);
end

function L = updateL(Lold, C, Z, mu)
    L = Lold + mu*(C-Z);
end

function answer = script1(X, r)
    %mode 1 unfolding of a tensor X along mode r
    [a,b,c] = size(X);
    if r == 1
        answer = zeros(a,b*c);
        for i = 1:a
            answer(i,:) = X(i,:);
        end
    elseif r == 2
       answer = zeros(b,a*c);
        for i = 1:b
            Z = X(:,i,:);
            answer(i,:) = Z(:);
        end     
    else
        answer = zeros(c,a*b);
        for i = 1:c
            Z = X(:,:,i);
            answer(i,:) = Z(:);
        end
    end
end