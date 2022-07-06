%getaverage(24,24,1,8,8)

%test update of Z
% Dw = rand(8,8); Dh = rand(8,8); Ds = rand(31,5);
% D = combine(Dw, Dh, Ds);
% L = randn(8,8,5);
% C = randn(8,8,5);
% Y = randn(8,8,31);
% mu = 1;
% Z = updateZ(D, L, C, Y, mu);
% firstprod = tensorprod(Z,Dw, 1);
% secondprod = tensorprod(firstprod, Dh, 2);
% thirdprod = tensorprod(secondprod,Ds,3);
% loss = (norm(script1(Y-thirdprod,1), 'fro'))^2 + L(:)'*(C(:)-Z(:)) + mu/2*(norm(script1(C-Z,1), 'fro'))^2
% 
% %take some random Z values around the best one and see if any of them are
% %better than the above loss 
% values = zeros(0,1);
% for i = 1:100
%     Znew = Z + 0.2*randn(size(Z));
%     firstprod = tensorprod(Znew,Dw, 1);
%     secondprod = tensorprod(firstprod, Dh, 2);
%     thirdprod = tensorprod(secondprod,Ds,3);
%     loss = (norm(script1(Y-thirdprod,1), 'fro'))^2 + L(:)'*(C(:)-Znew(:)) + mu/2*(norm(script1(C-Znew,1), 'fro'))^2;
%     values = [values; loss];
% end
%test is passed 4/16 6:07pm

%test update of C
% Dw = rand(8,8); Dh = rand(8,8); Ds = rand(31,5);
% D = combine(Dw, Dh, Ds);
% L = randn(8,8,5);
% Y = randn(8,8,31);
% Z = randn(8,8,5);
% mu = 5; lambda = 1;
% C = updateC(lambda, mu, Z, L);
% firstprod = tensorprod(Z,Dw, 1);
% secondprod = tensorprod(firstprod, Dh, 2);
% thirdprod = tensorprod(secondprod,Ds,3);
% loss = lambda*l0norm(C) + mu/2*(norm(script1(C-Z,1), 'fro'))^2
% 
% values = zeros(0,1);
% for i = 1:100
%     Cnew = Z + 0.2*randn(size(C));
%     firstprod = tensorprod(Z,Dw, 1);
%     secondprod = tensorprod(firstprod, Dh, 2);
%     thirdprod = tensorprod(secondprod,Ds,3);
%     loss = lambda*l0norm(Cnew) + mu/2*(norm(script1(Cnew-Z,1), 'fro'))^2;   
%     values = [values; loss];
% end
%OK 4/17 10:22am

% Dw = rand(8,8); Dh = rand(8,8); Ds = rand(31,5);
% D = combine(Dw, Dh, Ds);
% L = randn(961,8,8,5);
% Y = randn(961,8,8,31);
% Z = randn(961,8,8,5);
% C = randn(961,8,8,5);
% Dnew = updateD(Z, Dh, Ds, Y, Dw, 1);
% mysum = 0;
% for k = 1:961
%     Zk = squeeze(Z(k,:,:,:));
%     Yk = squeeze(Y(k,:,:,:));
%     first = tensorprod(Zk, Dh, 2);
%     second = tensorprod(first, Ds, 3);
%     mysum = mysum + (norm(script1(Yk,1)-Dnew*script1(second, 1), 'fro'))^2;
% end
% mysum
% 
% for i = 1:20
%     mysum = 0;
%     Dw = Dnew + 0.2*randn(size(Dnew));
%     for k = 1:961
%         Zk = squeeze(Z(k,:,:,:));
%         Yk = squeeze(Y(k,:,:,:));
%         first = tensorprod(Zk, Dh, 2);
%         second = tensorprod(first, Ds, 3);
%         mysum = mysum + (norm(script1(Yk,1)-Dw*script1(second, 1), 'fro'))^2;
%     end
%     mysum
% end
%seems to pass the test, 4/16 11:59pm

% Dw = rand(8,8); Dh = rand(8,8); Ds = rand(31,5);
% D = combine(Dw, Dh, Ds);
% L = randn(961,8,8,5);
% Y = randn(961,8,8,31);
% Z = randn(961,8,8,5);
% C = randn(961,8,8,5);
% Dnew = updateD(Z, Dh, Dw, Y, Ds, 3);
% mysum = 0;
% for k = 1:961
%     Zk = squeeze(Z(k,:,:,:));
%     Yk = squeeze(Y(k,:,:,:));
%     first = tensorprod(Zk, Dh, 1);
%     second = tensorprod(first, Dw, 2);
%     mysum = mysum + (norm(script1(Yk,3)-Dnew*script1(second, 3), 'fro'))^2;
% end
% mysum
% 
% for i = 1:20
%     mysum = 0;
%     Ds = Dnew + 0.2*randn(size(Dnew));
%     for k = 1:961
%         Zk = squeeze(Z(k,:,:,:));
%         Yk = squeeze(Y(k,:,:,:));
%         first = tensorprod(Zk, Dh, 1);
%         second = tensorprod(first, Dw, 2);
%         mysum = mysum + (norm(script1(Yk,3)-Ds*script1(second, 3), 'fro'))^2;
%     end
%     mysum
% end
%seems to pass the test 4/17 12:03am


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
                if M(i,j,k) > 1e-6
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
        A = kron(script1(fullprod, m), eye(idsize)); 
        sum1 = sum1 + A*A';
        asdf = script1(Yk,m);
        sum2 = sum2 + A*asdf(:); 
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