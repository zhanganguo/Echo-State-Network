function r = genspmatspecrad1(dim,con)
% genspmatspecrad1(dim,con)
% generates a square matrix of dimension dim
% with spectral radius 1, with connectivity con
r = zeros(dim, dim);
for i = 1:dim
    for j = 1:dim
        if rand < con
            r(i,j) = 2*(rand-0.5);
        end
    end
end
% r = randommat2zeroonemat(r);
maxval = max(abs(eig(r)));
r = r/maxval;