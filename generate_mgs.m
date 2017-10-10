function mgs = generate_mgs(len)

initWashoutLength = 1000;

tau = 17;
incrementsperUnit = 10; 
genHistoryLength = tau * incrementsperUnit ;
seed = 1.2 * ones(genHistoryLength,1)+ 0.2 * (rand(genHistoryLength,1)-0.5);
oldval = 1.2;
genHistory = seed;
speedup = 1;

mgs = zeros(len,1);

step = 0;
for n = 1: len + initWashoutLength
    for i = 1:incrementsperUnit * speedup
        step = step + 1;
        tauval = genHistory(mod(step,genHistoryLength)+1,1);
        newval = oldval + (0.2 * tauval/(1.0 + tauval^10) - 0.1 * oldval)/incrementsperUnit;
        genHistory(mod(step,genHistoryLength)+1,1) = oldval;
        oldval = newval;
    end
    if n > initWashoutLength
        mgs(n - initWashoutLength,1) = newval;
    end
end

mgs = (tanh(mgs - 1.0))';

end