function v = maxtrix_to_vector_column(u)
% Combine the initial condition function to be a vector
% Example [a11, a12, a13, ..., a1N]
%         [a21, a22, a23, ..., a2N]
%         [...  ...   ...   ...   ]
%         [aN1, aN2, aN3, ..., aNN]
% to be [a11, ..., aN1, a12, ..., aN2, ..., aNN]

% Size of the input
N = length(u);

% New variable
v = zeros(1, N * N);

% Record
j = 0;

% Loop
for i = 1 : N
    
    v(j + 1: j + N) = u(:,i);
    j = N * i;
    
end

end