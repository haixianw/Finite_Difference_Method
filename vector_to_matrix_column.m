function v = vector_to_matrix_column(u)
% change the vector to a matrix
% Example [a11, ..., aN1, a12, ..., aN2, ..., aNN]
%to be a matrix         [a11, a12, a13, ..., a1N]
%                       [a21, a22, a23, ..., a2N]
%                       [...  ...   ...   ...   ]
%                       [aN1, aN2, aN3, ..., aNN]

% Size of the input
N = sqrt(length(u));

% New variable
v = zeros(N);

% Record
j = 0;

% Loop
for i = 1 : N
    
    v(:,i) = u(j + 1: j + N);
    j = N * i;
    
end
end