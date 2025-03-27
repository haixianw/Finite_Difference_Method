function v = vector_to_matrix_final(u,N_x,N_y)
% change the vector to a matrix
% Example [a11, ..., a1N_y, a21, ..., a2N, ..., aN_xN_y]
%to be a matrix         [a11, a12, a13, ..., a1N_y]
%                       [a21, a22, a23, ..., a2N_y]
%                       [...  ...   ...   ...   ]
%                       [aN_x1, aN_x2, aN_x3, ..., aN_xN_y]

% New variable
v = zeros(N_x,N_y);

% Record
j = 0;

% Loop
for i = 1 : N_x
    
    v(i, :) = u(j + 1: j + N_y);
    j = N_y * i;
    
end
end