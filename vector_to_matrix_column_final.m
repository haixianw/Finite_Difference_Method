function v = vector_to_matrix_column_final(u,N_x,N_y)
% change the vector to a matrix
% Example [a11, ..., aN_x1, a12, ..., aN_x2, ..., aN_xN_y]
%to be a matrix         [a11, a12, a13, ..., a1N_y]
%                       [a21, a22, a23, ..., a2N_y]
%                       [...  ...   ...   ...   ]
%                       [aN_x1, aN_x2, aN_x3, ..., aN_xN_y]

% New variable
v = zeros(N_x,N_y);

% Record
j = 0;

% Loop
for i = 1 : N_y
    
    v(:,i) = u(j + 1: j + N_x);
    j = N_x * i;
    
end
end