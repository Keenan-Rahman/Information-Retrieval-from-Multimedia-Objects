import numpy as np

"""
Function for calculating SVD of the given matrix.
@param: An n x m matrix. The matrix's rows are the data objects (n), and the columns the features (m)
@returns: A list containing the a matrix of the eigenvectors of D*Dt (n x k)
          Diagonal matrix of the eigenvalues in descending order (k x k),
          and Eigenvectors of Dt*D (k x m).
"""


def svd(D, k):
    # test_U, test_s, test_VT = scipy.linalg.svd(D)

    # get U
    T = np.dot(D, np.transpose(D))
    # np.array(T, dtype=float)
    t_eigval, U = np.linalg.eig(np.array(T, dtype=float))

    # get V
    W = np.dot(np.transpose(D), D)
    w_eigval, V = np.linalg.eig(np.array(W, dtype=float))

    # take shorter list of eigenvalues for s
    used_t = False
    if len(t_eigval) <= len(w_eigval):
        s = np.sqrt(t_eigval)
        used_t = True
    else:
        s = np.sqrt(w_eigval)

    # sort eigenvalues
    sorted_eigvals = s[s.argsort()[::-1]]
    # create S
    S = np.diag(sorted_eigvals)

    # sort V's rows, Us columns according to eigenvalues in s
    VT = V[s.argsort()[::-1], :]
    U = U[:, s.argsort()[::-1]]

    # transpose for projection calculation
    if not used_t:
        VT = np.transpose(VT)

    # get k columns in U, k rows and columns in S, k rows in VT
    U_k = U[:, 0:k]
    S_k = S[0:k, 0:k]
    VT_k = VT[0:k, :]

    # apply dimensionality reduction projection using k latent features
    projection = np.dot(np.dot(U_k, S_k), VT_k)

    return [np.float32(U_k.real), np.float32(VT_k.real), projection, S_k]
    # return the projection, left factor matrix U, Core matrix S, and right factor matrix VT with respect to the k
    # latent feature semantics
