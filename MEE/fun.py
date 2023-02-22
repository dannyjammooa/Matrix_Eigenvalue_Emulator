import numpy as np
from numpy import linalg as la
import concurrent.futures
import heapq
import schedule
from scipy.linalg import expm


def poly_fit(dt1,dt2,E_true,pdeg):
    res = np.polyfit(dt1,E_true,deg=pdeg)
    pe = np.poly1d(res)
    poly = []
    for i in dt2:
        poly.append(pe(i))
    return np.array(poly)

#function used after parrallel computation to pick best coeff based off of min(cost)
def cleaning(results):
    results_array = np.array(results)
    mm_coeff = results_array[:, 0]
    mm_l2 = results_array[:, 1]
    min_l2 = heapq.nsmallest(1, mm_l2)[0]
    MM_coeff = mm_coeff[mm_l2 == min_l2]
    MM_l2 = mm_l2[mm_l2 == min_l2]
    return MM_coeff[0],MM_l2[0]

#function that sorts eigenvalues from smallest to largest
def eigen(A):
    eigenvalues, eigenvectors = la.eig(A)
    sorted_indices = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]
    return (eigenvalues, eigenvectors)

#creates a diagonal matrix A and symmetric matrix B
def sym_matrix(n, coeff):
    vec1 = coeff[0:n]
    I = np.eye(n)
    A = vec1.T*I
    vec2 = coeff[n:]
    B = np.zeros((n, n))
    upper_indices = np.triu_indices(n)
    lower_indices = np.tril_indices(n)
    B[upper_indices] = vec2[:n*(n+1)//2]
    B[lower_indices] = B.T[lower_indices]
    return A, B
def symT_matrix(n,coeff):
    A = np.array([[coeff[0],0],[0,-coeff[0]]])
    B = np.array([[coeff[1],coeff[2]],[coeff[2],coeff[3]]])
    return A,B

#creats nxn A,B complex hermitian matrix
def herm_matrix(n,coeff):
    #creating matrix A
    vec1 = coeff[0:n]
    I    = np.eye(n)
    A    = vec1.T*I

    #creating matrix B
    vec2         = coeff[n:]
    matrix       = vec2.reshape(n,n)           #fill matrix with coeff
    D            = matrix.diagonal()*np.eye(n) #store only the diagonal (D) elements
    U            = np.triu(matrix,k=1)         #store only the upper (U) triangular elements
    L            = np.triu(matrix.T,k=1)       #store only the lower (L) triangular elements
    B_U          = U + 1j*L                    #Using the U and L elements to make B_U
    B_L          = np.matrix(B_U).getH()       #using B_U to find B_L
    B            = B_U + B_L + D               #summing all elements to get hermitian matrix B
    return A,B

def Emin_loop(dt,A,B): 
    # Initialize empty list for minimum eigenvalues
    minimum_eigenvalues = []
    eigenvectors = []
    # Loop through time steps
    for t in dt:
        # Trotterize A and B
        trotterized_matrix = expm(-1j*A*t)@expm(-1j*B*t)
        # Compute eigenvalues
        w,v = eigen(trotterized_matrix)
        # Compute minimum eigenvalue
        log_eigenvalues = np.log(w) / (-1j*t)
        # Compute minimum eigenvalue
        min_eigenvalue = min(log_eigenvalues.real)
        # Add minimum eigenvalue to list
        minimum_eigenvalues.append(min_eigenvalue)
        psi = 0
        for i in range(len(log_eigenvalues)):
            if log_eigenvalues[i].real == min_eigenvalue.real:
                psi =v[:,i]
        eigenvectors.append(psi)
    # Return array of minimum eigenvalues
    return np.array(minimum_eigenvalues),np.array(eigenvectors)

def Emin(dt,A,B): 
    # Compute Trotterized matrix
    t = dt[:, np.newaxis, np.newaxis]
    trotterized_matrix = expm(-1j*A*t) @ expm(-1j*B*t)
    # Compute eigenvalues
    w, v = la.eig(trotterized_matrix)
    # Compute minimum eigenvalue
    log_eigenvalues = np.log(w) / (-1j*dt[:, np.newaxis])
    min_eigenvalues = log_eigenvalues.real.min(axis=1)
    # Compute eigenvectors corresponding to minimum eigenvalue
    min_indices = np.argmin(log_eigenvalues.real, axis=1)
    eigenvectors = v[np.arange(len(dt)), :, min_indices]
    # Return array of minimum eigenvalues and eigenvectors
    return min_eigenvalues, eigenvectors

def matrix(n,coeff,matrix_type):
    matrix_funcs = {
        'symT': symT_matrix,
        'sym': sym_matrix,
        'herm': herm_matrix}
    # Get the matrix function for the specified matrix type
    matrix_func = matrix_funcs.get(matrix_type)
    # Compute the matrices A and B using the specified matrix function
    A, B = matrix_func(n, coeff)
    return A,B

def fun(dt, n, coeff,matrix_type):
    A, B = matrix(n,coeff,matrix_type)
    return Emin(dt, A, B)

def BE_mat(n):
    result = []
    for i in range(n):
        for j in range(i, n):
            mat = np.zeros((n,n))
            mat[i,j] = 1
            if i == j or (i < j and np.sum(mat[:i,j]) == 0):
                result.append(mat)
    return np.array(result)

def AE_mat(n):
    result = []
    for i in range(n):
        matrix = np.zeros((n, n), dtype=int)
        matrix[i, i] = 1
        result.append(matrix)
    return np.array(result)

def Amat(A, B, epsilon, n, dt, Emat):
    expmB = expm(-1j*B*dt)
    E = Emat * epsilon
    M_list = [np.einsum('ij,jk->ik', expm(-1j*(A + E[i])*dt), expmB) for i in range(len(E))]
    return np.array(M_list)

def Bmat(B, A, epsilon, n, dt, Emat):
    expmA = expm(-1j*A*dt)
    E = Emat * epsilon
    M_list = [np.einsum('ij,jk->ik', expmA, expm(-1j*(B + E[i])*dt)) for i in range(len(E))]
    return np.array(M_list)

def up_tri(B,n):
    out = np.zeros((n, n))
    inds = np.triu_indices(len(out))
    out[inds] = B
    return out

def numerical_derivative(coeff,n,dt,psi,matrix_type,AE,BE, epsilon=1e-5):
    A,B = matrix(n,coeff,matrix_type)
    MA_grad = []
    MB_grad = []
    BEmat = BE
    AEmat = AE
    for t in range(len(dt)):
        M = expm(-1j*A*dt[t])@expm(-1j*B*dt[t])
        MB_copies = np.array([M]*int((n*(n+1))/2))
        MA_copies = np.array([M]*n)
        PSI = psi[t].reshape(n,1)
        MA_new = Amat(A,B,epsilon,n,dt[t],AEmat)
        MB_new = Bmat(B,A,epsilon,n,dt[t],BEmat)
        dM_dA = (MA_new-MA_copies)/epsilon
        dM_dB = (MB_new-MB_copies)/epsilon
        HF_A = np.einsum('ijk,jk,ik->i', dM_dA, PSI, np.conj(PSI))
        HF_B = np.einsum('ijk,jk,kj->i', dM_dB, PSI, np.conj(PSI))
        MA_grad.append(np.diag(HF_A.real))
        MB_grad.append(up_tri(HF_B.real,n))
    return MA_grad, MB_grad

#calculates the gradient of M = A+Bdt with respect to A and B
#want to use autograd
def numerical_derivative2(coeff,n,dt,psi,matrix_type, epsilon=1e-5):
    A,B = matrix(n, coeff,matrix_type)
    MA_grad = np.zeros((len(dt),n,n))
    MB_grad = np.zeros((len(dt),n,n))
    for t in range(len(dt)):
        M = expm(-1j*A*dt[t])@expm(-1j*B*dt[t])
        PSI = psi[t].reshape(n,1)
        for i in range(n):
            for j in range(i+1):
                if i == j:  # A is diagonal
                    A_new = A.copy()
                    A_new[i,j] += epsilon
                    MA_new = expm(-1j*A_new*dt[t])@expm(-1j*B*dt[t])
                    #hellmann-feynman theorem
                    dM_dA = PSI.conj().T@((MA_new-M)/epsilon)@PSI
                    MA_grad[t,i,j] = dM_dA[0][0]
                    
                    B_new = B.copy()
                    B_new[i,j] += epsilon
                    MB_new = expm(-1j*A*dt[t])@expm(-1j*B_new*dt[t])
                    dM_dB = PSI.conj().T@((MB_new-M)/epsilon)@PSI
                    MB_grad[t,i,j] = dM_dB[0][0]
                else:  # B is symmetric
                    B_new = B.copy()
                    B_new[i,j] += epsilon
                    MB_new = expm(-1j*A*dt[t])@expm(-1j*B_new*dt[t])
                    #hellmann-feynman theorem
                    dM_dB = PSI.conj().T@((MB_new-M)/epsilon)@PSI
                    MB_grad[t,i,j] = dM_dB[0][0]
                    MB_grad[t,j,i] = MB_grad[t,i,j]
    return MA_grad, MB_grad

def cost_func(coeff,y_true,dt,n,matrix_type):
    y_pred,psi = fun(dt,n,coeff,matrix_type)
    return np.sum((y_pred-y_true)**2)

def grad_cost_func(coeff,y_true,dt,n,matrix_type,AE,BE):
    y_pred,psi = fun(dt,n,coeff,matrix_type)
    dU_dA,dU_dB = numerical_derivative(coeff,n,dt,psi,matrix_type,AE,BE)
    A_grad = 2*np.sum((y_pred-y_true)[:,np.newaxis,np.newaxis]*dU_dA, axis=0)
    B_grad = 2*np.sum((y_pred-y_true)[:,np.newaxis,np.newaxis]*dU_dB, axis=0)
    return A_grad,B_grad

#this function makes sure our lerning_rage*grad has A to be diagonal and B to be symmetric
def update(A_grad,B_grad,optimizer_scheduler,n):
    A_change = optimizer_scheduler.update_change(A_grad)
    B_change = optimizer_scheduler.update_change(B_grad)
    A_change = np.diag(A_change) 
    lower_indices = np.tril_indices(n)
    B_change = np.array(B_change[lower_indices])
    change = np.concatenate((A_change.flatten(),B_change.flatten()),axis=0)
    return change