import pickle
import numpy as np
import copy
def left_shift(l):
    return l[1:]+l[:1]

def get_macro_data():
    """ need validation data"""
    str_dict = {}
    with open("Data/str_acc","rb") as f:
        str_dict = pickle.load(f) 

    for i in range(len(str_dict['structure'])):
        struct = str_dict['structure'][i]
        for j in range(len(struct)):
            str_dict['structure'][i][j] = left_shift(str_dict['structure'][i][j])

    """ adjency matrix , property matrix"""
     # print(str_dict['structure'][0])
    nums = len(str_dict['structure'])
    # print(nums)
    A = np.zeros((nums,12,12))
    P = np.zeros((nums,12,12))
    for i in range(nums):
        for j in range(12):
            P[i][j][j] = str_dict['structure'][i][j][-1]
            for k in range(j):
                A[i][j][k]=str_dict['structure'][i][j][k]
            if j != 11:
                A[i][j+1][j]= 1
    for i in range(nums):
        for j in range(12):
            if j != 11:
                A[i][j+1][j]= 1

    str_dict["P"] = P
    Adj = A+np.transpose(A,axes=[0,2,1])
    str_dict["Adj"] = Adj

    """ degreee matrix"""
    D = np.sum(Adj,axis=1)
    AD = np.zeros((nums,12,12))
    for i in range(nums):
        for j in range(12):
            AD[i][j][j] = D[i][j]
    str_dict["D"] = AD
    
    # print(str_dict["D"][0])
    # print(str_dict["ADAt"][0])


    """ normalized graph Laplacian"""
    
    datashape = np.shape(Adj)
    D = copy.deepcopy(AD)
    for i in range(len(D)):
        for j in range(len(D[0])):
            D[i][j][j] = 1/np.sqrt(D[i][j][j])

    L = []
    for i in range(len(Adj)):
        laplacian = np.eye(datashape[1],datashape[2]) - np.matmul(np.matmul(D[i],Adj[i]),D[i])
        eigenvalue = np.linalg.eigvals(laplacian)
        cheby_L = 2/max(eigenvalue) * laplacian - np.eye(datashape[1],datashape[2])
        L.append(cheby_L)
    # print(np.shape(str_dict['structure']))
    # print(np.shape(str_dict['accuracy']))
    Laplacian = {"L":L}

    with open("Data/Laplacian","wb") as f:
        pickle.dump(Laplacian,f) 

    return str_dict
if __name__ == "__main__":
    get_macro_data()