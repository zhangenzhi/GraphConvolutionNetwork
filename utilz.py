def get_macro_data():

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
    str_dict["P"] = P
    str_dict["A"] = A

    """diagnol property + adjency matrix"""
    str_dict["APAt"] = A+P+np.transpose(A,axes=[0,2,1])


    """ degreee matrix + adjency"""
    D = np.sum(A,axis=1)
    AD = np.zeros((nums,12,12))
    for i in range(nums):
        for j in range(12):
            AD[i][j][j] = D[i][j]
    str_dict["D"] = AD
    str_dict["AD"]= AD+A
    str_dict["ADAt"] = AD+np.transpose(A,axes=[0,2,1])+A
    # print(str_dict["D"][0])
    # print(str_dict["ADAt"][0])

    """degree matrix + direct graph"""
    str_dict["AP"] = A+P

    return str_dict
    # print(np.shape(str_dict['structure']))
    # print(np.shape(str_dict['accuracy']))

if __name__ == "__main__":
    get_macro_data()