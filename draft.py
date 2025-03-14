deg_seq= [5, 5, 3, 2, 2, 2, 1]
deg_seq.sort(reverse=True)

# find the index and values of nonzeros elements in degree sequence
deg_seq_nonzero_dict = {i: deg_seq[i] for i in range(len(deg_seq)) if deg_seq[i] != 0}
deg_seq_nonzero_dict_key  = list(deg_seq_nonzero_dict.keys())
deg_seq_nonzero_dict_value = list(deg_seq_nonzero_dict.values())

# Debug output to see initial conditions
print(f"Initial deg_seq_nonzero_dict_key: {deg_seq_nonzero_dict_key}")
print(f"Initial deg_seq_nonzero_dict_value: {deg_seq_nonzero_dict_value}")

# find the mts tree sequence
for ki in range(len(deg_seq_nonzero_dict_key)-2, 0, -1 ):
    mts = [0]*len(deg_seq)
    sum_dict_value_ki = sum(deg_seq_nonzero_dict_value[:ki])
    print(f"\033[32m the picked node {ki+1} has degree \033[31m {deg_seq_nonzero_dict_value[ki]} \033[0m; the sum degrees of of {ki} nodes is {sum_dict_value_ki} \033[0m")
    for ni in range(len(deg_seq_nonzero_dict_key)-1,ki-1,-1):
        # print(f"the value of n is {ni+1}")
        value_ki=ni+ki-sum_dict_value_ki
        print(f"nodes numbers are {ni+1}; the value of node {ki+1} is {value_ki}")
        # break when the mts exists
        if 0 < value_ki <= deg_seq_nonzero_dict_value[ki]:
            for kki in range(ki):
                mts[kki] = deg_seq_nonzero_dict_value[kki]
            mts[ki] = value_ki
            for kki in range(ki+1, ni+1):
                mts[kki]=1
            print(f"the max tree subsequence is {mts}")
            # update the remaning degree sequence
            deg_seq = [deg_seq[i] - mts[i] for i in range(len(deg_seq))]
            print(f"the remaining sequence is {deg_seq}")
            deg_seq_nonzero_dict = {i: deg_seq[i] for i in range(len(deg_seq)) if deg_seq[i] != 0}
            deg_seq_nonzero_dict_key  = list(deg_seq_nonzero_dict.keys())
            deg_seq_nonzero_dict_value = list(deg_seq_nonzero_dict.values())
            # Debug output after update
            print(f"the remaining sequence dict is {deg_seq_nonzero_dict}")
            print(f"the remaining sequence length is {len(deg_seq_nonzero_dict_key)}")
            print(f"Updated deg_seq_nonzero_dict_key: {deg_seq_nonzero_dict_key}")
            print(f"Updated deg_seq_nonzero_dict_value: {deg_seq_nonzero_dict_value}")
            break

    while a<=1:
        for ki in range(len(deg_seq_nonzero_dict_key) - 2, 0, -1):
            mts = [0] * len(deg_seq)
            sum_dict_value_ki = sum(deg_seq_nonzero_dict_value[:ki])
            print(
                f"\033[32m the picked node {ki + 1} has degree \033[31m {deg_seq_nonzero_dict_value[ki]} \033[0m; the sum degrees of of {ki} nodes is {sum_dict_value_ki} \033[0m")
            for ni in range(len(deg_seq_nonzero_dict_key) - 1, ki - 1, -1):
                # print(f"the value of n is {ni+1}")
                value_ki = ni + ki - sum_dict_value_ki
                print(f"nodes numbers are {ni + 1}; the value of node {ki + 1} is {value_ki}")
                # break when the mts exists
                if 0 < value_ki <= deg_seq_nonzero_dict_value[ki]:
                    for kki in range(ki):
                        mts[kki] = deg_seq_nonzero_dict_value[kki]
                    mts[ki] = value_ki
                    for kki in range(ki + 1, ni + 1):
                        mts[kki] = 1
                    print(f"the max tree subsequence is {mts}")
                    # update the remaning degree sequence
                    deg_seq = [deg_seq[i] - mts[i] for i in range(len(deg_seq))]
                    print(f"the remaining sequence is {deg_seq}")
                    deg_seq_nonzero_dict = {i: deg_seq[i] for i in range(len(deg_seq)) if deg_seq[i] != 0}
                    deg_seq_nonzero_dict_key = list(deg_seq_nonzero_dict.keys())
                    deg_seq_nonzero_dict_value = list(deg_seq_nonzero_dict.values())
                    # Debug output after update
                    print(f"the remaining sequence dict is {deg_seq_nonzero_dict}")
                    print(f"the remaining sequence length is {len(deg_seq_nonzero_dict_key)}")
                    print(f"Updated deg_seq_nonzero_dict_key: {deg_seq_nonzero_dict_key}")
                    print(f"Updated deg_seq_nonzero_dict_value: {deg_seq_nonzero_dict_value}")
                    break



