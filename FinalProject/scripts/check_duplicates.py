with open('difficult_pairs.txt','r') as f:
    corpus = f.readlines()
    right_side_dict = {}
    to_delete = []
    for line in corpus:
        spl = line.split('\t')
        right_side = spl[1].rstrip()
        if right_side not in right_side_dict:
            right_side_dict[right_side] = 1
        else:
            right_side_dict[right_side] += 1
            to_delete.append(line)

'''
for k,v in right_side_dict.items():
    if v > 1:
        print(k,':',v)
'''
with open('new_difficult_pairs.txt','w') as of:
    for line in corpus:
        if line not in to_delete:
            of.write(line)
