import numpy as np

def index_mapping(in_char):
    in_ascii = ord(in_char)

    if in_ascii >= ord('0') and in_ascii <= ord('9'):
        return in_ascii-ord('0')
    elif in_ascii >= ord('A') and in_ascii <= ord('Z'):
        return in_ascii-ord('A') + 10
    elif in_ascii >= ord('a') and in_ascii <= ord('z'):
        return in_ascii-ord('a') + 10 + 26
    elif in_ascii == ord('&'):
        return 62
    elif in_ascii == ord('('):
        return 63
    elif in_ascii == ord(')'):
        return 64
    else:
        return -1


def calc_tp_table(filename):
    print ('counting the words in ' + filename)
    fin = open(filename, 'r')
    for line in fin.readlines():
        c_list = list(line)
        for i in range(0, len(c_list)-2):
            c_idx = index_mapping(c_list[i])
            c_transition_idx = index_mapping(c_list[i+1])
            if c_idx != -1 and c_transition_idx != -1:
                tp_table[c_idx][c_transition_idx] += 1


def save_tp_table(filename):
    fout = open(filename, 'w')
    # Normalize the table, the sum of each row should be 1.0
    for row in tp_table:
        sum = 0
        for entry in row:
            sum += entry

        if sum > 0:
            row /= sum
    
    # rearrange the table
    for i in range(10):
        for j in range(10):
            tp_table[i][j] = 1.0/10
        for j in range(10, CHAR_NUM):
            tp_table[i][j] = 1.0/26

    for i in range(10, 36):
        for j in range(10):
            tp_table[i][j] = 1.0 / 26
        for j in range(10, 36):
            tp_table[i][j] = tp_table[i+26][j+26]
        for j in range(36, 62):
            tp_table[i][j] = tp_table[i+26][j]
        for j in range(62, CHAR_NUM):
            tp_table[i][j] = 1.0 / 26
    
    for i in range(36, 62):
        for j in range(10):
            tp_table[i][j] = 1.0 / 26
        for j in range(10, 36):
            tp_table[i][j] = tp_table[i][j+26]
        for j in range(62, CHAR_NUM):
            tp_table[i][j] = 1.0 / 26

    for i in range(62, CHAR_NUM):
        tp_table[i] = np.full((1, CHAR_NUM), 1.0/26)
        
    for row in tp_table:
        for entry in row:
            fout.write(str(entry)+' ')
        fout.write('\n')




CHAR_NUM = 65
tp_table = np.zeros((CHAR_NUM, CHAR_NUM))

calc_tp_table('dictionary/self_define_word.txt')
calc_tp_table('dictionary/words.txt')
calc_tp_table('dictionary/dictionary.txt')
save_tp_table('tp.txt')

print (tp_table)