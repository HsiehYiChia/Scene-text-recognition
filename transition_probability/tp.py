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

        for entry in row:
            fout.write(str(entry)+' ')
        fout.write('\n')




CHAR_NUM = 65
tp_table = np.zeros((CHAR_NUM, CHAR_NUM))

calc_tp_table('dictionary/self_define_word.txt')
calc_tp_table('dictionary/words.txt')
calc_tp_table('dictionary/passwords.txt')
calc_tp_table('dictionary/dictionary.txt')
calc_tp_table('dictionary/10_million_password_list_top_1000000.txt')
save_tp_table('dictionary/tp.txt')

print (tp_table)