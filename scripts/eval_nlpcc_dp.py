#!/usr/bin/python
# coding=utf-8

import sys

def read_data(file):
    try:
        infile = open(file, mode = "r", encoding = "utf-8")
        sens = {}
        heads = []
        rels = []
        sen = []
        head = []
        rel = []
        index = 0
        for line in infile:
            if line != "\n" and line != "\r\n":
                line_list = line.strip().split("\t")
                sen.append(line_list[1])
                head.append(line_list[6])
                rel.append(line_list[7])
            else:
                sens[tuple(sen)] = index
                heads.append(head)
                rels.append(rel)
                sen = []
                head = []
                rel = []
                index+=1
        infile.close()
        return sens,heads,rels
    except IOError:
        print('IOError: please check the filename')


def evaluation(answerfile,testfile):
        answer_sen, answer_heads, answer_rels= read_data(answerfile)
        test_sen, test_heads, test_rels= read_data(testfile)
        if len(answer_sen) != len(test_sen):
             print("gold sentence number does not match predict number.", len(answer_sen),"\t",len(test_sen))
        correct_arc,total_arc, correct_label, total_label, in_answer_file = 0, 0, 0, 0, 0
        for sen in test_sen:
            if sen in answer_sen:
                in_answer_file += 1
                ans_sen_index = answer_sen[sen]
                test_sen_index = test_sen[sen]
                pre_head = test_heads[test_sen_index]
                gold_head = answer_heads[ans_sen_index]
                pre_rel = test_rels[test_sen_index]
                gold_rel = answer_rels[ans_sen_index]
                if len(pre_head) != len(gold_head):
                    print("head gold length does not match predict length.")
                    continue
                if len(pre_rel) != len(gold_rel):
                    print("label gold length does not match predict length.")
                    continue
                for i in range(len(gold_head)):
                    if gold_head[i] == "-1":
                        continue
                    total_arc += 1
                    total_label += 1
                    if gold_head[i] == pre_head[i]:
                        correct_arc += 1
                        if gold_rel[i] == pre_rel[i]:
                            correct_label += 1
            else:
                print("sentence does not occur in the gold dataset", sen)
        print("There are ", in_answer_file, "sentences in total." )
        uas = correct_arc * 100.0 / total_arc
        las = correct_label * 100.0 /total_label
        return correct_arc, total_arc, correct_label, total_label,uas,las


if __name__ == "__main__":
    answer_name = sys.argv[1]  # gold conll
    test_name = sys.argv[2]  # sys conll
    correct_arc, toltal_arc, correct_label, total_label, uas, las = evaluation(answer_name,test_name)
    print("UAS = %d/%d = %.2f, LAS = %d/%d = %.2f" % (correct_arc, toltal_arc, uas, correct_label, total_label, las ))