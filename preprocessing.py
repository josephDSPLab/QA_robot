import pandas as pd
import jieba

NUM_PROGRAM = 8
programs = []
for i in range(1, NUM_PROGRAM + 1):
    program = pd.read_csv('dataset/Program0%d.csv' % (i))

    programs.append(program)

# print(programs[0].loc[0]['Content'])
questions = pd.read_csv('dataset/Question.csv')


jieba.set_dictionary('dataset/big5_dict.txt')
def jieba_lines(lines):
    cut_lines = []

    for line in lines:
        cut_line = jieba.lcut(line)
        cut_lines.append(cut_line)

    return cut_lines
cut_programs = []

for i, program in enumerate(programs):
    episodes = len(program)
    cut_program = []
    print('Processing the %d programs' % i)
    for episode in range(episodes):
        
        lines = program.loc[episode]['Content'].split('\n')
        cut_program.append(jieba_lines(lines))

    cut_programs.append(cut_program)
print("%d programs" % len(cut_programs))
print("%d episodes in program 0" % len(cut_programs[0]))
print("%d lines in first episode of program 0" % len(cut_programs[0][0]))

print()
print("first 3 lines in 1st episode of program 0: ")
print(cut_programs[0][0][:3])

cut_programs = []

for program in programs:
    episodes = len(program)
    cut_program = []

    for episode in range(episodes):
        lines = program.loc[episode]['Content'].split('\n')
        cut_program.append(jieba_lines(lines))

    cut_programs.append(cut_program)
print("%d programs" % len(cut_programs))
print("%d episodes in program 0" % len(cut_programs[0]))
print("%d lines in first episode of program 0" % len(cut_programs[0][0]))

print()
print("first 3 lines in 1st episode of program 0: ")
print(cut_programs[0][0][:3])

cut_questions = []
n = len(questions)

for i in range(n):
    cut_question = []
    lines = questions.loc[i]['Question'].split('\n')
    cut_question.append(jieba_lines(lines))
    
    for j in range(6):
        line = questions.loc[i]['Option%d' % (j)]
        cut_question.append(jieba.lcut(line))
    
    cut_questions.append(cut_question)
print("%d questions" % len(cut_questions))
print(len(cut_questions[0]))

# 1 question
print(cut_questions[0][0])

# 6 optional reponses
for i in range(1, 7):
    print(cut_questions[0][i])

import numpy as np

np.save('cut_Programs.npy', cut_programs)
np.save('cut_Questions.npy', cut_questions)

cut_programs = np.load('cut_Programs.npy')
cut_Question = np.load('cut_Questions.npy')