# -*- coding: utf-8 -*-
import json
import argparse
from mwptoolkit.data.dataset.korean_dataset import transfer_digit_to_str
import pathlib

DATA_PATH = '/home/agc2021/dataset/problemsheet.json'
EVAL_PATH ='./dataset/eval/'

# +
def find_quantity(q):              # 음수들 처리는 안한 상태 (양수는 int, float 경우 모두 대응.)
    dic = {'한':'1', '두':'2', '세':'3', '네':'4',
           '다섯':'5', '여섯':'6', '일곱':'7', '여덟':'8',
           '아홉':'9', '열':'10'}     # 추가 가능
    cnt = 0
    numbers = []
    num = ''
    q_ = ''
    find_kor_num2 = False
    try:
#         if q == '어떤 소수의 소수점을 오른쪽으로 한 자리 옮기면 원래보다 2.7만큼 커집니다. 원래의 소수를 구하시오.':
#             q_ = '어떤 소수의 소수점을 오른쪽으로 number0 자리 옮기면 원래보다 number1 만큼 커집니다. 원래의 소수를 구하시오.'
#             numbers = ['1', '2.7']
#             return q_, numbers
#         if q == '5개의 수 1.4, 9/10, 1.1, 0.5, 13/10이 있습니다. 이 중에서 0.9보다 큰 수는 모두 몇 개입니까?':
#             q_ = 'number0 개의 수 number1 , number2 , number3 , number4 , number5 이 있습니다. 이 중에서 number6 보다 큰 수는 모두 몇 개입니까?'
#             numbers = ['5', '1.4', '9/10', '1.1', '0.5', '13/10', '0.9']
#             return q_, numbers

        for idx, letter in enumerate(q):
            if find_kor_num2:
                find_kor_num2 = False
                continue
            try:
                int(letter)
                num += letter
            except:
                if (letter == '.') or (letter == '/'):
                    if num:
                        num += letter
                        continue
                if num:
                    numbers.append(num)
                    num = ''
                    q_ += f'NUM_{cnt}'
                    if letter != ' ':
                        q_ += ' '
                    cnt += 1
                if idx != len(q)-1:
                    if q[idx:idx+2] in dic:
                        find_kor_num2 = True
                        numbers.append(dic[q[idx:idx+2]])
                        q_ += f'NUM_{cnt}'
                        cnt += 1
                        continue

                if letter in dic:
                    find_kor_num = False
                    if (idx == 0) and (q[1] == ' '):
                        find_kor_num = True
                    elif (idx == len(q)-1) and (q[-2] == ' '):
                        find_kor_num = True
                    elif (q[idx-1] == ' ') and (q[idx+1] == ' '):
                        find_kor_num = True
                    if find_kor_num:      
                        numbers.append(dic[letter])
                        q_ += f'NUM_{cnt}'
                        cnt += 1
                        continue
                q_ += letter
        if num:
            numbers.append(num)
            q_ += f'NUM_{cnt}'

        if not numbers:
            numbers = ['0']
            q_ += ' NUM_0'

#         new_numbers = []
#         for i in numbers:
#             new_numbers.append(str(eval(i)))
#         return q_, new_numbers
        return q_, numbers
    except:
        numbers = ['0']
        q_ = q + ' NUM_0'
        return q_, numbers


# -

def sheet2json_main(args):
    data_path = args.data_path
    eval_path = args.eval_path
    pathlib.Path(eval_path).mkdir(parents=True, exist_ok=True) 
    with open(data_path, encoding='utf-8-sig') as f:
        data = json.load(f)
        
    total_question_length = len(data)+1
    
    problem_list = []
    for i in range(1, total_question_length):
        q_dict = {}
        mask_question, num_list = transfer_digit_to_str(data[str(i)]['question'])
#         mask_question, num_list = find_quantity(data[str(i)]['question'])
        q_dict['Question'] = mask_question
        q_dict['Numbers'] = " ".join(num_list)
        q_dict['Answer'] = 1
        q_dict['Equation'] = "- NUM_0 NUM_1"
        q_dict['ID']=str(i)
        problem_list.append(q_dict)

    with open(eval_path+'testset.json', 'w', encoding='utf-8-sig') as f:
        json.dump(problem_list, f,  indent="\t")
    with open(eval_path+'trainset.json', 'w', encoding='utf-8-sig') as f:
        json.dump([], f,  indent="\t")
    with open(eval_path+'validset.json', 'w', encoding='utf-8-sig') as f:
        json.dump([], f,  indent="\t")
