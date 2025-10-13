# coding=utf-8
import random
import re
import os
from PIL import Image
import numpy as np

data_path = 'your_dataset'

original_train_data = os.path.join(data_path, 'train_nonrumor')
original_test_data = os.path.join(data_path, 'test_nonrumor')

new_train = os.path.join(data_path, 'train.txt')  # training set
new_test = os.path.join(data_path, 'test.txt')  # test set
llm_train = os.path.join(data_path, 'llm_train.txt')  # llm response of training set
llm_test = os.path.join(data_path, 'llm_test.txt')  # llm response of test set

image_file_list = [os.path.join(data_path, 'rumor_images/'), os.path.join(data_path, 'nonrumor_images/')]


def makedir(new_dir):
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)


def read_images(file_list):
    image_list = {}
    for path in file_list:
        for filename in os.listdir(path):
            try:
                img = Image.open(path + filename).convert('RGB')
                img_id = filename.split('.')[0]
                image_list[img_id] = img

            except:
                print(filename)
    return image_list  # , img_num


def select_image(image_num, image_id_list, image_list):
    for i in range(image_num):
        image_id = image_id_list[i]
        if image_id in image_list:
            return image_id
    return False


def select_data(twitter_original_data, twitter_selected_data):
    """
    select features that we need from original data
    """
    f_new = open(twitter_selected_data, 'w', encoding='UTF-8')

    fake_count = 0
    real_count = 0

    for filename in os.listdir(twitter_original_data):
        if filename.endswith(".txt"):
            file_path = os.path.join(twitter_original_data, filename)
            f_old = open(file_path, 'r', encoding='UTF-8')
            lines = f_old.readlines()
            img_result = ''
            for i, l in enumerate(lines):
                if (i + 1) % 3 == 0:
                    postText = l.replace("\n", "").replace("\r", "")
                if (i + 1) % 3 == 1:
                    postId = l.split('|')[0]
                if (i + 1) % 3 == 2:
                    img_result_temp = ''
                    pattern = r"/large/(.*?)(?:\.jpg|\||$)"
                    matches = re.findall(pattern, l)
                    for result in matches:
                        img_result_temp += str(result) + ','
                    img_result = img_result_temp
                # fakenews is 1 and true news is 0
                label = '0'
                
                last_comma_index = img_result.rfind(',')
                img_res = ''
                if last_comma_index != -1:
                    img_res = img_result[:last_comma_index] + img_result[last_comma_index + 1:]
                if (i + 1) % 3 == 0:
                    f_new.write(postId + '|' + postText + '|' + img_res + '|' + label + '\n')

            f_old.close()
    f_new.close()

    return fake_count, real_count


def get_max_len(file):
    # Get the maximal length of sentence in dataset

    f = open(file, 'r', encoding='UTF-8')

    max_post_len = 0

    lines = f.readlines()
    post_num = len(lines)
    for i in range(post_num):
        post_content = list(lines[i].split('|')[1].split())
        tmp_len = len("".join(post_content))
        if tmp_len > max_post_len:
            max_post_len = tmp_len

    f.close()
    return max_post_len


def get_data(dataset, image_list):
    if dataset == 'train':
        data_file = new_train
        llm_file = llm_train
    else:
        data_file = new_test
        llm_file = llm_test

    f = open(data_file, 'r', encoding='UTF-8')
    lines = f.readlines()

    if dataset == 'train':
        num_select = int(len(lines) * 0.2)
        lines = random.sample(lines, num_select)


    llm_f = open(llm_file, 'r', encoding='UTF-8')
    llm_lines = llm_f.readlines()

    data_post_id = []
    data_post_content = []
    data_image = []
    data_label = []
    data_llm_response = []
    data_llm_context = []
    data_llm_score = []

    data_num = len(lines)
    unmatched_num = 0

    for i, line in enumerate(lines):
        post_id = line.split('|')[0]
        post_content = line.split('|')[1]
        label = line.split('|')[-1].strip()

        image_id_list = line.split('|')[2].strip().split(',')
        img_num = len(image_id_list)
        # by default, select the first one, you can change the algorithm
        image_id = select_image(img_num, image_id_list, image_list)

        # get the llm's response
        llm_response = ''
        response_context_list = []
        response_score_list = []
        for j in range(0, len(llm_lines), 2):
            if post_id == llm_lines[j].split('|')[0]:
                response_context_list = llm_lines[j].strip().split('|')[
                                        1:-1]
                response_context_list_without_null = [s for s in response_context_list if s != 'null']
                llm_response = ''.join(response_context_list_without_null)
                response_score_list = llm_lines[j + 1].strip().split('|')[1:-1]
                response_score_list = [s for s in response_score_list if s != 'null']

        if image_id != False:
            image = image_list[image_id]

            data_post_id.append(int(post_id))
            data_post_content.append(post_content)
            data_image.append(image)
            data_label.append(int(label))
            data_llm_response.append(llm_response)
            data_llm_context.append(response_context_list_without_null)
            data_llm_score.append(response_score_list)
        else:
            unmatched_num += 1
            continue

    f.close()
    print(len(data_post_id))
    data_dic = {'post_id': np.array(data_post_id),
                'post_content': data_post_content,
                'image': data_image,
                'label': np.array(data_label),
                'llm_response': data_llm_response,  # entire response type:str
                'llm_context': data_llm_context,  # seperate responses type: list
                'llm_score': data_llm_score  # confidence of the seperate responses type: list
                }


    return data_dic, data_num - unmatched_num



if __name__ == '__main__':
    img_list = read_images(image_file_list)
    train, train_num = get_data('train', img_list)

