import base64
import time

import requests
import os

# OpenAI API Key
url = "https://giegie.green/v1/chat/completions"

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def gpt4o_infer(prompt_text, image_path, _retry=0):
    if _retry > 0:
        print('retrying...')
        st = 2 ** _retry
        time.sleep(st)

    try:
        print('calling gpt4o...')
        base64_image = encode_image(image_path)

        headers = {
            'Content-Type': 'application/json',
            'Authorization': ''
        }

        data = {
            "model": "gpt-4-turbo",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt_text
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": 300
        }

        response = requests.post(url, headers=headers, json=data)
    except Exception as e:
        print(type(e), e)
        if str(e) == 'You exceeded your current quota, please check your plan and billing details.':
            exit(1)
        return gpt4o_infer(prompt_text, image_path, _retry + 1)

    response_txt = response.json()['choices'][0]['message']['content'].strip()

    return response_txt


if __name__ == '__main__':

    train_data = open('./news.txt', 'r', encoding='UTF-8')
    result_list = []
    for line in train_data.readlines():
        split_list = line.split('|')
        id = split_list[0]
        text = split_list[1]
        image_url = split_list[2].split(',')[0]
        classication = split_list[3]
        logic_prompt = 'The news text content is:' + text + 'the image content is from the image link below. Please determine whether this news is true. Note that real news usually conforms to common sense and logic.Please combine your common sense and logical reasoning to analyze the authenticity of this news. Note, provide your analysis in a single paragraph summary, and return a confidence score. The output format should be: "True (percentage score)'

        result = id + '|'
        if int(classication) == 1:
            image_path = image_url + '.jpg'
        else:
            image_path = image_url + '.jpg'
        logic_response = ''
        for i in range(0, 2):
            if i == 0:
                logic_response = gpt4o_infer(logic_prompt, image_path)
                result = result + logic_response + '|'
            if i == 1:
                consistency_prompt = 'The news text content is:' + text + 'the image content is from the image link below.Please examine the news image and its corresponding text above to determine if they are consistent. If there is a clear inconsistency between the image and the text, the news is likely false. Carefully check whether the image and text content match, and based on this assessment, judge the authenticity of the news. Note, provide your analysis in a single paragraph summary, and return a confidence score. The output format should be: "True (percentage score)'
                response = gpt4o_infer(consistency_prompt, image_path)
                result = result + response + '|'
        result_list.append(result)

    f_new = open('response_llm.txt', 'w', encoding='UTF-8')
    for item in result_list:
        f_new.write(item + '\n')
