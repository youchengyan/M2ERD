import os


data_path = 'MM17-WeiboRumorSet\MM17-WeiboRumorSet\\tweets'
train_data_fake = os.path.join(data_path, 'train_rumor.txt')
train_data_true = os.path.join(data_path, 'train_nonrumor.txt')
test_data_fake = os.path.join(data_path, 'test_rumor.txt')
test_data_true = os.path.join(data_path, 'test_nonrumor.txt')

new_data_path = "dataset\\test_nonrumor"
new_train_data_fake = os.path.join(new_data_path, 'test_rumor.txt')
new_train_data_true = os.path.join(new_data_path, 'test_nonrumor.txt')

def append_lines_to_file(filename, lines_to_append):
    with open(filename, 'a', encoding='UTF-8') as file:
        for line in lines_to_append:
            file.write(line)


if __name__ == '__main__':
    f_old = open(test_data_true, 'r', encoding='UTF-8')
    lines = f_old.readlines()
    j = 0
    for i, l in enumerate(lines):
        if "condition" in l:
            j += 1
            lines_to_append = []
            lines_to_append.append(lines[i-2])
            lines_to_append.append(lines[i - 1])
            lines_to_append.append(lines[i])
            append_lines_to_file(new_train_data_true, lines_to_append)
    print(j)