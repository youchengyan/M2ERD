import requests


url = "http://pbs.twimg.com/media/B66xzMDCMAEkd7.jpg"

response = requests.get(url)

print(response.content)

if response.status_code == 200:
    filename = 'B66xzMDCMAEkd7.jpg'

    with open('../pheme_data_process/nonrumor_images/' + filename + '.jpg', 'wb') as f:
        f.write(response.content)

    print(f"download：{filename}")
else:
    print(f"failure, code：{response.status_code}")