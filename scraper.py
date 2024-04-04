import requests
import os

url_list = []
with open("catalog.txt", 'r', encoding='utf-8', errors='ignore') as file:
    count = 0
    for line in file:
        line = line.split("\t")
        genre = line[7]

        if genre == "painting":
            author = line[0].strip('\"')
            author_year = line[1]
            title = line[2].strip('\"')
            year = line[3]
            medium= line[4]
            museum = line[5]
            url = line[6].strip()
            if ("html" in url):
                url = url.replace("html","art",1)
                url = url.rstrip(url[-4:]) + "jpg"
                print(url)
                try:
                    response = requests.get(url)
                    if response.status_code == 200:
                        filename = f"images/{title} {author}.jpg"
                        with open(filename, 'wb') as f:
                            f.write(response.content)
                    else:
                        print(f"Failed to download {url}: Status code {response.status_code}")
                    
                except Exception as e:
                    print(f"Failed to download {url}: {e}")
            else:
                print("error: wrong format at:")
                print(url)
                break
            