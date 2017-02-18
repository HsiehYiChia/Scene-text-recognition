import urllib.request
import time


for i in range(799, 1000):
    path = "res/neg5/"
    response = urllib.request.urlopen('http://lorempixel.com/g/640/480')
    data = response.read()
    if data != b'':
        with open(path+str(i)+".jpg", 'wb') as f:
            f.write(data)
    print(str(i) + " done")
    time.sleep(1)