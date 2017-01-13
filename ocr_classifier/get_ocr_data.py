import os
import os.path
import string
import subprocess

font_name = ['Cambria','Coda', 'Comic_Sans_MS', 'Courier_New', 'Domine', 'Droid_Serif', 'Fine_Ming', 'Francois_One', 'Georgia', 'Impact', 
            'Neuton', 'Play', 'PT_Serif', 'Russo_One',  'Sans_Serif', 'Syncopate', 'Time_New_Roman', 'Trebuchet_MS', 'Twentieth_Century', 'Verdana']
font_type = ['Bold', 'Bold_and_Italic', 'Italic', 'Normal']
lower = list(string.ascii_lowercase)
upper = list(string.ascii_uppercase)

message = '''Option:
    a) clean, clean the remain files
    b) get, get OCR data, you should clean before get
    c) train, train svm model by libsvm, you should get before train
'''

if __name__ == '__main__':
    func = input(message)
    if func == 'a' or func == 'clean':
        for i in font_name:
            for j in font_type:
                path = i + '/' + j
                os.remove('%s/lower.txt' % path) if os.path.exists('%s/lower.txt' % path) else None
                os.remove('%s/upper.txt' % path) if os.path.exists('%s/upper.txt' % path) else None
                os.remove('%s/number.txt' % path) if os.path.exists('%s/number.txt' % path) else None
                os.remove('%s/symbol.txt' % path) if os.path.exists('%s/symbol.txt' % path) else None
        
        os.remove('Other/number/number.txt') if os.path.exists('Other/number/number.txt') else None
        os.remove('Other/upper/upper.txt') if os.path.exists('Other/upper/upper.txt') else None
        os.remove('Other/lower/lower.txt') if os.path.exists('Other/lower/lower.txt') else None
        os.remove('OCR.data') if os.path.exists('OCR.data') else None
        os.remove('OCR.model') if os.path.exists('OCR.model') else None
        os.remove('out') if os.path.exists('out') else None
        

    elif func == 'b' or func == 'get':
        for i in font_name:
            for j in font_type:
                if os.path.exists( i + '/' + j + '/upper/a.jpg' ) != True:
                    print (i + '/' + j + ' Font empty')
                    continue

                # for number
                for n in range(0, 10):
                    infile = i + '/' + j + '/number/' + str(n) + '.jpg'
                    outfile = i + '/' + j + '/number.txt'
                    arg = "get_ocr_data.exe %s %s %d" % (infile, outfile, n )
                    subprocess.run(arg)

                # for upper case
                for k in upper:
                    infile = i + '/' + j + '/upper/' + k + '.jpg'
                    outfile = i + '/' + j + '/upper.txt'
                    arg = "get_ocr_data.exe %s %s %d" % (infile, outfile, ord(k)-ord('A')+10 )
                    subprocess.run(arg)
                
                # for lower case
                for k in lower:
                    infile = i + '/' + j + '/lower/' + k + '.jpg'
                    outfile = i + '/' + j + '/lower.txt'
                    arg = "get_ocr_data.exe %s %s %d" % (infile, outfile, ord(k)-ord('a')+10+26)
                    subprocess.run(arg)

                # for symbol
                outfile = i + '/' + j + '/symbol.txt'
                infile = i + '/' + j + '/symbol/and.jpg'
                arg = "get_ocr_data.exe %s %s %d" % (infile, outfile, 10+26+26)
                subprocess.run(arg)
                infile = i + '/' + j + '/symbol/left.jpg'
                arg = "get_ocr_data.exe %s %s %d" % (infile, outfile, 10+26+26+1)
                subprocess.run(arg)
                infile = i + '/' + j + '/symbol/right.jpg'
                arg = "get_ocr_data.exe %s %s %d" % (infile, outfile, 10+26+26+2)
                subprocess.run(arg)

        # for other
        for i in range(0, 10):
            path = 'Other/number/'+str(i)
            outfile = 'Other/number/number.txt'
            for files in os.listdir(path):
                arg = "get_ocr_data.exe %s %s %d" % (infile, outfile, i)
                subprocess.run(arg)

        for i in upper:
            path = 'Other/upper/' + i
            outfile = 'Other/upper/upper.txt'
            for files in os.listdir(path):
                arg = "get_ocr_data.exe %s %s %d" % (infile, outfile, ord(i)-ord('A') + 10)
                subprocess.run(arg)

        for i in lower:
            path = 'Other/lower/' + i
            outfile = 'Other/lower/lower.txt'
            for files in os.listdir(path):
                arg = "get_ocr_data.exe %s %s %d" % (infile, outfile, ord(i)-ord('a') + 10+26)
                subprocess.run(arg)
            




        collect = open('OCR.data', mode='w')
        for i in font_name:
            for j in font_type:
                try:
                    f = open('%s/number.txt' % (i+'/'+j), mode='r') 
                    collect.write(f.read())
                    f = open('%s/upper.txt' % (i+'/'+j), mode='r') 
                    collect.write(f.read())
                    f = open('%s/lower.txt' % (i+'/'+j), mode='r') 
                    collect.write(f.read())
                    f = open('%s/symbol.txt' % (i+'/'+j), mode='r') 
                    collect.write(f.read())
                    print ('collect from %s' % (i+'/'+j))
                except:
                    pass
        
        try:
            f = open('Other/number/number.txt', mode='r')
            collect.write(f.read())
            print ('collect from %s' % 'Other/number/number.txt')
        except:
            pass

        try:
            f = open('Other/upper/upper.txt', mode='r')
            collect.write(f.read())
            print ('collect from %s' % 'Other/upper/upper.txt')
        except:
            pass

        try:
            f = open('Other/lower/lower.txt', mode='r')
            collect.write(f.read())
            print ('collect from %s' % 'Other/lower/lower.txt')
        except:
            pass

        collect.close()
        
    elif func == 'c' or func == 'train':
        os.system('C:/libsvm-3.21/windows/svm-train -b 1 -c 2048.0 -g 0.0001220703125 OCR.data OCR.model')
        os.system('C:/libsvm-3.21/windows/svm-predict -b 1 OCR.data OCR.model out')

    else:
        print ('Wrong option!')
    