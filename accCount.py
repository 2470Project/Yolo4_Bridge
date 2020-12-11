res=[]
cross=[]
for i in open('test_img.txt'):
    cross.append(i.replace('\n',''))
for i in open('accuracy.txt'):
    res.append(i.split(','))

res.pop(0)
# print(res)
sum_car=0
sum_rec=0
count=0
num_correct_label = 0
for i in res:
    if i[0] in cross:
        count+=1
        sum_car+=int(i[1])
        sum_rec+=int(i[2])
        num_correct_label += int(i[1]) * float(i[3])
print('The ratio of vehicles which are recognized by model is {}'.format(round(sum_rec/sum_car, 2)))

print('The accuracy of classification of recognized vehiclesia {}'.format(round(num_correct_label/sum_rec, 2)))
