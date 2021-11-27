



import matplotlib.pyplot as plt

b = 3
c = 7
x = [1,2,3,4,5]
y = []
y = [10, 13, 16, 19, 22]
for i in x:
	result = b*i + c
	y.append(result)
	
print(y)


plt.title('I Implement Linear Regresion Model')
plt.xlabel('X_Axis')
plt.ylabel('Y_Axis')
plt.plot(x,y)
plt.show()





