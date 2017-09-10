'''
load weights and biases from Q - a 100X100X40 feedforward fully connected network with a 4X1 input vector and a 5X1 output vector
(used for self balancing an inverted pendulum using DQN)
'''

Ws, bs = Q.get_weights()
W_tensor = sess.run(Ws)
B_tensor = sess.run(bs)


# input layer:
scipy.io.savemat('./weights/net.IW{1}(all,1).mat', mdict={'arr': W_tensor[0][0]})
scipy.io.savemat('./weights/net.IW{1}(all,2).mat', mdict={'arr': W_tensor[0][1]})
scipy.io.savemat('./weights/net.IW{1}(all,3).mat', mdict={'arr': W_tensor[0][2]})
scipy.io.savemat('./weights/net.IW{1}(all,4).mat', mdict={'arr': W_tensor[0][3]})


# first 100 neurons layer	(100X100)
for i in range(100):
	scipy.io.savemat('./weights/net.LW{2,1}(all,'+str(i+1)+').mat', mdict={'arr': W_tensor[1][i]})


# second 100 neurons layer	(40X100)
for i in range(100):
	scipy.io.savemat('./weights/net.LW{3,2}(all,'+str(i+1)+').mat', mdict={'arr': W_tensor[2][i]})

# output layer	(5X40)	WE HAVE 5 ACTIONS HERE!
for i in range(40):
	scipy.io.savemat('./weights/net.LW{4,3}(all,'+str(i+1)+').mat', mdict={'arr': W_tensor[3][i]})


#*****************************BIASES************************
scipy.io.savemat('./weights/net.b{4}.mat', mdict={'arr': B_tensor[3]})
scipy.io.savemat('./weights/net.b{3}.mat', mdict={'arr': B_tensor[2]})
scipy.io.savemat('./weights/net.b{2}.mat', mdict={'arr': B_tensor[1]})
scipy.io.savemat('./weights/net.b{1}.mat', mdict={'arr': B_tensor[0]})