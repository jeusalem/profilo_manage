from keras.models import Sequential,Model
from keras.layers import Dense,activations,Input,convolutional,merge
import tensorflow as tf
from keras import backend as K
from keras.optimizers import Adam

HIDDEN1_UNITS = 300
HIDDEN2_UNITS = 600

class CriticNetwork(object):

    def __init__(self,sess,state_size,action_size,LEARNING_RATE,TAU):
        self.sess = sess
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = LEARNING_RATE
        self.TAU = TAU

        K.set_session(sess)

        self.model,self.action,self.state = self.creat_critic_network()
        self.target_model,self.target_action,self.target_state = self.creat_critic_network()
        self.action_gradients = tf.gradients(self.model.output,self.action)
        self.sess.run(tf.initialize_all_variables())

    def gradients(self,states,actions):
        return self.sess.run(self.action_gradients,feed_dict={
            self.state : states,
            self.action : actions
        })[0]

    def target_train(self):
        critic_weights = self.model.get_weights()
        critic_target_weights = self.target_model.get_weights()
        for i in range(len(critic_weights)):
            critic_target_weights[i] = self.TAU * critic_weights[i] + (1 - self.TAU)* critic_target_weights[i]
        self.target_model.set_weights(critic_target_weights)

    def creat_critic_network(self):
        S = Input(shape=self.state_size)
        w1 = convolutional.Conv2D(filters=4,kernel_size=6,strides=4,
                                  padding='same')(S)
        w2 = convolutional.Conv2D(filters=6,kernel_size=4,strides=2,
                                  padding='same')(w1)
        w3 = convolutional.Conv2D(filters=6,kernel_size=3,strides=1,
                                  padding='same')(w2)
        w4 = convolutional.Conv2D(filters=10,kernel_size=3,strides=3,
                                  padding='same')(w3)
        w5 = Dense(HIDDEN1_UNITS,activation='relu')(w4)
        w6 = Dense(units=HIDDEN2_UNITS, activation='linear')(w5)
        #w6 = w6[0][0][0]
        #w6 = K.reshape(w6,(-1,600))

        #S2 = Dense(HIDDEN2_UNITS,activation='relu')(w6)
        #print('w6',w6)

        A = Input(shape=(1,1,self.action_size+1))
        a1 = Dense(HIDDEN2_UNITS, activation='linear')(A)
        #a1 = a1[0]
        #print('a1', a1)

        h2 = merge([w6,a1],mode='sum')
        #print('h2',h2)
        h3 = Dense(HIDDEN2_UNITS, activation='relu')(h2)
        V = Dense(self.action_size+1, activation='linear')(h3)

        model = Model(inputs=[S,A],outputs=V)

        adam = Adam(lr=self.learning_rate)
        model.compile(loss='mse',optimizer=adam)
        print('---------------------------------------------------------')
        print('-----------------CriticNetwork---------------------------')
        model.summary()
        print('-----------------CriticNetwork---------------------------')
        print('---------------------------------------------------------')
        return model,A,S


