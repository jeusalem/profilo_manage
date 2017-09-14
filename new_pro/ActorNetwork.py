from keras.models import Sequential,Model
from keras.layers import Dense,Input,convolutional
import keras.backend as K
import tensorflow as tf


HIDDEN1_UNITS = 300
HIDDEN2_UNITS = 600
ACTION_DIM = 11

class ActorNetwork(object):

    def __init__(self, sess, state_size, LEARNING_RATE,TAU ):
        self.sess = sess
        self.state_size = state_size
        self.TAU = TAU
        self.learning_rate = LEARNING_RATE

        K.set_session(sess)

        self.model,self.weights,self.state = self.creat_actor_network(self.state_size)
        self.target_model,self.target_weights,self.target_state = self.creat_actor_network(self.state_size)
        self.action_gradient = tf.placeholder(tf.float32,[None,1,1,ACTION_DIM])
        self.params_gradient = tf.gradients(self.model.output,self.weights,-self.action_gradient)
        grads = zip(self.params_gradient,self.weights)
        self.optimize = tf.train.AdamOptimizer(self.learning_rate).apply_gradients(grads)
        self.sess.run(tf.global_variables_initializer())

    def train(self,states,action_grads):
        self.sess.run(self.optimize,feed_dict={
            self.state : states,
            self.action_gradient : action_grads
        })

    def target_train(self):
        actor_weights = self.model.get_weights()
        actor_target_weights = self.target_model.get_weights()
        for i in range(len(actor_weights)):
            actor_target_weights[i] = self.TAU * actor_weights[i] + (1-self.TAU) * actor_weights[i]
        self.target_model.set_weights(actor_target_weights)


    def creat_actor_network(self,input_shape):
        S = Input(shape=input_shape)
        w1 = convolutional.Conv2D(filters=4,kernel_size=6,strides=4,
                                  padding='same')(S)
        w2 = convolutional.Conv2D(filters=6,kernel_size=4,strides=2,
                                  padding='same')(w1)
        w3 = convolutional.Conv2D(filters=6,kernel_size=3,strides=1,
                                  padding='same')(w2)
        w4 = convolutional.Conv2D(filters=10,kernel_size=3,strides=3,
                                  padding='same')(w3)
        w5 = Dense(11,activation='softmax')(w4)
        model = Model(inputs=S,outputs=w5)

        print('---------------------------------------------------------')
        print('-----------------ActorNetwork---------------------------')
        model.summary()
        print('-----------------ActorNetwork---------------------------')
        print('---------------------------------------------------------')

        return model,model.trainable_weights,S