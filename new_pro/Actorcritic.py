from profilo_manage.new_pro.ActorNetwork import ActorNetwork
from profilo_manage.new_pro.CriticNetwork import CriticNetwork
from profilo_manage.new_pro.ReplayBuffer import ReplayBuffer
from openpyxl import load_workbook
from profilo_manage.new_pro.Equation import Equation
from profilo_manage.new_pro.Datamerge import Data
import profilo_manage.new_pro.constant_def as NC
import numpy as np
import tensorflow as tf
import keras.backend as K
import os
import math
import time



deafualt_weights_path = NC.DEAFUALT_WEIGHTS_PATH #存放权重的默认文件夹路径
action_dim = 1+NC.NUM_STOCKS #1 是表示现金

class Actorcritic(object):

    def __init__(self,read_file,weights_path=deafualt_weights_path,is_train=True):
        self.read_file = read_file
        self.weights_path = weights_path
        self.is_train = is_train

    # 读取excel文件，将每个股票的每一天数据读作一列
    # 例如：[[15.9, 9.19, 16.06, 9.75, 15.5, 9.19, 15.89, 9.64, 2735500, 844200, 43018480, 8043152.5]
    #       [16, 9.64, 16, 9.74, 15.6, 9.55, 15.75, 9.61, 1193200, 488500, 18696056, 4712759]]
    #两天，a,b两只股票的数据[a_open,b_open,a_high,b_high,a_low,b_low,a_close,b_close,a_volumn,b_volumn,a_amount,b_amount]
    #len(full_days)=所有股票的天数
    def get_full_days(self):

        wb = load_workbook(self.read_file)
        ws = wb.get_active_sheet()
        rows = ws.rows
        for i in range(NC.TITLE_START_ROW): #去掉标题栏加前面空着的几行
            rows.__next__()
        index_features = 0
        full_days = []
        one_day = []
        for row in rows:
            if index_features<len(NC.FEATURES):
                # one_day.extend(row[NC.START_COLUMN_DATE+1:].value) #去掉日期和feature那两列
                for i in range(NC.START_COLUMN_DATE+1,len(row)):
                    one_day.append(row[i].value)
                index_features += 1
            else:
                #print(one_day)
                full_days.append(one_day)
                one_day = []
                for i in range(NC.START_COLUMN_DATE+1,len(row)):
                    one_day.append(row[i].value)
                index_features = 1
        full_days.append(one_day)

        return  full_days


    def get_ratio_days(self):
        full_days = np.array(self.get_full_days())
        ratio_days = []
        for i in range(1,len(full_days)): #后一天除以前一天的，所以从1开始
            pre_day = full_days[i-1]
            now_day = full_days[i]
            ratio_day = now_day/pre_day
            ratio_day = ratio_day.reshape(NC.NUM_FEATURES,NC.NUM_STOCKS)
            ratio_days.append(ratio_day)
        ratio_days = np.array(ratio_days)
        return ratio_days


    def ratio_days_iter(self):
        ratio_days = self.get_ratio_days()
        for ratio_day in ratio_days:
            yield ratio_day


    def get_states(self):
        state = []
        iter_ratio_day = self.ratio_days_iter()
        NUM_FEATURES = len(NC.FEATURES)
        NUM_STOCKS = NC.NUM_STOCKS
        NUM_DAYS = NC.NUM_DAYS
        full_states = []
        i = 0
        for i in range(NC.NUM_DAYS):
            ratio_day = iter_ratio_day.__next__()
            print(ratio_day.shape)
            ratio_day = ratio_day.reshape((NC.NUM_FEATURES, NC.NUM_STOCKS))
            state.append(ratio_day)
        full_states.append(state)

        try:
            while(True):
                state = state[1:]
                ratio_day = iter_ratio_day.__next__()
                ratio_day = ratio_day.reshape((NC.NUM_FEATURES, NC.NUM_STOCKS))
                state.append(ratio_day)
                full_states.append(state)


        except StopIteration:

            return full_states


    def train_an_esisode(self,actor,critic):
        full_states = self.get_states()
        loss = 0
        p_c = 1
        buff = ReplayBuffer(NC.BUFFER_SIZE)

        b_imp_without_cash = np.zeros(NC.NUM_STOCKS)
        b_imp = np.append(1,b_imp_without_cash)
        print('b_imp____without_cash',b_imp_without_cash)

        for i in range(len(full_states)-1):
            s_t = np.array(full_states[i])
            s_t1 = np.array(full_states[i+1])
            x_t1 = s_t1[-1][NC.close_feature_index]
            x_t1 = np.append(1,x_t1)
            print('x_t1',x_t1)
            print('s_t1',s_t1)
            s_t1 = s_t1.transpose((1,2,0))
            s_t = s_t.transpose((1,2,0))
            a_t = actor.model.predict(s_t.reshape(1,NC.NUM_FEATURES,NC.NUM_STOCKS,NC.NUM_DAYS),NC.BATCH_SIZE).flatten()
            print('a_t',a_t)
            # print(s_t1.transpose(2,0,1)[-1][3])
            # x_t1 = s_t1.transpose(2,0,1)[-1][3] #第t+1天与第t天的收盘价比值,未加现金
            # x_t1 = np.append(1,x_t1) #加上现金，比例总是1
            equation = Equation(b_imp_without_cash,a_t[1:],NC.c_s,NC.c_p)
            u_t = equation.get_u_t()
            print('u_t',u_t)

            #################################
            #reward
            r_t = math.log(np.inner(a_t, x_t1)) + math.log(u_t)
            # r_t = p_c*u_t*(np.inner(a_t,x_t1)-1)
            #################################

            buff.add(s_t,a_t,r_t,s_t1)
            p_a = p_c*u_t
            t_c = p_c - p_a
            #print('t_c',t_c)
            print('p_a',p_a)
            b_imp_next = a_t * x_t1 / np.inner(a_t, x_t1)
            p_c_next = p_a * np.inner(a_t, x_t1)
            #####################################
            #TODO write_to_excel
            #####################################
            p_c = p_c_next
            b_imp = b_imp_next
            b_imp_without_cash = b_imp_next[1:]

            batch = buff.getBatch(NC.BATCH_SIZE)
            states = np.asarray([e[0] for e in batch])
            actions = np.asarray([e[1] for e in batch])
            actions = actions.reshape((len(batch), 1, 1, 1 + NC.NUM_STOCKS))
            rewards = np.asarray([e[2] for e in batch])
            new_states = np.asarray([e[3] for e in batch])
            y_t = np.asarray([e[1] for e in batch])
            # print('y_t___________',y_t)

            target_q_value = critic.model.predict([new_states, actor.target_model.predict(new_states)])
            # print('target',target_q_value)
            # print('target_q_shape',target_q_value.shape)
            # print('target_q_value',target_q_value)

            for m in range(len(batch)):

                y_t[m] = rewards[m] + NC.GAMMA*target_q_value[m]
            # print('y_t', y_t)
            # print('y_t_shape',y_t.shape)
            y_t = y_t.reshape((len(batch),1,1,NC.NUM_STOCKS+1))

            if self.is_train :
                loss = critic.model.train_on_batch([states, actions], y_t)
                print('loss',loss)
                a_for_grad = actor.model.predict(states)
                grads = critic.gradients(states, a_for_grad)
                actor.train(states, grads)

                actor.target_train()
                critic.target_train()

    #
    #
    #
    def stock_manage(self):
        state_iter = self.get_states()
        sess = tf.Session()
        K.set_session(sess)

        actor = ActorNetwork(sess, (NC.NUM_FEATURES, NC.NUM_STOCKS, NC.NUM_DAYS),
                             NC.LEARNING_RATE, NC.TAU)
        critic = CriticNetwork(sess, (NC.NUM_FEATURES, NC.NUM_STOCKS, NC.NUM_DAYS),
                               NC.NUM_STOCKS, NC.LEARNING_RATE,NC.TAU)


        weights_path = self.weights_path
        actor_weights = weights_path +r'\actor_weights.h5'
        critic_weights = weights_path +r'\critic_weights.h5'
        target_actor_weights = weights_path +r'\target_weights.h5'
        target_critic_weights = weights_path +r'\target_critic_weights.h5'

        if os.path.exists(actor_weights):
            print('loading weights')
            actor.model.load_weights(actor_weights)
            actor.target_model.load_weights(target_actor_weights)
            critic.model.load_weights(critic_weights)
            critic.target_model.load_weights(target_critic_weights)

            # try:
            #     while(True):
            #         state = state_iter.__next__()
            #         print(state)
            # except StopIteration:
            #     print('end')
        self.train_an_esisode(actor,critic)

        #########################
        #save weights
        actor.model.save_weights(actor_weights)
        actor.target_model.save_weights(target_actor_weights)
        critic.model.save_weights(critic_weights)
        critic.target_model.save_weights(target_critic_weights)


    # def manage_stock2(self):
    #     sess = tf.Session()
    #     K.set_session(sess)
    #     actor = ActorNetwork(sess, (NC.NUM_FEATURES, NC.NUM_STOCKS, NC.NUM_DAYS),
    #                          NC.LEARNING_RATE, NC.TAU)
    #     critic = CriticNetwork(sess, (NC.NUM_FEATURES, NC.NUM_STOCKS, NC.NUM_DAYS),
    #                            NC.NUM_STOCKS, NC.LEARNING_RATE,NC.TAU)
    #     buff = ReplayBuffer(NC.BUFFER_SIZE)
    #     ratio_day = self.get_ratio_days()
    #     actor_weights = self.weights_path + r'\actor_weights.h5'
    #     critic_weights = self.weights_path + r'\critic_weights.h5'
    #     target_actor_weights = self.weights_path + r'\target_weights.h5'
    #     target_critic_weights = self.weights_path + r'\target_critic_weights.h5'
    #     if os.path.exists(actor_weights):
    #         print('loading weights')
    #         actor.model.load_weights(actor_weights)
    #         actor.target_model.load_weights(target_actor_weights)
    #         critic.model.load_weights(critic_weights)
    #         critic.target_model.load_weights(target_critic_weights)
    #     p_c = 1
    #     b_imp = np.append(1,np.zeros(NC.NUM_STOCKS))
    #
    #     for i in range(len(ratio_day) - NC.NUM_DAYS):
    #         loss = 0
    #
    #         # print('i',i)
    #         # print('p----c',p_c)
    #
    #         s_t = ratio_day[i:i + NC.NUM_DAYS, :, :]
    #         s_t1 = ratio_day[i + 1:i + 1 + NC.NUM_DAYS, :, :]
    #
    #         s_t = np.transpose(s_t, (1, 2, 0))
    #         s_t1 = np.transpose(s_t1, (1, 2, 0))
    #         # buff.add(s_t, a_t[0], r_t, s_t1, done)  # Add replay buffer
    #         #
    #         # # Do the batch update
    #         # batch = buff.getBatch(BATCH_SIZE)
    #         # states = np.asarray([e[0] for e in batch])
    #         price_change = []
    #         for l in range(NC.NUM_STOCKS):
    #             price_change.append(s_t1[NC.close_feature_index][l][NC.NUM_DAYS - 1])
    #         price_change = np.append(1, price_change)  # ratio of cash is always 1
    #         # print('price_change',price_change)
    #
    #         # print('s_t',s_t)
    #         s_to_a = s_t.reshape(
    #             (1, NC.NUM_FEATURES, NC.NUM_STOCKS, NC.NUM_DAYS))  # add batchsize as the first dimension
    #         a_t = actor.model.predict(s_to_a, NC.BATCH_SIZE)[0][0][0]  # 变成一维的 shape=(None, 1, 1, 11)
    #         print('a_t', a_t)
    #
    #         print('p__c', p_c)
    #         equation = Equation(b_imp[1:], a_t[1:], NC.c_s, NC.c_p)
    #         u_t = equation.get_u_t()
    #         r_t = math.log(np.inner(a_t, price_change)) + math.log(u_t)
    #         buff.add(s_t, a_t, r_t, s_t1)
    #
    #         #################################################
    #         # write to table to show the performance
    #
    #         pa = p_c * u_t
    #         b_imp_next = a_t * price_change / np.inner(a_t, price_change)
    #         p_c_next = pa * np.inner(a_t, price_change)
    #
    #         # the next day
    #
    #         ############
    #         #data.write_result(wb, b_imp, a_t, p_c, pa, u_t, r_t, p_c - pa, i)
    #         ############
    #
    #         p_c = p_c_next
    #         b_imp = b_imp_next
    #
    #         batch = buff.getBatch(NC.BATCH_SIZE)
    #         states = np.asarray([e[0] for e in batch])
    #         actions = np.asarray([e[1] for e in batch])
    #         actions = actions.reshape((len(batch), 1, 1, 1 + NC.NUM_STOCKS))
    #         print('actions', actions)
    #         # print('actons,shape',actions.shape)
    #         rewards = np.asarray([e[2] for e in batch])
    #         new_states = np.asarray([e[3] for e in batch])
    #         y_t = np.asarray([e[1] for e in batch])
    #
    #         target_q_value = critic.model.predict([new_states, actor.target_model.predict(new_states)])
    #
    #         for m in range(len(batch)):
    #             if i == (len(ratio_day) - 4 - 1):
    #                 y_t[m] = rewards[m]
    #             else:
    #                 y_t[m] = rewards[m] + NC.GAMMA * target_q_value[m]
    #         y_t = y_t.reshape((len(batch), 1, 1, NC.NUM_STOCKS + 1))
    #         print('y_t', y_t)
    #         loss = critic.model.train_on_batch([states, actions], y_t)
    #         a_for_grad = actor.model.predict(states)
    #         grads = critic.gradients(states, a_for_grad)
    #         actor.train(states, grads)
    #
    #         actor.target_train()
    #         critic.target_train()
    #
    #     #wb.save(write_file_path)
    #
    #     actor.model.save_weights(actor_weights)
    #     actor.target_model.save_weights(target_actor_weights)
    #     critic.model.save_weights(critic_weights)
    #     critic.target_model.save_weights(target_critic_weights)
    #





if __name__ == '__main__':
    # original_txt_files = r'C:\Users\wade\Desktop\txtfile'
    # merge_files_path =  r'C:\Users\wade\Desktop\mergedata'
    # meige_data = Data(original_txt_files,merge_files_path)
    #
    # time.sleep(5)
    #
    # original_txt_files = r'C:\Users\wade\Desktop\txtfile2'
    # merge_files_path = r'C:\Users\wade\Desktop\mergedata'
    # meige_data = Data(original_txt_files, merge_files_path)
    #
    #
    # readfile_path = os.listdir(merge_files_path)
    # for file_name in readfile_path:
    #     readfile = merge_files_path+"\\"+file_name
    #     print(readfile)
    #     weights_file = r'C:\Users\wade\Desktop\weights'
    #     actor_critic = Actorcritic(readfile,weights_file)
    #     actor_critic.stock_manage()


    #############################
    readfile = r'C:\Users\wade\Desktop\mergedata\result_2017_09_14_11_10_53.xlsx'
    actorcritic = Actorcritic(readfile)
    actorcritic.stock_manage()
