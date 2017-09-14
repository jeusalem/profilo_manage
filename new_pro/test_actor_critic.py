from profilo_manage.new_pro.Actorcritic import Actorcritic
import os

merge_files_path = r'C:\Users\wade\Desktop\mergedata'
readfile_path = os.listdir(merge_files_path)
for filename in readfile_path:
    readfile = merge_files_path+"\\"+filename
    weights_path = r'C:\Users\wade\Desktop\weights_old'
    actor_critic = Actorcritic(readfile, weights_path)
    actor_critic.stock_manage()