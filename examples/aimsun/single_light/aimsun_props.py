import os
import csv
import sys
sys.path.append('/home/damian/anaconda3/envs/aimsun_flow/lib/python2.7/site-packages')
import numpy as np
import pandas as pd
import json

cwd = os.getcwd() #Get current working directory
class Aimsun_Params:
    
    def __init__(self,my_csv):
        self.df = pd.read_csv(my_csv)

    def get_green_phases(self, node_id):
        gp = self.df.loc[self.df.node_id == node_id, 'green_phases_list'].values[0]
        green_phases = json.loads(gp)
        #gp_list = green_phases[0]
        return green_phases

    def get_cp_cycle_dict(self, node_id, rep_name):
        cp_cycle_dict = {}
        df_cp = self.df[self.df['node_id'] == node_id]
        cp = df_cp.loc[df_cp.rep_name == rep_name, 'cp_id_list'].values[0]
        cp = json.loads(cp)

        df_cycle = self.df[self.df['node_id'] == node_id]
        cycle = df_cycle.loc[df_cycle.rep_name == rep_name, 'cycle_list'].values[0]
        cycle = json.loads(cycle)

        cp_cycle_dict = dict(zip(cp, cycle))

        return cp_cycle_dict

    def get_sum_interphase_per_ring(self, node_id):
        s_int = self.df.loc[self.df.node_id == node_id, 'sum_interphase'].values[0]
        sum_interphases = json.loads(s_int)
        return sum_interphases

    def get_max_dict(self, node_id, rep_name):
        in_df = self.df.set_index('node_id')
        b_df = in_df.loc[[node_id],['rep_name','maxd_1','maxd_2','maxd_3','maxd_p']]
        in_df = b_df.set_index('rep_name')
        c_df = in_df.loc[[rep_name],['maxd_1','maxd_2','maxd_3','maxd_p']]
        maxd_1 = [json.loads(c_df.iloc[0]['maxd_1'])]
        maxd_2 = [json.loads(c_df.iloc[0]['maxd_2']) ]
        maxd_3 = [json.loads(c_df.iloc[0]['maxd_3'])]
        maxd_p = json.loads(c_df.iloc[0]['maxd_p'])
        max_dict = sum([maxd_1,maxd_2,maxd_3],[])
        # max_dict = dict(zip(maxd_p, maxd_list))
        return max_dict, maxd_p

    def get_detector_ids(self, node_id):
        pass

    def get_start_phases(self, node_id):
        sp = self.df.loc[self.df.node_id == node_id, 'start_phase'].values[0]
        start_phases = json.loads(sp)
        #gp_list = green_phases[0]
        return start_phases

class Export_Params:
    def __init__(self,rep_name, rep_seed):
        # create folder to store csv files
        self.csv_dir = os.path.join(cwd, r'csv_files')
        if not os.path.exists(self.csv_dir):
            os.mkdir(self.csv_dir)
            print('Created new directory:{}'.format(self.csv_dir))
        self.file_name = str(rep_name) + '_'  + str(rep_seed) + '_'


    def export_delay_action(self, node_id, rep_seed, delay, action_list, util_list, r_queue, time, timeSta):
        time = time
        timeSta = timeSta
        ave_app_delay = delay
        data_list = [rep_seed, time, node_id, delay, r_queue]

        for util in util_list:
            data_list.append(util)
        
        for action in action_list:
            data_list.append(action)

        with open(self.rep_name, 'a') as csvFile:
            csv_writer = csv.writer(csvFile)
            csv_writer.writerows([data_list,])

    def export_env_rewards(self, rep_name, rep_seed, d_list):
        # d_list = [node_id, time, reward, r_queue, gutil]
        new_dir = os.path.join(self.csv_dir, r'rewards')
        if not os.path.exists(new_dir):
            os.makedirs(new_dir)
            print('Created new directory:{}'.format(new_dir))

        file_name = new_dir + '/' + str(rep_name) + '_'  + str(rep_seed) + '_' + 'rewards.csv'
        fieldnames = ['node_id', 'time', 'reward', 'r_queue',' gutil']
        
        with open(file_name, 'a') as csvFile:
            csv_writer = csv.writer(csvFile, fieldnames)

            if csvFile.tell() == 0: # if file doesnt exists write header
                csv_writer.writerow(fieldnames)
                print('Created new file: {}'.format(file_name))

            csv_writer.writerows([d_list,])

    def export_sys_data(self, data_list):
        # sys d_list: [time, flow, tta, dta, sta, sa, density, numstops, volume(count), inflow, incount, missed_turns]
        new_dir = os.path.join(self.csv_dir, r'sys')
        if not os.path.exists(new_dir):
            os.makedirs(new_dir)
            print('Created new directory:{}'.format(new_dir))

        file_name = new_dir + '/' + self.file_name + 'sys.csv'
        fieldnames = ['time', 'flow', 'tta', 'dta', 'sta', 'sa', 'density', 'numstops', 'count', 'inflow', 'incount', 'missed_turns']
        
        with open(file_name, 'a') as csvFile:
            csv_writer = csv.writer(csvFile, fieldnames)
            
            if csvFile.tell() == 0: # if file doesnt exists write header
                csv_writer.writerow(fieldnames)
                print('Created new file: {}'.format(file_name))

            csv_writer.writerows([data_list,])

    #FIXME make input to dictionary
    def export_node_data(self, node_data_list):
    # node d_list: [node_id, time, delay, missed_turns, **phase durations] # stat data
        new_dir = os.path.join(self.csv_dir, r'node')
        if not os.path.exists(new_dir):
            os.makedirs(new_dir)
            print('Created new directory:{}'.format(new_dir))

        directory = '/csv_files/node/'
        file_name = new_dir +'/' + self.file_name + 'node.csv'
        fieldnames = ['node_id', 'time', 'delay', 'missed_turns', '1', '3', '5', '7', '9', '11', '13', '15']
         
        with open(file_name, 'a') as csvFile:
            csv_writer = csv.writer(csvFile, fieldnames)
            
            if csvFile.tell() == 0: # if file doesnt exists write header
                csv_writer.writerow(fieldnames)
                print('Created new file: {}'.format(file_name))

            csv_writer.writerows([node_data_list,])
            # for node, d_list in node_data_dict.items():
            #     csv_writer.writerow([node, d_list])

    #TODO section  d_list: [node_id, section_id, time, num_lanes, queue, flow, tta, dta, sta, sa, 
                    # density, numstops, volume(count), inflow, incount, lq_ave, lq_max]
    def export_section_data(self, section_data_dict):
        new_dir = os.path.join(self.csv_dir, r'section')
        if not os.path.exists(new_dir):
            os.makedirs(new_dir)
            print('Created new directory:{}'.format(new_dir))

        file_name = new_dir + '/' + self.file_name + '_' + 'section.csv'
        fieldnames = ['node_id', 'section_id', 'time', 'num_lanes', 'queue', 'flow', 'tta', 'dta', 'sta', 'sa', 'density', 'numstops', 'count', 'inflow', 'incount', 'lq_ave', 'lq_max']
 
        with open(file_name, 'a') as csvFile:
            csv_writer = csv.DictWriter(csvFile)

            if csvFile.tell() == 0:
                csv_writer.writeheader()

