import csv
import sys
sys.path.append('/home/cjrsantos/anaconda3/envs/aimsun_flow/lib/python2.7/site-packages')
import numpy as np
import pandas as pd
import json

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

## Insert function for detectors and for gUtil

class Export_Params:
    def __init__(self, output_csv, green_phases):
        self.green_phases = green_phases
        with open(output_csv,'a') as csvFile:
            data = []
            fieldnames = ['time', 'delay_time', '1', '3', '5', '7', '9', '11', '13', '15', 'cycle', 'barrier']
            csv_writer = csv.DictWriter(csvFile, fieldnames=fieldnames)
            csv_writer.writerow({'time': current_time})
            csv_writer.writeheader()

    def export_delay_time(self,node_id, time, timeSta, timeTrans, acycle):
        time = time
        timeSta = timeSta
        ave_app_delay = aapi.AKIEstGetPartialStatisticsNodeApproachDelay(node_id)

        if replication_name == 8050297:
            with open('{}.csv'.format(replication_name), 'a') as csvFile:
                csv_writer = csv.DictWriter(csvFile, fieldnames=fieldnames)
                csv_writer.writerow({'time': '{}'.format(time), 'delay_time': '{}'.format(ave_app_delay)})


        elif replication_name == 8050315:
            with open('{}.csv'.format(replication_name), 'a') as csvFile:
                csv_writer = csv.DictWriter(csvFile, fieldnames=fieldnames)
                csv_writer.writerow({'time': '{}'.format(time), 'delay_time': '{}'.format(ave_app_delay)})


        if replication_name == 8050322:
            with open('{}.csv'.format(replication_name), 'a') as csvFile:
                csv_writer = csv.DictWriter(csvFile, fieldnames=fieldnames)
                csv_writer.writerow({'time': '{}'.format(time), 'delay_time': '{}'.format(ave_app_delay)})

        ##add phase_durations

##test
#print(get_green_phases(3344))
#print('********************')
#ap = Aimsun_Params('/home/cjrsantos/flow/examples/aimsun/single_light/aimsun_params.csv')
#print(ap.get_cp_cycle_dict(3369,8050315), print(type(ap.get_cp_cycle_dict(3369,8050315))))
#print('********************')
#print(get_sum_interphase_per_ring(3344), print(type(get_sum_interphase_per_ring(3344))))
#print('********************')
#max_d, max_p = ap.get_max_dict(3344,8050322)
#cur_mad = max_p[4]
#ms = max_d[cur_mad]
#print(ms)