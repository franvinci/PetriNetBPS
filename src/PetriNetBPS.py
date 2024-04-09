import random
from tqdm import tqdm
import pandas as pd
import pm4py
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from src.temporal_utils import find_execution_distributions, find_arrival_distribution, find_arrival_calendar, return_time_from_calendar
from src.resources_utils import create_resources, find_roles, find_calendars
from src.distribution_utils import sample_time
from datetime import datetime, timedelta
from src.transitions_utils import return_transitions_frequency, build_models, compute_proba
from src.controlflow_utils import return_enabled_transitions, return_fired_transition, update_markings


class SimulatorParameters:

    def __init__(self, net, initial_marking, final_marking):
        
        self.net = net
        self.initial_marking = initial_marking
        self.final_marking = final_marking
        self.net_transition_labels = list(set([t.label for t in self.net.transitions if t.label]))

        self.mode_trans_weights = 'equal'
        self.transition_weights = {t: 1 for t in list(self.net.transitions)}
        self.arrival_time_distr = ('fixed', {'value': 0})
        self.arrival_calendar = find_arrival_calendar(log=None, mode='24/7')
        self.exec_distr = {l: ('fixed', {'value': 0}) for l in self.net_transition_labels}
        self.roles = None
        self.role_calendars = None
        # self.act_resources = dict()

        self.mode_trans_weights = None
        self.data_attributes = []
        self.data_attributes_categorical = []
        self.mode_history = None
        self.mode_ex_time = 'activity'


    def discover_weight_transitions(self, log, mode_trans_weights='frequency', data_attributes=[], categorical_attributes=[], history_weights=None, model_type='LogisticRegression', scaler = True):

        self.mode_trans_weights = mode_trans_weights
        self.data_attributes = data_attributes
        self.data_attributes_categorical = categorical_attributes
        self.mode_history = history_weights

        if mode_trans_weights == 'frequency':
            transition_weights = return_transitions_frequency(log, self.net, self.initial_marking, self.final_marking)
            return transition_weights

        elif mode_trans_weights == 'data_attributes':
            self.attr_values_categorical = dict()
            for a in categorical_attributes:
                self.attr_values_categorical[a] = list(pm4py.get_event_attribute_values(log, a).keys())

            transition_weights, model_coefficients, scaler_params = build_models(log, 
                                                                                 self.net, 
                                                                                 self.initial_marking, 
                                                                                 self.final_marking,
                                                                                 data_attributes, 
                                                                                 categorical_attributes,
                                                                                 self.attr_values_categorical, 
                                                                                 self.net_transition_labels, 
                                                                                 history_weights,
                                                                                 scaler=scaler,
                                                                                 model_type=model_type)
        
            return transition_weights, model_coefficients, scaler_params
    

    def discover_from_eventlog(self, 
                               log, 
                               disc_weight_transitions=True, disc_arrival_time_distr = True, disc_arrival_calendar=True, disc_exec_distr=True, disc_role_calendars=True, return_data_distr=True,
                               mode_ex_time='activity',
                               mode_trans_weights='frequency', data_attributes=[], categorical_attributes=[], history_weights=None, model_type='LogisticRegression'):

        if disc_weight_transitions:
            if mode_trans_weights == 'frequency':
                self.transition_weights = self.discover_weight_transitions(log, mode_trans_weights='frequency')
            elif mode_trans_weights == 'data_attributes':
                self.transition_weights, self.model_coefficients, self.scaler_params = self.discover_weight_transitions(log, mode_trans_weights, data_attributes, categorical_attributes, history_weights, model_type=model_type)

        if disc_arrival_calendar:
            self.arrival_calendar = find_arrival_calendar(log)

        if disc_arrival_time_distr:
            self.arrival_time_distr = find_arrival_distribution(log, self.arrival_calendar)
        
        if disc_exec_distr:
            self.mode_ex_time = mode_ex_time
            exec_distr_acts = find_execution_distributions(log, mode='activity')
            self.exec_distr = find_execution_distributions(log, mode_ex_time)
            for res in self.exec_distr.keys():
                for l in self.net_transition_labels:
                    if l not in self.exec_distr[res].keys():
                        self.exec_distr[res][l] = exec_distr_acts[l]

        if disc_role_calendars:
            if not self.roles:
                self.roles = find_roles(log)
            self.role_calendars = find_calendars(self.roles, mode='discover', log = log)

        if return_data_distr:
            self.distr_data_attr = []
            for trace in log:
                self.distr_data_attr.append([trace[0][a] for a in data_attributes])





class SimulatorEngine:

    def __init__(self, net, initial_marking, final_marking, simulation_parameters: SimulatorParameters):

        self.net = net
        self.initial_marking = initial_marking
        self.final_marking = final_marking
        self.simulation_parameters = simulation_parameters


    def insert_times_and_resources(self, traces_times):

        n_sim = len(traces_times)
        exec_times_act = dict()
        if self.simulation_parameters.mode_ex_time == 'activity':
            for act in self.simulation_parameters.net_transition_labels:
                exec_times_act[act] = list(sample_time(self.simulation_parameters.exec_distr[act], self.count_activities_sim[act]))

        simulated_events = []
        current_time = self.resource_availability.min()
        case_ids_active = list(range(len(traces_times)))

        while True:
            enabled_traces = []
            not_enabled_traces = []
            for i in case_ids_active:
                t = traces_times[i]
                if not t[-1]:
                    break
                if t[0] and t[-2] <= current_time:
                    enabled_traces.append([i] + t)
                if t[0] and t[-2] > current_time:
                    not_enabled_traces.append([i] + t)

            if not enabled_traces:
                if not not_enabled_traces:    
                    break
                else:
                    enabled_traces = not_enabled_traces

            en_t = min(enabled_traces, key=lambda x: x[-2])
            current_time = en_t[-2]
            i = en_t[0]

            j = 1
            while i+j not in case_ids_active and i+j<n_sim:
                j += 1
            if i+j<n_sim:
                traces_times[i+j][-1] = True
            act = en_t[1][0]
            del traces_times[i][0][0]

            resources_act = self.act_resources[act]
            available_resource = list(self.resource_availability[resources_act][(self.resource_availability[resources_act]<current_time)].index)
            if available_resource:
                resource = random.choices(available_resource)[0]
            else:
                resource_role_availability = {r: self.resource_availability[r] for r in resources_act}
                resource = min(resource_role_availability, key=resource_role_availability.get)
                current_time = resource_role_availability[resource]

            role = self.resource_role[resource]
            start_time = return_time_from_calendar(current_time, self.simulation_parameters.role_calendars[role])
            if self.simulation_parameters.mode_ex_time == 'activity':
                ex_time = int(exec_times_act[act].pop())
            if self.simulation_parameters.mode_ex_time == 'resource':
                ex_time = sample_time(self.simulation_parameters.exec_distr[resource][act], 1)[0]
            end_time = start_time + timedelta(seconds=ex_time)
            traces_times[i][-2] = end_time
            if not traces_times[i][0]:
                del traces_times[i]
                case_ids_active.remove(i)
            self.resource_availability[resource] = end_time
            current_time = end_time

            simulated_events.append((i, act, start_time, end_time, resource, role))

        log_data = pd.DataFrame(simulated_events, columns=['case:concept:name', 'concept:name', 'start:timestamp', 'time:timestamp', 'org:resource', 'org:role'])
        log_data.sort_values(by='time:timestamp', inplace=True)
        log_data.index = range(len(log_data))

        return log_data


    def simulate_trace(self, transition_weights):

        trace_sim = []
        tkns = list(self.initial_marking)
        enabled_transitions = return_enabled_transitions(self.net, tkns)
        
        if not self.simulation_parameters.mode_history:    
            t_fired = return_fired_transition(transition_weights, enabled_transitions)
            
            if t_fired.label:
                trace_sim.append(t_fired.label)
                self.count_activities_sim[t_fired.label] += 1   # questo serve per poi sapere quanti sample time fare per quell'attività

            tkns = update_markings(tkns, t_fired)
            while set(tkns) != set(self.final_marking):
                enabled_transitions = return_enabled_transitions(self.net, tkns) 
                t_fired = return_fired_transition(transition_weights, enabled_transitions)
                if t_fired.label:
                    trace_sim.append(t_fired.label)
                    self.count_activities_sim[t_fired.label] += 1   # questo serve per poi sapere quanti sample time fare per quell'attività
                tkns = update_markings(tkns, t_fired)

            return trace_sim
        
        else:
            transition_weights = dict()

            if self.simulation_parameters.data_attributes:
                x = random.sample(self.simulation_parameters.distr_data_attr, k=1)
            else:
                x = [[]]
            x_history = {t_l: 0 for t_l in self.simulation_parameters.net_transition_labels}
            X = x[0] + list(x_history.values())
            dict_x = dict(zip(self.simulation_parameters.data_attributes + self.simulation_parameters.net_transition_labels, X))
            if self.simulation_parameters.scaler_params:
                for c in self.simulation_parameters.data_attributes:
                    if (c not in self.simulation_parameters.data_attributes_categorical):
                        if self.simulation_parameters.scaler_params[c][0] != self.simulation_parameters.scaler_params[c][1]:
                            dict_x[c] = (dict_x[c] - self.simulation_parameters.scaler_params[c][0]) / (self.simulation_parameters.scaler_params[c][1] - self.simulation_parameters.scaler_params[c][0])
            for a in self.simulation_parameters.data_attributes_categorical:
                for v in self.simulation_parameters.attr_values_categorical[a]:
                    dict_x[a+'_'+v] = (dict_x[a] == v)*1
                del dict_x[a]
            for t in self.net.transitions:
                if type(self.simulation_parameters.transition_weights[t]) == LogisticRegression:
                    transition_weights[t] = compute_proba(self.simulation_parameters.transition_weights, t, list(dict_x.values()))
                elif type(self.simulation_parameters.transition_weights[t]) in [DecisionTreeClassifier, RandomForestClassifier]:
                    transition_weights[t] = compute_proba(self.simulation_parameters.transition_weights, t, list(dict_x.values()), logistic_regression=False)
                else:
                    transition_weights[t] = 1
            t_fired = return_fired_transition(transition_weights, enabled_transitions)
            if t_fired.label:
                trace_sim.append(t_fired.label)
                self.count_activities_sim[t_fired.label] += 1   # questo serve per poi sapere quanti sample time fare per quell'attività
            tkns = update_markings(tkns, t_fired)
            while set(tkns) != set(self.final_marking):
                if t_fired.label:
                    if self.simulation_parameters.mode_history == 'count':
                        dict_x[t_fired.label] += 1
                        if self.simulation_parameters.scaler_params:
                            if self.simulation_parameters.scaler_params[t_fired.label][0] != self.simulation_parameters.scaler_params[t_fired.label][1]:
                                dict_x[t_fired.label] = (dict_x[t_fired.label] - self.simulation_parameters.scaler_params[t_fired.label][0]) / (self.simulation_parameters.scaler_params[t_fired.label][1] - self.simulation_parameters.scaler_params[t_fired.label][0])
                    if self.simulation_parameters.mode_history == 'binary':
                        dict_x[t_fired.label] = 1
                for t in self.net.transitions:
                    if type(self.simulation_parameters.transition_weights[t]) == LogisticRegression:
                        transition_weights[t] = compute_proba(self.simulation_parameters.transition_weights, t, list(dict_x.values()))
                    elif type(self.simulation_parameters.transition_weights[t]) in [DecisionTreeClassifier, RandomForestClassifier]:
                        transition_weights[t] = compute_proba(self.simulation_parameters.transition_weights, t, list(dict_x.values()), logistic_regression=False)
                    else:
                        transition_weights[t] = 1
                enabled_transitions = return_enabled_transitions(self.net, tkns)
                t_fired = return_fired_transition(transition_weights, enabled_transitions)
                if t_fired.label:
                    trace_sim.append(t_fired.label)
                    self.count_activities_sim[t_fired.label] += 1
                tkns = update_markings(tkns, t_fired)

            return trace_sim, x[0]
        

    def simulate(self, n_istances, remove_head_tail = 0.2, starting_time = "2011-01-01 00:00:00", resource_availability=None):
    
        if resource_availability:
            starting_time = resource_availability.min()
        else:
            starting_time = datetime.strptime(starting_time, "%Y-%m-%d %H:%M:%S")
        
        n_sim = int(n_istances/(1-remove_head_tail))
        if type(list(self.simulation_parameters.roles.values())[0][1]) == int:
            self.role_resources = create_resources(self.simulation_parameters.roles)  # {role: list of res}
        else:
            self.role_resources = {role: self.simulation_parameters.roles[role][1] for role in self.simulation_parameters.roles.keys()}
        self.resources = []
        for role in self.simulation_parameters.roles:
            self.resources.extend(self.role_resources[role])
        
        if not resource_availability:
            self.resource_availability = dict()
            for r in self.resources:
                self.resource_availability[r] = starting_time
            self.resource_availability = pd.Series(self.resource_availability)
        else:
            self.resource_availability = resource_availability

        self.act_resources = dict()
        for act in self.simulation_parameters.net_transition_labels:
            for role in self.simulation_parameters.roles.keys():
                if act in self.simulation_parameters.roles[role][0]:
                    if act not in self.act_resources.keys():
                        self.act_resources[act] = self.role_resources[role]
                    else:
                        self.act_resources[act].extend(self.role_resources[role])
                        self.act_resources[act] = list(set(self.act_resources[act]))
        
        self.resource_role = dict()
        for role in self.role_resources.keys():
            for res in self.role_resources[role]:
                self.resource_role[res] = role

        current_time = starting_time
        arrival_times_diff = list(sample_time(self.simulation_parameters.arrival_time_distr, N=n_sim))
        arrival_times = []
        for t in arrival_times_diff:
            arrival_times.append(current_time)
            current_time = current_time + timedelta(seconds=int(t))
            current_time = return_time_from_calendar(current_time, self.simulation_parameters.arrival_calendar)

        self.count_activities_sim = {a: 0 for a in self.simulation_parameters.net_transition_labels}

        if (self.simulation_parameters.mode_trans_weights == 'data_attributes') and (not self.simulation_parameters.mode_history):
            transition_weights_list = dict()
            try:
                x_attr = random.sample(self.simulation_parameters.distr_data_attr, k=n_sim)
            except ValueError:
                x_attr = random.sample(self.simulation_parameters.distr_data_attr, k=len(self.simulation_parameters.distr_data_attr))
                x_attr.extend(random.sample(self.simulation_parameters.distr_data_attr, k=n_sim-len(self.simulation_parameters.distr_data_attr)))
            df_x = pd.DataFrame(x_attr, columns=self.simulation_parameters.data_attributes)
            if self.simulation_parameters.scaler_params:
                for c in list(df_x.columns):
                    if c != 'class' and (c not in self.simulation_parameters.data_attributes_categorical):
                        if self.simulation_parameters.scaler_params[c][0] != self.simulation_parameters.scaler_params[c][1]:
                            df_x[c] = (df_x[c] - self.simulation_parameters.scaler_params[c][0]) / (self.simulation_parameters.scaler_params[c][1] - self.simulation_parameters.scaler_params[c][0])
                
            for a in self.simulation_parameters.data_attributes_categorical:
                for v in self.simulation_parameters.attr_values_categorical[a]:
                    df_x[a+'_'+v] = (df_x[a] == v)*1
                del df_x[a]

            for t in self.net.transitions:
                if type(self.simulation_parameters.transition_weights[t]) in [LogisticRegression, DecisionTreeClassifier, RandomForestClassifier]:
                    transition_weights_list[t] = list(self.simulation_parameters.transition_weights[t].predict_proba(df_x)[:,1])
                else:
                    transition_weights_list[t] = [1]*n_sim
        
        if (self.simulation_parameters.mode_trans_weights == 'data_attributes') and (self.simulation_parameters.mode_history):
            x_attr = []

        simulated_traces = dict()

        for i in tqdm(range(n_sim)):
            if (self.simulation_parameters.mode_trans_weights == 'data_attributes') and (not self.simulation_parameters.mode_history):
                transition_weights = {t: transition_weights_list[t][i] for t in self.net.transitions}
            elif (self.simulation_parameters.mode_trans_weights != 'data_attributes') and (not self.simulation_parameters.mode_history):
                transition_weights = self.simulation_parameters.transition_weights
            elif (self.simulation_parameters.mode_trans_weights == 'data_attributes') and (self.simulation_parameters.mode_history):
                transition_weights = None

            arrival_time = arrival_times[i]
            if (self.simulation_parameters.mode_history):
                trace, x = self.simulate_trace(transition_weights)
                if len(x) > 0:
                    x_attr.append(x)
            else:
                trace = self.simulate_trace(transition_weights)

            simulated_traces[i] = [trace, arrival_time, False]
        
        simulated_traces[0][-1] = True

        log_data = self.insert_times_and_resources(simulated_traces)

        for n, attr in enumerate(self.simulation_parameters.data_attributes):
            log_data[attr] = log_data['case:concept:name'].apply(lambda x: x_attr[x][n])

        log_data = log_data[(log_data['case:concept:name']>int((remove_head_tail/2)*n_istances)) & (log_data['case:concept:name']<=int((remove_head_tail/2)*n_istances)+n_istances)]
        log_data['case:concept:name'] = log_data['case:concept:name'] - log_data['case:concept:name'].min() + 1

        # reset
        self.role_resources = None
        self.resource_availability = None
        self.act_resources = None
        self.count_activities_sim = None
        self.resource_role = None

        return log_data