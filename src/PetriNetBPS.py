import random
from tqdm import tqdm
import pandas as pd
import pm4py
from sklearn.linear_model import LogisticRegression
from src.temporal_utils import find_execution_distributions, find_arrival_distribution, find_arrival_calendar, return_time_from_calendar
from src.resources_utils import n_to_weekday, create_resources, find_roles, find_calendars
from src.distribution_utils import sample_time
from datetime import datetime, timedelta
from src.pn_utils import add_start_end_transitions
from src.transitions_utils import return_transitions_frequency, build_models, return_scaler_params, compute_proba
from src.controlflow_utils import return_enabled_transitions, return_fired_transition, update_markings


# class SimulatorParameters:

#     def __init__(self):
#         None




class SimulatorEngine:

    def __init__(self, 
                 net, initial_marking, final_marking, 
                 transition_weights='frequency', 
                 log=[], data_attributes=[], categorical_attributes=[], history_weights=None, scaler=False, 
                 discovery_simulation_parameters=True,
                 calendars_mode = 'discover',
                 exec_distr = None, roles = None, arrival_time_distr = None, arrival_calendar = None,
                 starting_time = "2011-01-01 00:00:00"):

        self.net = net
        self.initial_marking = initial_marking
        self.final_marking = final_marking
        # add_start_end_transitions(self.net, self.initial_marking, self.final_marking)
        self.net_transition_labels = list(set([t.label for t in self.net.transitions if t.label]))

        self.data_attributes = data_attributes
        self.history_weights = history_weights
        if self.data_attributes:
            self.x_attr = []

        self.starting_time = datetime.strptime(starting_time, "%Y-%m-%d %H:%M:%S")
        if discovery_simulation_parameters:
            self.exec_distr = find_execution_distributions(log)
            self.arrival_time_distr = find_arrival_distribution(log)
            self.arrival_calendar = find_arrival_calendar(log)
            if not roles:
                self.roles = find_roles(log)
            else:
                self.roles = roles
            if calendars_mode == '24/7':
                self.calendars = find_calendars(self.roles)
            elif calendars_mode == 'manual':
                self.calendars = find_calendars(self.roles, mode='manual')
            elif calendars_mode == 'discover':
                self.calendars = find_calendars(self.roles, mode='discover', log = log)
        else:
            self.exec_distr = exec_distr
            self.arrival_time_distr = arrival_time_distr
            self.arrival_calendar = arrival_calendar
            self.roles = roles
            self.calendars = find_calendars(self.roles, mode='manual')

        self.role_resources = create_resources(self.roles)
        resources = []
        for role in self.roles:
            resources.extend(self.role_resources[role])
            self.resource_availability = dict()
            for r in resources:
                self.resource_availability[r] = self.starting_time
            self.resource_availability = pd.Series(self.resource_availability)

        self.act_resources = dict()
        for act in self.net_transition_labels:
            for role in self.roles.keys():
                if act in self.roles[role][0]:
                    self.act_resources[act] = self.role_resources[role]
        
        if transition_weights == 'equal':
            self.transition_weights = {t: 1 for t in list(self.net.transitions)}
            print('\n', self.transition_weights, '\n')
        if transition_weights == 'frequency':
            if log == []:
                print('Error: Empty event-log.')
            else:
                self.log = log
                self.transition_weights = return_transitions_frequency(self.log, self.net, self.initial_marking, self.final_marking)
                print('\n', self.transition_weights, '\n')
        if transition_weights == 'manually':
            self.transition_weights = dict()
            print('Insert weight for each transition...')
            for t in list(self.net.transitions):
                t_w = float(input(str(t) + ": "))
                self.transition_weights[t] = t_w
            print('\n', self.transition_weights, '\n')

        if transition_weights == 'data_attributes':
            self.data_attributes_categorical = categorical_attributes
            self.scaler = scaler
            if self.scaler:
                self.scaler_params = return_scaler_params(self.net, self.data_attributes, self.data_attributes_categorical)
            else:
                self.scaler_params = None
            if log == []:
                print('Error: Empty event-log.')
            else:
                self.log = log
                if data_attributes == []:
                    print('No data_attributes.')
                    self.models_t, self.coefficients = build_models(self.log, self.net, self.initial_marking, self.final_marking, self.scaler, self.data_attributes, self.data_attributes_categorical, self.attr_values_categorical, self.net_transition_labels, self.history_weights, self.scaler_params)
                else:
                    self.distr_data_attr = []
                    for trace in self.log:
                        self.distr_data_attr.append([trace[0][a] for a in data_attributes])
                    self.data_attributes_categorical = categorical_attributes
                    self.attr_values_categorical = dict()
                    for a in self.data_attributes_categorical:
                        self.attr_values_categorical[a] = list(pm4py.get_event_attribute_values(self.log, a).keys())
                    self.models_t, self.coefficients = build_models(self.log, self.net, self.initial_marking, self.final_marking, self.scaler, self.data_attributes, self.data_attributes_categorical, self.attr_values_categorical, self.net_transition_labels, self.history_weights, self.scaler_params)


    def insert_times_and_resources(self, traces_times):

        exec_times_act = dict()
        for act in self.net_transition_labels:
            exec_times_act[act] = list(sample_time(self.exec_distr[act], self.count_activities_sim[act]))

        simulated_events = []
        current_time = self.starting_time
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
            while i+j not in case_ids_active and i+j<self.n_sim:
                j += 1
            if i+j<self.n_sim:
                traces_times[i+j][-1] = True
            act = en_t[1][0]
            del traces_times[i][0][0]

            resources = self.act_resources[act]
            available_resource = list(self.resource_availability[resources][(self.resource_availability[resources]<current_time)].index)
            if available_resource:
                resource = random.choices(available_resource)[0]
            else:
                resource_role_availability = {r: self.resource_availability[r] for r in resources}
                resource = min(resource_role_availability, key=resource_role_availability.get)
                current_time = resource_role_availability[resource]

            start_time = return_time_from_calendar(current_time, self.calendars[resource])
            ex_time = int(exec_times_act[act].pop())
            end_time = start_time + timedelta(seconds=ex_time)
            traces_times[i][-2] = end_time
            if not traces_times[i][0]:
                del traces_times[i]
                case_ids_active.remove(i)
            self.resource_availability[resource] = end_time
            current_time = end_time

            simulated_events.append((i, act, start_time, end_time, resource))

        log_data = pd.DataFrame(simulated_events, columns=['case:concept:name', 'concept:name', 'start:timestamp', 'time:timestamp', 'org:resource'])
        log_data.sort_values(by='time:timestamp', inplace=True)
        log_data.index = range(len(log_data))

        return log_data


    def simulate_trace(self):

        # self.trace_time = self.current_time
        
        if not self.history_weights:
            trace_sim = []
            tkns = list(self.initial_marking)
            enabled_transitions = return_enabled_transitions(self.net, tkns)
            t_fired = return_fired_transition(self.transition_weights, enabled_transitions)
            
            if t_fired.label:# and (t_fired.label not in ['<START>', '<END>']):
                trace_sim.append(t_fired.label)
                self.count_activities_sim[t_fired.label] += 1

            tkns = update_markings(tkns, t_fired)
            while set(tkns) != set(self.final_marking):
                enabled_transitions = return_enabled_transitions(self.net, tkns)
                t_fired = return_fired_transition(self.transition_weights, enabled_transitions)
                if t_fired.label:# and (t_fired.label not in ['<START>', '<END>']):
                    trace_sim.append(t_fired.label)
                    self.count_activities_sim[t_fired.label] += 1
                tkns = update_markings(tkns, t_fired)

            return [trace_sim, self.arrival_time, False]
        
        else:
            trace_sim = []
            self.transition_weights = dict()
            tkns = list(self.initial_marking)
            enabled_transitions = return_enabled_transitions(self.net, tkns)

            if self.data_attributes:
                x = random.sample(self.distr_data_attr, k=1)
                self.x_attr.append(x[0])
            else:
                x = [[]]
            x_history = {t_l: 0 for t_l in self.net_transition_labels}
            X = x[0] + list(x_history.values())
            dict_x = dict(zip(self.data_attributes + self.net_transition_labels, X))
            # del dict_x['<START>']
            # del dict_x['<END>']
            if self.scaler:
                for c in self.data_attributes:
                    if (c not in self.data_attributes_categorical):
                        if self.scaler_params[c][0] != self.scaler_params[c][1]:
                            dict_x[c] = (dict_x[c] - self.scaler_params[c][0]) / (self.scaler_params[c][1] - self.scaler_params[c][0])
            for a in self.data_attributes_categorical:
                for v in self.attr_values_categorical[a]:
                    dict_x[a+'_'+v] = (dict_x[a] == v)*1
                del dict_x[a]
            for t in self.net.transitions:
                if type(self.models_t[t]) == LogisticRegression:
                    self.transition_weights[t] = compute_proba(self.models_t, t, list(dict_x.values()))
                else:
                    self.transition_weights[t] = 1
            t_fired = return_fired_transition(self.transition_weights, enabled_transitions)
            if t_fired.label:# and (t_fired.label not in ['<START>', '<END>']):
                trace_sim.append(t_fired.label)
                self.count_activities_sim[t_fired.label] += 1
            tkns = update_markings(tkns, t_fired)
            while set(tkns) != set(self.final_marking):
                if t_fired.label:# and (t_fired.label not in ['<START>', '<END>']):
                    if self.history_weights == 'count':
                        dict_x[t_fired.label] += 1
                        if self.scaler:
                            if self.scaler_params[t_fired.label][0] != self.scaler_params[t_fired.label][1]:
                                dict_x[t_fired.label] = (dict_x[t_fired.label] - self.scaler_params[t_fired.label][0]) / (self.scaler_params[t_fired.label][1] - self.scaler_params[t_fired.label][0])
                    if self.history_weights == 'binary':
                        dict_x[t_fired.label] = 1
                for t in self.net.transitions:
                    if type(self.models_t[t]) == LogisticRegression:
                        self.transition_weights[t] = compute_proba(self.models_t, t, list(dict_x.values()))
                    else:
                        self.transition_weights[t] = 1
                enabled_transitions = return_enabled_transitions(self.net, tkns)
                t_fired = return_fired_transition(self.transition_weights, enabled_transitions)
                if t_fired.label:# and (t_fired.label not in ['<START>', '<END>']):
                    trace_sim.append(t_fired.label)
                    self.count_activities_sim[t_fired.label] += 1
                tkns = update_markings(tkns, t_fired)

            return [trace_sim, self.arrival_time, False]


    def simulate(self, n_istances, remove_head_tail = 0.2):

        self.n_sim = int(n_istances/(1-remove_head_tail))

        self.current_time = self.starting_time
        arrival_times_diff = list(sample_time(self.arrival_time_distr, N=self.n_sim))
        self.arrival_times = []
        for t in arrival_times_diff:
            self.arrival_times.append(self.current_time)
            self.current_time = self.current_time + timedelta(seconds=int(t))
            self.current_time = return_time_from_calendar(self.current_time, self.arrival_calendar)

        self.count_activities_sim = {a: 0 for a in self.net_transition_labels}

        if (self.data_attributes) and (not self.history_weights):
            transition_weights_list = dict()
            try:
                self.x_attr = random.sample(self.distr_data_attr, k=self.n_sim)
            except ValueError:
                self.x_attr = random.sample(self.distr_data_attr, k=len(self.distr_data_attr))
                self.x_attr.extend(random.sample(self.distr_data_attr, k=self.n_sim-len(self.distr_data_attr)))
            df_x = pd.DataFrame(self.x_attr, columns=self.data_attributes)
            if self.scaler:
                for c in list(df_x.columns):
                    if c != 'class' and (c not in self.data_attributes_categorical):
                        if self.scaler_params[c][0] != self.scaler_params[c][1]:
                            df_x[c] = (df_x[c] - self.scaler_params[c][0]) / (self.scaler_params[c][1] - self.scaler_params[c][0])
                
            for a in self.data_attributes_categorical:
                for v in self.attr_values_categorical[a]:
                    df_x[a+'_'+v] = (df_x[a] == v)*1
                del df_x[a]

            for t in self.net.transitions:
                if type(self.models_t[t]) == LogisticRegression:
                    transition_weights_list[t] = list(self.models_t[t].predict_proba(df_x)[:,1])
                else:
                    transition_weights_list[t] = [1]*self.n_sim

        simulated_traces = dict()

        for i in tqdm(range(self.n_sim)):
            if self.data_attributes and (not self.history_weights):
                self.transition_weights = {t: transition_weights_list[t][i] for t in self.net.transitions}
            self.arrival_time = self.arrival_times[i]
            trace = self.simulate_trace()
            # simulated_traces.extend([(str(i + 1), e) for e in ['<START>'] + trace])
            simulated_traces[i] = trace
        simulated_traces[0][-1] = True
        log_data = self.insert_times_and_resources(simulated_traces)

        for n, attr in enumerate(self.data_attributes):
            log_data[attr] = log_data['case:concept:name'].apply(lambda x: self.x_attr[x][n])

        return log_data