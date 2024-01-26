import pm4py
from src.temporal_utils import n_to_weekday


def find_roles(log):
    roles_pm4py = pm4py.discover_organizational_roles(log)
    roles = dict()
    for i in range(len(roles_pm4py)):
        roles['ROLE'+str(i)] =  (roles_pm4py[i][0], len(roles_pm4py[i][1]))
    return roles


def create_resources(roles):
    return {role: [role+'_'+str(i) for i in range(roles[role][1])] for role in roles.keys()}


def find_calendars(roles, mode='24/7', log=None):
    """
    {role: {WEEKDAY: (sH,eH)}}
    se quel weekday non si lavora mettere None
    """

    weekday_labels = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    if mode == '24/7':
        return {role: {wd: (0,23) for wd in weekday_labels} for role in roles.keys()}
    
    if mode == 'manual':
        calendars = dict()
        for role in roles:
            calendars[role] = dict()
            for wd in weekday_labels:
                try:
                    start_hour = int(input(role + ' ' + wd + ' ' + 'Start Hour: '))
                    end_hour = int(input(role + ' ' + wd + ' ' + 'Final Hour: '))
                    calendars[role][wd] = (start_hour, end_hour)
                except:
                    calendars[role][wd] = None
        return calendars
    
    if mode == 'discover':
        calendars = dict()
        if not log:
            print('Error: Insert Event Log for discover mode.')
            print('Mode set to manual')
            return find_calendars(roles, mode='manual')
        log_df = pm4py.convert_to_dataframe(log)
        log_df['weekday'] = log_df['time:timestamp'].apply(lambda x: n_to_weekday(x.weekday()))
        log_df['hour'] = log_df['time:timestamp'].apply(lambda x: x.hour)
        for role in roles.keys():
            calendars[role] = dict()
            role_acts = roles[role][0]
            log_df_act_roles = log_df[log_df['concept:name'].isin(role_acts)]
            for wd in weekday_labels:
                log_df_act_roles_wd = log_df_act_roles[log_df_act_roles['weekday'] == wd]
                if len(log_df_act_roles_wd) == 0:
                    calendars[role][wd] = None
                calendars[role][wd] = (log_df_act_roles_wd['hour'].min(), log_df_act_roles_wd['hour'].max())
        return calendars