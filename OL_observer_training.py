# -*- coding: utf-8 -*-
"""
Created on Sun May 25 19:33:44 2025

@author: shukl
"""

#%% Import libraries

import pandas as pd
import re
import os
import glob
from datetime import datetime
import numpy as np
import smoothfit
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

#%% Function for parsing behavioral events from .stateScriptLog files (single rat experiments)

def parse_leader_and_follower_events(file_path):
    """
    Parses logfile of behavioral poke/unpoke events,
    outputs two DataFrames: one for the leader, one for the follower.
    For the follower, goal_trajectory is from follower's current well to leader's current well.
    """

    with open(file_path, 'r') as f:
        log_lines = f.readlines()

    # Patterns
    leader_poke_pattern = re.compile(r'(\d+)\s+Leader poke in well (\d) - (LEFT|CENTER|RIGHT)')
    leader_unpoke_pattern = re.compile(r'(\d+)\s+Leader unpoke in well (\d) - (LEFT|CENTER|RIGHT)')
    leader_reward_pattern = re.compile(r'(\d+)\s+leaderRewardCount = (\d+)')
    
    follower_poke_pattern = re.compile(r'(\d+)\s+Follower poke in well (\d) - (LEFT|CENTER|RIGHT)')
    follower_unpoke_pattern = re.compile(r'(\d+)\s+Follower unpoke in well (\d) - (LEFT|CENTER|RIGHT)')
    follower_reward_pattern = re.compile(r'(\d+)\s+followerRewardCount = (\d+)')

    def extract_entries(poke_pattern, unpoke_pattern, reward_pattern):
        entries = []
        reward_timestamps = []

        for line in log_lines:
            line = line.strip()

            for pattern, action in [(poke_pattern, "poke"), (unpoke_pattern, "unpoke")]:
                match = pattern.match(line)
                if match:
                    timestamp, well_num, well_label = match.groups()
                    entries.append({
                        "timestamp": int(timestamp),
                        "action": action,
                        "well_num": int(well_num),
                        "well_label": well_label
                    })

            reward_match = reward_pattern.match(line)
            if reward_match:
                reward_ts = int(reward_match.group(1))
                reward_timestamps.append(reward_ts)

        df = pd.DataFrame(entries).sort_values("timestamp").reset_index(drop=True)

        # Remove redundant pokes (poke followed by poke)
        redundant_indices = []
        i = 1
        while i < len(df):
            if df.loc[i - 1, "action"] == "poke" and df.loc[i, "action"] == "poke":
                redundant_indices.append(i)
                for j in range(i + 1, len(df)):
                    if df.loc[j, "action"] == "unpoke" and df.loc[j, "well_num"] == df.loc[i, "well_num"]:
                        redundant_indices.append(j)
                        break
            i += 1
        df = df.drop(index=redundant_indices).reset_index(drop=True)

        return df, reward_timestamps

    # Extract for leader and follower
    df_leader, leader_rewards = extract_entries(leader_poke_pattern, leader_unpoke_pattern, leader_reward_pattern)
    df_follower_followinger, follower_rewards = extract_entries(follower_poke_pattern, follower_unpoke_pattern, follower_reward_pattern)

    # --- LEADER GOAL/TRAJECTORY LOGIC ---
    goal_wells = []
    trajectories = []
    goal_trajectories = []
    visit_sequence = []
    last_well = None

    for idx, row in df_leader.iterrows():
        current = row["well_label"]
        traj = None
        goal_traj = None
        goal = None

        if row["action"] == "poke":
            if current == "CENTER" and last_well in ("LEFT", "RIGHT"):
                goal_traj = traj = "inbound"
            elif (current == "LEFT" and last_well == "RIGHT") or (current == "RIGHT" and last_well == "LEFT"):
                goal_traj = "inbound"
                traj = "outbound"
            elif current in ("LEFT", "RIGHT") and last_well == "CENTER":
                goal_traj = traj = "outbound"

            last_well = current

            if visit_sequence:
                prev = visit_sequence[-1]
                if current == "CENTER" and prev in ("LEFT", "RIGHT"):
                    goal = "RIGHT" if prev == "LEFT" else "LEFT"
                elif current in ("LEFT", "RIGHT") and prev == "CENTER":
                    goal = "CENTER"
            visit_sequence.append(current)

        trajectories.append(traj)
        goal_trajectories.append(goal_traj)
        goal_wells.append(goal)

    df_leader["goal_well"] = goal_wells
    df_leader["trajectory"] = trajectories
    df_leader["goal_trajectory"] = goal_trajectories
    df_leader[["goal_well", "trajectory", "goal_trajectory"]] = df_leader[["goal_well", "trajectory", "goal_trajectory"]].ffill()

    # Annotate rewards (leader)
    df_leader["rewarded"] = 0
    poke_indices = df_leader[df_leader["action"] == "poke"].index
    for ts in leader_rewards:
        nearest_idx = (df_leader.loc[poke_indices, "timestamp"] - ts).abs().idxmin()
        df_leader.at[nearest_idx, "rewarded"] = 1

    # --- FOLLOWER GOAL/TRAJECTORY LOGIC ---
    # df_follower_followinger["trajectory"] = None
    # df_follower_followinger["goal_trajectory"] = None
    df_follower_followinger["goal_well"] = None
    df_follower_followinger["rewarded"] = 0
    
    last_follower_well = None
    for idx, row in df_follower_followinger.iterrows():
        if row["action"] == "poke":
            current_well = row["well_label"]
            follower_time = row["timestamp"]
    
            # Get leader's most recent poke before this follower poke
            leader_prior = df_leader[(df_leader["action"] == "poke") & (df_leader["timestamp"] <= follower_time)]
            if not leader_prior.empty:
                leader_current_well = leader_prior.iloc[-1]["well_label"]
                df_follower_followinger.at[idx, "goal_well"] = leader_current_well
                # df_follower_followinger.at[idx, "goal_trajectory"] = f"{current_well}->{leader_current_well}"
    
            # if last_follower_well:
            #     df_follower_followinger.at[idx, "trajectory"] = f"{last_follower_well}->{current_well}"
    
            last_follower_well = current_well

    # Annotate rewards (follower)
    poke_indices = df_follower_followinger[df_follower_followinger["action"] == "poke"].index
    for ts in follower_rewards:
        nearest_idx = (df_follower_followinger.loc[poke_indices, "timestamp"] - ts).abs().idxmin()
        df_follower_followinger.at[nearest_idx, "rewarded"] = 1

    return df_leader, df_follower_followinger



#%% Function for extracting transition sequences 

def combine_consecutive_wells(df: pd.DataFrame) -> pd.DataFrame:
    """
    Function to extract the transition sequence of well
    visits from rat poking data. It also calculates duration
    a rat remains / dwells at a well.
    
    Args:
        df: DataFrame with poke/unpoke events
        
    Returns:
        DataFrame with combined consecutive well visits
    """
    if df.empty:
        return df
    
    # Filter only poke events for transition analysis
    poke_df = df[df['action'] == 'poke'].copy()
    
    if poke_df.empty:
        return poke_df
    
    # Identify groups of consecutive rows with the same well_num
    poke_df['group'] = (poke_df['well_num'] != poke_df['well_num'].shift()).cumsum()
    
    # Build aggregation dictionary dynamically based on available columns
    agg_dict = {
        'timestamp': 'first',
        'well_num': 'first',
        'well_label': 'first',
        'rewarded': 'first'  # Sum rewards in case multiple pokes in same well were rewarded
    }
    
    # Add optional columns if they exist
    if 'goal_well' in poke_df.columns:
        agg_dict['goal_well'] = 'first'
    
    if 'trajectory' in poke_df.columns:
        agg_dict['trajectory'] = 'first'
    
    if 'goal_trajectory' in poke_df.columns:
        agg_dict['goal_trajectory'] = 'first'
    
    # Group by consecutive wells and aggregate
    result = (
        poke_df.groupby('group')
        .agg(agg_dict)
        .reset_index(drop=True)
    )
    
    # Calculate dwell duration if we have enough data
    if len(result) > 1:
        # Duration is time until next well visit (or end of session for last visit)
        result['dwell_duration_ms'] = result['timestamp'].shift(-1) - result['timestamp']
        # For the last row, we can't calculate duration to next well
        result.loc[result.index[-1], 'dwell_duration_ms'] = None
    else:
        result['dwell_duration_ms'] = None
    
    return result


#%% Function for extracting trialwise follower dataframe

def combine_consecutive_goalWells(df: pd.DataFrame) -> pd.DataFrame:
    """
    Function to extract the transition sequence of well
    visits from rat poking data. It also calculates duration
    a rat remains / dwells at a well.
    
    Args:
        df: DataFrame with poke/unpoke events
        
    Returns:
        DataFrame with combined consecutive well visits
    """
    if df.empty:
        return df
    
    # Filter only poke events for transition analysis
    poke_df = df[df['action'] == 'poke'].copy()
    
    if poke_df.empty:
        return poke_df
    
    # Identify groups of consecutive rows with the same well_num
    poke_df['group'] = (poke_df['goal_well'] != poke_df['goal_well'].shift()).cumsum()
    
    # Build aggregation dictionary dynamically based on available columns
    agg_dict = {
        'timestamp': 'first',
        'well_num': 'first',
        'well_label': 'first',
        'goal_well': 'first',
        'rewarded': 'first'  # Sum rewards in case multiple pokes in same well were rewarded
    }
    
    # # Add optional columns if they exist
    # if 'goal_well' in poke_df.columns:
    #     agg_dict['goal_well'] = 'first'
    
    if 'trajectory' in poke_df.columns:
        agg_dict['trajectory'] = 'first'
    
    if 'goal_trajectory' in poke_df.columns:
        agg_dict['goal_trajectory'] = 'first'
    
    # Group by consecutive wells and aggregate
    result = (
        poke_df.groupby('group')
        .agg(agg_dict)
        .reset_index(drop=True)
    )
    
    # Calculate dwell duration if we have enough data
    if len(result) > 1:
        # Duration is time until next well visit (or end of session for last visit)
        result['dwell_duration_ms'] = result['timestamp'].shift(-1) - result['timestamp']
        # For the last row, we can't calculate duration to next well
        result.loc[result.index[-1], 'dwell_duration_ms'] = None
    else:
        result['dwell_duration_ms'] = None
    
    return result


#%% Test a single file

file_path = "E:/Jadhav lab data/Behavior/Observational learning/Observer training/05-21-2025/log05-21-2025(10-OL1-OL2).stateScriptLog"

df_leader, df_follower= parse_leader_and_follower_events(file_path)

df_lead = combine_consecutive_wells(df_leader)
df_follower_transitions = combine_consecutive_wells(df_follower)


#%% Batch process data from cohort
 
# Define the dtype for the structured array
dtype = np.dtype([
    ('name', 'U100'),      # File name
    ('folder', 'U255'),    # Folder path
    ('date', 'U50'),       # Extracted date
    # ('cohortnum', 'U50'),  # Cohort number
    ('runum', 'U50'),      # Run number
    ('ratnums', 'U50'),    # Rat numbers as string
    ('ratnames', 'O'),     # List of rat names (object)
    ('ratsamples', 'O'),   # Tuple of two DataFrames (object)
    ('leaderData',  'O'),
    ('followerData', 'O'),
    ('followerFilteredData', 'O'), # omits trials where follower doesn't transition
    ('followerTransitionData', 'O'),
    ('reward', 'O'),       # rewards for rats
    ('nTransitions', 'O'), # transitions for rats
    ('perf', 'O'),       # performance metric for pair
    ('duration', 'U50'),   # session duration
    ('transition sequence', 'O') # transition sequence
])


# Function to extract data from filename
def parse_filename(filename):
    """
    Function to extract data from filename
    """
    match = re.search(r"log(\d{2}-\d{2}-\d{4})\((\d+)-([A-Z]+\d+)-([A-Z]+\d+)\)\.stateScriptLog", filename)
    if match:
        date, runum, rat1, rat2 = match.groups()
        return date, runum, f"{rat1},{rat2}", [rat1, rat2]
    return None, None, None, None

# Function to load .stateScriptLog files
def process_social_stateScriptLogs(base_dir):
    
    """
    Function to extract data from filename
    """
    
    struct_data = []
    
    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.endswith(".stateScriptLog"):
                full_path = os.path.join(root, file)
                date, runum, ratnums, ratnames = parse_filename(file)

                # Dummy DataFrames for example
                df_leader, df_follower = parse_leader_and_follower_events(full_path)
                
                
                
                df_lead = combine_consecutive_wells(df_leader)

                # List of columns to drop if they exist
                cols_to_drop = ['trajectory', 'goal_trajectory', 'goal_well']
                existing_cols = [col for col in cols_to_drop if col in df_lead.columns]
                
                if existing_cols:
                    df_lead = df_lead.drop(columns=existing_cols)

                    
                df_follower_following = combine_consecutive_goalWells(df_follower)
                df_follower_transitions = combine_consecutive_wells(df_follower)
                df_follower_transitions = df_follower_transitions.drop(columns=['goal_well'])
                
                
                threshold = 5000  # desired threshold (2s in this case)
                df_lead = df_lead[df_lead['dwell_duration_ms'] >= threshold].reset_index(drop=True)
                df_follower_following = df_follower_following[df_follower_following['dwell_duration_ms'] >= threshold].reset_index(drop=True)
                df_follower_transitions = df_follower_transitions[df_follower_transitions['dwell_duration_ms'] >= threshold].reset_index(drop=True)

                # Filter to keep only rows where well_label changes from the previous row
                follower_filtered_df = df_follower_following[df_follower_following['well_label'] != df_follower_following['well_label'].shift()]

                leader_rewards = df_lead['rewarded'].sum()
                follower_rewards = df_follower_following['rewarded'].sum()
                
                session_length = (max(df_lead['timestamp'].max(), df_follower_following['timestamp'].max()) 
                                  - min(df_lead['timestamp'].min(), df_follower_following['timestamp'].min())) / (60*1000)
                
                
                seq1 = df_lead['well_num'].astype(int).astype(str).str.cat()
                seq2 = df_follower_transitions['well_num'].astype(int).astype(str).str.cat()
                seq = [seq1, seq2]
                
                leader_transitions = len(df_lead)
                follower_transitions = len(df_follower_following)

                
                if np.isnan(leader_transitions) or leader_transitions == 0:
                    print(f"Warning: Division issue in file {file}")
                    leader_perf = np.nan
                else:
                    leader_perf = 100 * (leader_rewards / leader_transitions)
                    
                if np.isnan(follower_transitions) or follower_transitions == 0:
                    print(f"Warning: Division issue in file {file}")
                    follower_perf = np.nan
                else:
                    follower_perf = 100 * (follower_rewards / follower_transitions)



                struct_data.append((
                    file,  # name
                    root,  # folder
                    date,  # date
                    # root.split("/")[-2],  # cohortnum (assuming one level up)
                    runum,  # runum
                    ratnums,  # ratnums
                    ratnames,  # ratnames
                    (df_leader, df_follower),  # ratsamples (tuple of DataFrames)
                    df_lead, 
                    df_follower_following,
                    follower_filtered_df,
                    df_follower_transitions,
                    (leader_rewards, follower_rewards),  # rewards for rats
                    (leader_transitions, follower_transitions),        # transitions for rats
                    (leader_perf, follower_perf),                # pair performance
                    session_length,      # session duration
                    seq                 # transition sequences
                ))
                
                
                # Initialize an empty list to store the data
                structured_data = []

                for entry in struct_data:
                    data_entry = {
                        'name': entry[0],
                        'folder': entry[1],
                        'date': entry[2],
                        # 'cohortnum': entry[3],
                        'runum': entry[3],
                        'ratnum': entry[4],
                        'ratnames': entry[5],
                        'ratsamples': entry[6],  # DataFrame
                        'leaderData': entry[7],
                        'followerData': entry[8],
                        'followerFilteredData': entry[9],
                        'followerTransitionData': entry[10],
                        'reward': entry[11],
                        'nTransitions': entry[12],
                        'perf': entry[13],
                        'duration': entry[14],
                        'transition_sequence': entry[15]  # List
                        # 'triplet counts': entry[14]  # Dict
                    }
                    structured_data.append(data_entry)
    
    return structured_data

# Define base directory
base_dir = r"E:/Jadhav lab data/Behavior/Observational learning/OL pilot/Data/Observer training"

# Load the structured array
data = process_social_stateScriptLogs(base_dir)

# Find indices with date as None
index = []
for i in range(len(data)):  
   if data[i]['date'] is None:
       index.append(i)

# Exclude those entries 
new_data = [ele for idx, ele in enumerate(data) if idx not in index]

data = new_data
# Sort by converted date and integer run number
sorted_data = sorted(data, key=lambda x: (datetime.strptime(x['date'], "%m-%d-%Y"), int(x['runum'])))

#%% Parse performance, reward rate, etc. for each rat separately

# Dictionary to store performance data for each individual rat
ratwise_perf = {}
ratwise_reward_rate = {}
ratwise_transition_rate = {}


for entry in sorted_data:
    rat1, rat2 = entry['ratnames']  # entry['ratnames'] contains ['Rat1', 'Rat2']
    perf = entry['perf']  # performance value for that session
    duration = float(entry['duration'])

    # Extract values separately for each rat
    rewards = entry['reward']
    transitions = entry['nTransitions']


    for i, rat in enumerate([rat1, rat2]):  # Iterate over both rats with index
        if rat not in ratwise_perf:
            ratwise_perf[rat] = []
            ratwise_reward_rate[rat] = []
            ratwise_transition_rate[rat] = []

        # Compute rates while avoiding division by zero
        reward_rate = rewards[i] / duration #if perf else np.nan
        transition_rate = transitions[i] / duration 

        ratwise_perf[rat].append(perf[i])  # Assuming performance is the same for both rats
        ratwise_reward_rate[rat].append(reward_rate)  # Extract individual reward rate
        ratwise_transition_rate[rat].append(transition_rate)  # Extract individual transition rate


        
# Convert to NumPy arrays
ratwise_perf = {rat: np.array(perfs, dtype = float) for rat, perfs in ratwise_perf.items()}
ratwise_reward_rate = {rat: np.array(rates, dtype = float) for rat, rates in ratwise_reward_rate.items()}
ratwise_transition_rate = {rat: np.array(rates, dtype = float) for rat, rates in ratwise_transition_rate.items()}

#%% Function for assigning W rule logic to rat's trajectories

def assign_w_rule_logic(df):
    df['last_well'] = None
    df['sec_last_well'] = None
    df['last_well'] = df['well_label'].shift(1)
    df['sec_last_well'] = df['last_well'].shift(1)
    df['goal_trajectory'] = None
    df['trajectory'] = None
    df['W rule reward'] = None

    trajectories = []
    goal_trajectories = []
    rewards = []

    for idx, row in df.iterrows():
        current = row["well_label"]
        last_well = row["last_well"]
        sec_last_well = row["sec_last_well"]
        traj = None
        goal_traj = None

        # Goal trajectory logic
        if current == "CENTER" and last_well in ("LEFT", "RIGHT"):
            goal_traj = "inbound"
            traj = "inbound correct"
            reward = 1

        elif current == "CENTER" and last_well in ("LEFT", "RIGHT") and sec_last_well is None:
            goal_traj = "inbound"
            traj = "inbound correct"
            reward = 1

        elif (current == "LEFT" and last_well == "RIGHT") or (current == "RIGHT" and last_well == "LEFT"):
            goal_traj = "inbound"
            traj = "inbound error"
            reward = 0

        elif current == "LEFT" and last_well == "CENTER" and sec_last_well == "RIGHT":
            goal_traj = "outbound left"
            traj = "outbound correct"
            reward = 1

        elif current == "LEFT" and last_well == "CENTER" and sec_last_well is None:
            goal_traj = "outbound left"
            traj = "outbound correct"
            reward = 1

        elif current == "RIGHT" and last_well == "CENTER" and sec_last_well == "LEFT":
            goal_traj = "outbound right"
            traj = "outbound correct"
            reward = 1

        elif current == "RIGHT" and last_well == "CENTER" and sec_last_well is None:
            goal_traj = "outbound right"
            traj = "outbound correct"
            reward = 1

        elif current == "RIGHT" and last_well == "CENTER" and sec_last_well == "RIGHT":
            goal_traj = "outbound left"
            traj = "outbound error"
            reward = 0

        elif current == "LEFT" and last_well == "CENTER" and sec_last_well == "LEFT":
            goal_traj = "outbound right"
            traj = "outbound error"
            reward = 0

        elif last_well is None and sec_last_well is None:
            reward = 1

        trajectories.append(traj)
        goal_trajectories.append(goal_traj)
        rewards.append(reward)

    df["trajectory"] = trajectories
    df["goal_trajectory"] = goal_trajectories
    df["W rule reward"] = rewards

    return df


#%% Check W-rule performance for observers during training

for entry in sorted_data:

    df1 = entry['followerTransitionData']
    df1 = assign_w_rule_logic(df1)
    
    df2 = entry['leaderData']
    df2 = assign_w_rule_logic(df2)

#%% Dictionary to store performance data for each individual rat (follower, W rule)

Wrule_perf = {}
Wrule_reward_rate = {}

for entry in sorted_data:
    rat = entry['ratnames'][1]  # entry['ratnames'] contains ['Rat1', 'Rat2']
    df = entry['followerTransitionData']
    duration = float(entry['duration'])

    rewards = df['W rule reward'].sum()
    transitions = len(df)
    perf = 100 * (rewards / transitions)

    if rat not in Wrule_perf:
        Wrule_perf[rat] = []
        Wrule_reward_rate[rat] = []

    reward_rate = rewards / duration

    Wrule_perf[rat].append(perf)
    Wrule_reward_rate[rat].append(reward_rate)

# Convert to NumPy arrays
Wrule_perf = {rat: np.array(perfs, dtype=float) for rat, perfs in Wrule_perf.items()}
Wrule_reward_rate = {rat: np.array(rates, dtype=float) for rat, rates in Wrule_reward_rate.items()}


#%% Plot performance for demonstrators and observers

import smoothfit 

demonstrators = ['OL1', 'OL3']
observers = ['OL2', 'OL4']

follower_following_perf = {}

# Calculate follower's performance on follow rule
for entry in sorted_data:
    rat = entry['ratnames'][1]  # rat name
    df = entry['followerFilteredData']
    rewards = df['rewarded'].sum()  # Assuming 'rewarded' is boolean or 1/0
    transitions = len(df)
    perf = 100 * (rewards / transitions) if transitions > 0 else np.nan  # Avoid division by zero

    

    if rat not in follower_following_perf:
        follower_following_perf[rat] = []

    follower_following_perf[rat].append(perf)

# Convert to NumPy arrays
follower_following_perf = {rat: np.array(perfs, dtype=float) for rat, perfs in follower_following_perf.items()}


sns.set(style='ticks')
sns.set_context('poster')

lmbda = 5.0e-1

# Use the default matplotlib color cycle
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

plt.figure()

# Plot follow rule observer rats
for i, rat in enumerate(observers):
    color = colors[i % len(colors)]
    y0 = follower_following_perf[rat] # only selecting trials where obsrver made an active choice
    # y0 = ratwise_perf[rat] # # selecting all active and passive trials from observer
    x0 = np.linspace(0.0, len(y0), len(y0))
    plt.plot(x0, y0, "x", color=color)
    basis, coeffs = smoothfit.fit1d(x0, y0, np.min(x0), np.max(x0), 1000, degree=1, lmbda=lmbda)
    plt.plot(basis.mesh.p[0], coeffs[basis.nodal_dofs[0]], "-", color=color, label=f"{rat} (observer)")
    plt.xlabel("Session")
    plt.ylabel("Performance (%)")
    plt.ylim((10, 100))
    plt.legend()
    plt.show()
    
plt.figure()
# Plot Wrule_perf rats
for i, rat in enumerate(observers):
    color = colors[i % len(colors)]
    y0 = Wrule_perf[rat]
    x0 = np.linspace(0.0, len(y0), len(y0))
    plt.plot(x0, y0, "x", color=color)
    basis, coeffs = smoothfit.fit1d(x0, y0, np.min(x0), np.max(x0), 1000, degree=1, lmbda=lmbda)
    plt.plot(basis.mesh.p[0], coeffs[basis.nodal_dofs[0]], "-", color=color, label=f"{rat} (Wrule)")
    plt.xlabel("Session")
    plt.ylabel("Performance (%)")
    plt.ylim((20, 100))
    plt.legend()
    plt.show()
    
plt.figure()
# Plot demonstrator rats
for i, rat in enumerate(demonstrators):
    color = colors[i % len(colors)]
    y0 = ratwise_perf[rat]
    x0 = np.linspace(0.0, len(y0), len(y0))
    plt.plot(x0, y0, "x", color=color)
    basis, coeffs = smoothfit.fit1d(x0, y0, np.min(x0), np.max(x0), 1000, degree=1, lmbda=lmbda)
    plt.plot(basis.mesh.p[0], coeffs[basis.nodal_dofs[0]], "-", color=color, label=f"{rat} (demo)")
    plt.xlabel("Session")
    plt.ylabel("Performance (%)")
    plt.ylim((20, 100))
    plt.legend()
    plt.show()

#%% Get binary vectors for rewarded / unrewarded trials of follower rat (follower, follow rule)

follower_follow_reward_df = {}

for entry in sorted_data:
    rat = entry['ratnames'][1]
    vec = entry['followerFilteredData']['rewarded'].reset_index(drop=True)  # ensure clean indexing

    if rat not in follower_follow_reward_df:
        follower_follow_reward_df[rat] = []

    session_number = len(follower_follow_reward_df[rat])  # Serial session index
    df_session = pd.DataFrame({
        'reward': vec,
        'session': session_number
    })

    follower_follow_reward_df[rat].append(df_session)

# Concatenate into a single DataFrame per rat
for rat in follower_follow_reward_df:
    follower_follow_reward_df[rat] = pd.concat(follower_follow_reward_df[rat], ignore_index=True)

#%% Get binary vectors for rewarded / unrewarded trials (follower, W rule)

follower_reward_df = {}

for entry in sorted_data:
    rat = entry['ratnames'][1]
    vec = entry['followerTransitionData']['W rule reward'].reset_index(drop=True)  # ensure clean indexing

    if rat not in follower_reward_df:
        follower_reward_df[rat] = []

    session_number = len(follower_reward_df[rat])  # Serial session index
    df_session = pd.DataFrame({
        'reward': vec,
        'session': session_number
    })

    follower_reward_df[rat].append(df_session)

# Concatenate into a single DataFrame per rat
for rat in follower_reward_df:
    follower_reward_df[rat] = pd.concat(follower_reward_df[rat], ignore_index=True)

#%% Reward vector by trajectory type (follower, W rule)

follower_reward_by_trajectory = {}
follower_accuracy_by_trajectory = {}

for entry in sorted_data:
    rat = entry['ratnames'][1]
    poke_df = entry['followerTransitionData'].reset_index(drop=True)

    if rat not in follower_reward_by_trajectory:
        follower_reward_by_trajectory[rat] = {'inbound': [], 'outbound': []}
        follower_accuracy_by_trajectory[rat] = {'inbound': [], 'outbound': []}

    # Inbound binary vector
    inbound_binary = poke_df[poke_df['trajectory'].isin(['inbound correct', 'inbound error'])]['trajectory'].map({
        'inbound correct': 1,
        'inbound error': 0
    }).to_numpy()

    # Outbound binary vector
    outbound_binary = poke_df[poke_df['trajectory'].isin(['outbound correct', 'outbound error'])]['trajectory'].map({
        'outbound correct': 1,
        'outbound error': 0
    }).to_numpy()

    # Store binary vectors
    follower_reward_by_trajectory[rat]['inbound'].append(inbound_binary)
    follower_reward_by_trajectory[rat]['outbound'].append(outbound_binary)

    # Store accuracies (mean of binary vectors, avoiding division by zero)
    follower_accuracy_by_trajectory[rat]['inbound'].append(np.mean(inbound_binary) if len(inbound_binary) > 0 else np.nan)
    follower_accuracy_by_trajectory[rat]['outbound'].append(np.mean(outbound_binary) if len(outbound_binary) > 0 else np.nan)



follower_reward_dfs_by_traj = {}

for rat, traj_dict in follower_reward_by_trajectory.items():
    follower_reward_dfs_by_traj[rat] = {}
    for traj_type, sessions in traj_dict.items():
        rows = []
        for session_idx, rewards in enumerate(sessions):
            for reward in rewards:
                rows.append({
                    'session': session_idx,
                    'rewarded': reward
                })
        follower_reward_dfs_by_traj[rat][traj_type] = pd.DataFrame(rows)


       
#%% Plot accuracy for indivdual rats (follower, W rule)

trajectory_types = ['inbound', 'outbound']

# Use the default matplotlib color cycle
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

lmbda = 3.0e-1
for traj_type in trajectory_types:
    plt.figure()
    handles = []  # Store legend handles
    labels = []   # Store corresponding labels
    
    for i, rat in enumerate(follower_accuracy_by_trajectory):
        if traj_type in follower_accuracy_by_trajectory[rat]:
            perf = follower_accuracy_by_trajectory[rat][traj_type]
            y0 = perf
            x0 = np.linspace(0.0, len(y0), len(y0))

            # Pick color based on rat index
            color = colors[i % len(colors)]

            # Raw data points
            scatter_handle = plt.plot(x0, y0, "x", color=color)[0]

            # Smooth fit
            basis, coeffs = smoothfit.fit1d(x0, y0, np.min(x0), np.max(x0), 1000, degree=1, lmbda=lmbda)
            line_handle = plt.plot(basis.mesh.p[0], coeffs[basis.nodal_dofs[0]], "-", color=color)[0]

            handles.append(scatter_handle)
            labels.append(f'{rat}')

    plt.title(f'{traj_type.capitalize()}')
    plt.xlabel('Session')
    plt.ylabel('Proportion Correct')
    plt.ylim(0, 1.05)
    plt.yticks(np.arange(0.0, 1.0, 0.25))
    plt.legend(handles, labels, title='Rat')
    plt.tight_layout()
    plt.show()


#%% Get binary vectors for rewarded / unrewarded trials of demonstrator rat (W rule)

demonstrator_reward_df = {}

for entry in sorted_data:
    rat = entry['ratnames'][0]
    vec = entry['leaderData']['W rule reward'].reset_index(drop=True)  # ensure clean indexing

    if rat not in demonstrator_reward_df:
        demonstrator_reward_df[rat] = []

    session_number = len(demonstrator_reward_df[rat])  # Serial session index
    df_session = pd.DataFrame({
        'reward': vec,
        'session': session_number
    })

    demonstrator_reward_df[rat].append(df_session)

# Concatenate into a single DataFrame per rat
for rat in demonstrator_reward_df:
    demonstrator_reward_df[rat] = pd.concat(demonstrator_reward_df[rat], ignore_index=True)

#%% Reward vector by trajectory type (demonstrator rats)

demonstrator_reward_by_trajectory = {}
demonstrator_accuracy_by_trajectory = {}

for entry in sorted_data:
    rat = entry['ratnames'][0]
    poke_df = entry['leaderData'].reset_index(drop=True)

    if rat not in demonstrator_reward_by_trajectory:
        demonstrator_reward_by_trajectory[rat] = {'inbound': [], 'outbound': []}
        demonstrator_accuracy_by_trajectory[rat] = {'inbound': [], 'outbound': []}

    # Inbound binary vector
    inbound_binary = poke_df[poke_df['trajectory'].isin(['inbound correct', 'inbound error'])]['trajectory'].map({
        'inbound correct': 1,
        'inbound error': 0
    }).to_numpy()

    # Outbound binary vector
    outbound_binary = poke_df[poke_df['trajectory'].isin(['outbound correct', 'outbound error'])]['trajectory'].map({
        'outbound correct': 1,
        'outbound error': 0
    }).to_numpy()

    # Store binary vectors
    demonstrator_reward_by_trajectory[rat]['inbound'].append(inbound_binary)
    demonstrator_reward_by_trajectory[rat]['outbound'].append(outbound_binary)

    # Store accuracies (mean of binary vectors, avoiding division by zero)
    demonstrator_accuracy_by_trajectory[rat]['inbound'].append(np.mean(inbound_binary) if len(inbound_binary) > 0 else np.nan)
    demonstrator_accuracy_by_trajectory[rat]['outbound'].append(np.mean(outbound_binary) if len(outbound_binary) > 0 else np.nan)



demonstrator_reward_dfs_by_traj = {}

for rat, traj_dict in demonstrator_reward_by_trajectory.items():
    demonstrator_reward_dfs_by_traj[rat] = {}
    for traj_type, sessions in traj_dict.items():
        rows = []
        for session_idx, rewards in enumerate(sessions):
            for reward in rewards:
                rows.append({
                    'session': session_idx,
                    'rewarded': reward
                })
        demonstrator_reward_dfs_by_traj[rat][traj_type] = pd.DataFrame(rows)


#%% Plot accuracy for indivdual demonstrator rats (trajectory wise)

trajectory_types = ['inbound', 'outbound']

# Use the default matplotlib color cycle
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

lmbda = 3.0e-1
for traj_type in trajectory_types:
    plt.figure()
    handles = []  # Store legend handles
    labels = []   # Store corresponding labels
    
    for i, rat in enumerate(demonstrator_accuracy_by_trajectory):
        if traj_type in demonstrator_accuracy_by_trajectory[rat]:
            perf = demonstrator_accuracy_by_trajectory[rat][traj_type]
            y0 = perf
            x0 = np.linspace(0.0, len(y0), len(y0))

            # Pick color based on rat index
            color = colors[i % len(colors)]

            # Raw data points
            scatter_handle = plt.plot(x0, y0, "x", color=color)[0]

            # Smooth fit
            basis, coeffs = smoothfit.fit1d(x0, y0, np.min(x0), np.max(x0), 1000, degree=1, lmbda=lmbda)
            line_handle = plt.plot(basis.mesh.p[0], coeffs[basis.nodal_dofs[0]], "-", color=color)[0]

            handles.append(scatter_handle)
            labels.append(f'{rat}')

    plt.title(f'{traj_type.capitalize()}')
    plt.xlabel('Session')
    plt.ylabel('Proportion Correct')
    plt.ylim(0, 1.05)
    plt.yticks(np.arange(0.0, 1.01, 0.25))
    plt.legend(handles, labels, title='Rat')
    plt.tight_layout()
    plt.show()


#%% faster and more efficient implementation of a state-space model for

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy import integrate
from numba import jit  # For JIT compilation
from tqdm import tqdm  # For progress tracking

# JIT compile the state update function for faster execution
@jit(nopython=True)
def get_state_update(x_pred, v_pred, b0, n):
    """
    Newton-Raphson algorithm to update the state - optimized with Numba
    """
    M = 500  # maximum iterations
    
    it = np.zeros(M)
    it[0] = x_pred
    
    for i in range(M-1):
        exp_term = np.exp(b0 + it[i])
        denom = (1 + exp_term)
        
        func = it[i] - x_pred - v_pred * (n - exp_term / denom)
        df = 1 + v_pred * exp_term / (denom ** 2)
        it[i + 1] = it[i] - func / df
        
        if abs(it[i + 1] - it[i]) < 1e-14:
            return it[i + 1]
    
    # If we didn't converge, return the last value instead of raising an error
    return it[M-1]

# Optimize confidence limit calculation
@jit(nopython=True)
def calculate_integrand(p, v, b0, x):
    """Calculate the integrand for confidence limits"""
    valid_indices = (p > 0) & (p < 1)
    result = np.zeros_like(p)
    
    # Use log for numerical stability
    log_p = np.log(p[valid_indices])
    log_1_minus_p = np.log(1 - p[valid_indices])
    
    # Calculate the integrand
    exponent = (-1) / (2 * v) * (log_p - log_1_minus_p - b0 - x) ** 2
    result[valid_indices] = 1 / (np.sqrt(2 * np.pi * v) * p[valid_indices] * (1 - p[valid_indices])) * np.exp(exponent)
    
    return result

def get_pk_conf_lims(v, b0, x):
    """
    Calculate confidence limits for p_k - more efficient implementation
    Uses adaptive step sizes for the p values
    """
    # Use fewer points for faster integration, but concentrate them where needed
    # Start with a coarse grid
    p_coarse = np.concatenate([
        np.linspace(1e-6, 0.01, 50),
        np.linspace(0.01, 0.99, 500),
        np.linspace(0.99, 1-1e-6, 50)
    ])
    
    # Calculate integrand on coarse grid
    integrand = calculate_integrand(p_coarse, v, b0, x)
    
    # Compute CDF
    fp = integrate.cumtrapz(integrand, p_coarse, initial=0)
    fp = fp / fp[-1]  # Normalize to ensure it reaches 1
    
    # Find indices where CDF crosses our thresholds
    n_indices = np.where(fp <= 0.975)[0]
    m_indices = np.where(fp < 0.025)[0]
    
    ucl = p_coarse[n_indices[-1]] if len(n_indices) > 0 else 1
    lcl = p_coarse[m_indices[-1]] if len(m_indices) > 0 else 0
    
    return lcl, ucl

def run_em_algorithm(n, K, max_iterations=20000, tol=1e-8, base_prob=0.25, 
                     initial_ve=0.005, batch_conf_lims=True):
    """
    Run the EM algorithm with optimizations
    """
    M = max_iterations
    ve = np.zeros(M)
    
    x_pred = np.zeros(K)
    v_pred = np.zeros(K)
    x_updt = np.zeros(K)
    v_updt = np.zeros(K)
    x_smth = np.zeros(K)
    v_smth = np.zeros(K)
    p_updt = np.zeros(K)
    
    A = np.zeros(K)
    W = np.zeros(K)
    CW = np.zeros(K)
    
    ve[0] = initial_ve
    x_smth[0] = 0
    b0 = np.log(base_prob / (1 - base_prob))
    
    # Pre-allocate memory for the exp terms to avoid recomputation
    exp_b0_x = np.zeros(K)
    
    # Use tqdm for progress tracking
    with tqdm(total=M, desc="EM Algorithm Progress") as pbar:
        for m in range(M):
            # Forward pass - can be vectorized in parts but the update step requires iteration
            for k in range(K):
                if k == 0:  # boundary condition
                    x_pred[k] = x_smth[0]
                    v_pred[k] = 2 * ve[m]  # Simplified from ve[m] + ve[m]
                else:
                    x_pred[k] = x_updt[k - 1]
                    v_pred[k] = v_updt[k - 1] + ve[m]
                
                x_updt[k] = get_state_update(x_pred[k], v_pred[k], b0, n[k])
                
                # Precompute the exponential term
                exp_term = np.exp(b0 + x_updt[k])
                exp_b0_x[k] = exp_term
                
                p_updt[k] = 1 / (1 + exp_term)
                v_updt[k] = 1 / ((1 / v_pred[k]) + p_updt[k] * (1 - p_updt[k]))
            
            # Copy values for the last element
            x_smth[K-1] = x_updt[K-1]
            v_smth[K-1] = v_updt[K-1]
            W[K-1] = v_smth[K-1] + x_smth[K-1]**2
            
            # Compute A efficiently
            A[:(K-1)] = v_updt[:(K-1)] / v_pred[1:]
            x0_prev = x_smth[0]
            
            # Backward smoothing pass
            for k in range(K-2, -1, -1):
                x_smth[k] = x_updt[k] + A[k] * (x_smth[k + 1] - x_pred[k + 1])
                v_smth[k] = v_updt[k] + A[k]**2 * (v_smth[k + 1] - v_pred[k + 1])
                
                CW[k] = A[k] * v_smth[k + 1] + x_smth[k] * x_smth[k + 1]
                W[k] = v_smth[k] + x_smth[k]**2
            
            if m < M - 1:
                # More efficient computation of ve update
                ve[m + 1] = (np.sum(W[1:]) + np.sum(W[:(K-1)]) - 2 * np.sum(CW) + 0.5 * W[0]) / (K + 1)
                x0 = x_smth[0] / 2
                
                if (abs(ve[m + 1] - ve[m]) < tol) and (abs(x0 - x0_prev) < tol):
                    print(f'm = {m}\nx0 = {x_smth[0]:.18f}\nve = {ve[m]:.18f}\n')
                    print(f'Converged at m = {m}\n')
                    break
                else:
                    # Only print status every 100 iterations to reduce overhead
                    if m % 100 == 0:
                        print(f'm = {m}\nx0 = {x_smth[0]:.18f}\nve = {ve[m+1]:.18f}\n')
                    
                    # Reset arrays for next iteration - reuse existing arrays
                    x_pred.fill(0)
                    v_pred.fill(0)
                    x_updt.fill(0)
                    v_updt.fill(0)
                    
                    # Keep x_smth[0] for next iteration
                    x0_temp = x0
                    x_smth.fill(0)
                    x_smth[0] = x0_temp
                    
                    v_smth.fill(0)
                    p_updt.fill(0)
                    A.fill(0)
                    W.fill(0)
                    CW.fill(0)
            
            pbar.update(1)
            if m < M-1 and abs(ve[m + 1] - ve[m]) < tol and abs(x0 - x0_prev) < tol:
                pbar.update(M - m - 1)  # Jump to end if converged
                break
    
    # Calculate smoothed probability
    p_smth = 1 / (1 + np.exp((-1) * (b0 + x_smth)))
    
    # Calculate confidence limits for state
    lcl_x = norm.ppf(0.025, x_smth, np.sqrt(v_smth))
    ucl_x = norm.ppf(0.975, x_smth, np.sqrt(v_smth))
    
    # Calculate certainty metric
    median_x = np.median(x_smth)
    certainty = 1 - norm.cdf(median_x * np.ones(K), x_smth, np.sqrt(v_smth))
    
    # Calculate confidence limits for probability
    lcl_p = np.zeros(K)
    ucl_p = np.zeros(K)
    
    if batch_conf_lims:
        # Calculate confidence limits for p_k in batches for better performance
        batch_size = 10  # Process confidence limits in batches
        print('Calculating the pk confidence limits in batches...')
        
        with tqdm(total=K, desc="Confidence Limits Progress") as pbar:
            for i in range(0, K, batch_size):
                end_idx = min(i + batch_size, K)
                for k in range(i, end_idx):
                    lcl_p[k], ucl_p[k] = get_pk_conf_lims(v_smth[k], b0, x_smth[k])
                pbar.update(end_idx - i)
    else:
        # Original sequential calculation
        print('Calculating the pk confidence limits...')
        with tqdm(total=K, desc="Confidence Limits Progress") as pbar:
            for k in range(K):
                lcl_p[k], ucl_p[k] = get_pk_conf_lims(v_smth[k], b0, x_smth[k])
                pbar.update(1)
    
    print('Finished calculating the pk confidence limits.')
    
    return x_smth, v_smth, p_smth, lcl_x, ucl_x, lcl_p, ucl_p, certainty, ve[:m+1]

def extract_reward_and_sessions(data, rat, condition, traj_type):
    """Get reward vector and session IDs for a given rat and condition."""
    if condition == 'all':
        rewards = data[rat]['reward']
        sessions = data[rat]['session']
    else:
        rewards = data[rat][traj_type]['rewarded']
        sessions = data[rat][traj_type]['session']
    return rewards.to_numpy(), sessions.to_numpy()

def plot_em_output(n, p_smth, lcl_p, ucl_p, certainty, base_prob, sessions):
    K = len(n)
    t = np.arange(K)
    session_changes = np.where(np.diff(sessions) != 0)[0] + 1

    plt.figure(figsize=(12, 10))

    # Plot 1: binary outcomes
    plt.subplot(311)
    plt.scatter(np.where(n == 1)[0], np.ones(np.sum(n == 1)), 5, 'g', alpha=0.5)
    plt.scatter(np.where(n == 0)[0], np.zeros(np.sum(n == 0)), 5, 'r', alpha=0.5)
    for s in session_changes:
        plt.axvline(x=s, linestyle='--', color='gray', linewidth=0.5)
    plt.xlim(0, K)

    # Plot 2: smoothed probability
    plt.subplot(312)
    plt.plot(t, p_smth, 'r', linewidth=1.5)
    plt.fill_between(t, lcl_p, ucl_p, color=(1, 0, 127/255), alpha=0.3)
    plt.axhline(y=base_prob, linestyle='--', color='k')
    for s in session_changes:
        plt.axvline(x=s, linestyle='--', color='gray', linewidth=0.5)
    plt.ylabel('P (Correct)')
    plt.ylim(0.1, 1.0)
    plt.xlim(0, K)
    plt.tick_params(labelbottom=False)

    # Plot 3: certainty
    plt.subplot(313)
    plt.fill_between([0, K], [0.9]*2, [1]*2, color='green', alpha=0.7)
    plt.fill_between([0, K], [0]*2, [0.1]*2, color='red', alpha=0.7)
    plt.plot(t, certainty, color=(138/255, 43/255, 226/255), linewidth=1.5)
    for s in session_changes:
        plt.axvline(x=s, linestyle='--', color='gray', linewidth=0.5)
    plt.ylabel('Certainty')
    plt.xlabel('Trials')
    plt.xlim(0, K)

    plt.tight_layout()
    plt.show()

def plot_smoothed_probability(t, p_smth, lcl_p, ucl_p, base_prob, sessions):
    session_changes = np.where(np.diff(sessions) != 0)[0] + 1
    plt.figure()
    plt.plot(t, p_smth, 'r', linewidth=1.5)
    plt.fill_between(t, lcl_p, ucl_p, color=(1, 0, 127/255), alpha=0.3)
    plt.axhline(y=base_prob, linestyle='--', color='k')
    for s in session_changes:
        plt.axvline(x=s, linestyle='--', color='gray', linewidth=0.5)
    plt.ylabel('Smoothed p(k)')
    plt.ylim(0.1, 1.0)
    plt.xlim(0, len(t))
    plt.tight_layout()
    plt.show()

#%% Main code

if __name__ == "__main__":
    base_prob = None
    condition = 'all'  # or 'by_traj'
    traj_type = 'outbound'

    data = follower_follow_reward_df if condition == 'all' else demonstrator_reward_dfs_by_traj
    
    
    for rat in data:
        if base_prob == None and condition == 'all':
            # Get rewards from the 1st 2 sessions
            rwd = data[rat].loc[data[rat]['session'].isin([0, 1]), ['reward']].to_numpy()
            if np.mean(rwd) > 0.5:
                base_prob = 0.5
            else:
                base_prob = np.mean(rwd)
                
        elif base_prob == None and condition == 'by_traj':
            # Get rewards from the 1st 2 sessions
            rwd = data[rat][traj_type].loc[data[rat][traj_type]['session'].isin([0, 1]), ['rewarded']].to_numpy()
            if np.mean(rwd) > 0.5:
                base_prob = 0.5
            else:
                base_prob = np.mean(rwd)
        
        u, sessions = extract_reward_and_sessions(data, rat, condition, traj_type)
        K = len(u)
        n = (u > 0).astype(int)

        x_smth, v_smth, p_smth, lcl_x, ucl_x, lcl_p, ucl_p, certainty, ve_history = run_em_algorithm(
            n, K, max_iterations=20000, tol=1e-8, base_prob=base_prob, initial_ve=0.005
        )

        plot_em_output(n, p_smth, lcl_p, ucl_p, certainty, base_prob, sessions)
        plot_smoothed_probability(np.arange(K), p_smth, lcl_p, ucl_p, base_prob, sessions)

#%% Anne Smith's code

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumtrapz


def newtonsolve(muone, qold, sigoldsq, mm, ll):
    """
    Solve the posterior mode equation using Newton's method.

    Parameters:
    - muone: scalar
    - qold: scalar
    - sigoldsq: scalar
    - mm: scalar
    - ll: scalar

    Returns:
    - q: solution
    - timefail: 0 if converged, 1 if failed to converge
    """
    it = [qold + sigoldsq * (mm - ll * np.exp(muone) * np.exp(qold) /
                            (1 + np.exp(muone) * np.exp(qold)))]

    for i in range(40):
        exp_term = np.exp(muone) * np.exp(it[i])
        g = qold + sigoldsq * (mm - ll * exp_term / (1 + exp_term)) - it[i]
        gprime = -ll * sigoldsq * exp_term / (1 + exp_term)**2 - 1
        next_it = it[i] - g / gprime
        it.append(next_it)

        if abs(next_it - it[i]) < 1e-14:
            return next_it, 0  # Converged

    # Failed to converge
    return it[-1], 1




def findj(p, n, pcrit):
    """
    Find the minimum number of ones in a row needed to be confident that p < pcrit.

    Parameters:
    - p: float, probability of correct response
    - n: int, length of sequence
    - pcrit: float, critical p-value

    Returns:
    - jfinal: int, minimum run length
    """
    res = []

    for j in range(2, 26):  # Check j values from 2 to 25
        f = []
        if n <= 2 * j:
            f.append(p**j)
            for _ in range(2, n - j + 2):
                f.append(p**j * (1 - p))
            total = sum(f)
        else:
            f.append(p**j)
            for _ in range(2, j + 2):
                f.append(p**j * (1 - p))
            for i in range(j + 2, n - j + 2):
                xx = list(range(i - j - 1))
                s = sum(f[k] for k in xx)
                f.append(p**j * (1 - p) * (1 - s))
            total = sum(f)

        res.append((j, total))

    for jval, pval in res:
        if pval < pcrit:
            return jval + 1  # +1 because original MATLAB loop starts at j=2

    return None  # If no j satisfies the condition


# Assuming the following functions are defined based on your provided MATLAB functions: 
def recfilter(I, sige, qguess, sigsqguess, muone):
    """
    Implements the forward recursive filtering algorithm.
    
    Args:
        I: 2 x T array, where I[0, :] = mm (number correct), I[1, :] = ll (total trials)
        sige: float, sqrt of process noise variance
        qguess: float, initial guess of q
        sigsqguess: float, initial guess of variance
        muone: float, log-odds of background probability

    Returns:
        p: probability estimates
        qhat: posterior mode
        sigsq: posterior variance
        qhatold: one-step prediction
        sigsqold: one-step prediction variance
    """
    mm = I[0, :]
    ll = I[1, :]
    T = I.shape[1]

    qhat = np.zeros(T + 1)
    sigsq = np.zeros(T + 1)
    qhatold = np.zeros(T + 1)
    sigsqold = np.zeros(T + 1)

    qhat[0] = qguess
    sigsq[0] = sigsqguess

    number_fail = []

    for t in range(1, T + 1):
        qhatold[t] = qhat[t - 1]
        sigsqold[t] = sigsq[t - 1] + sige ** 2

        qhat[t], flagfail = newtonsolve(muone, qhatold[t], sigsqold[t], mm[t - 1], ll[t - 1])

        if flagfail > 0:
            number_fail.append(t)

        denom = -1 / sigsqold[t] - (ll[t - 1] * np.exp(muone) * np.exp(qhat[t])) / \
                (1 + np.exp(muone) * np.exp(qhat[t])) ** 2
        sigsq[t] = -1 / denom

    if number_fail:
        print(f"Newton convergence failed at times {number_fail}")

    p = np.exp(muone) * np.exp(qhat) / (1 + np.exp(muone) * np.exp(qhat))

    return p, qhat, sigsq, qhatold, sigsqold


def backest(q, qold, sigsq, sigsqold):
    """
    Backward smoothing step.

    Args:
        q: posterior mode from forward pass
        qold: one-step prediction from forward pass
        sigsq: posterior variance
        sigsqold: one-step prediction variance

    Returns:
        qnew: smoothed posterior
        signewsq: smoothed posterior variance
        a: smoothing gain
    """
    T = q.shape[0]
    qnew = np.zeros(T)
    signewsq = np.zeros(T)
    a = np.zeros(T)

    qnew[-1] = q[-1]
    signewsq[-1] = sigsq[-1]

    for i in range(T - 2, 0, -1):
        a[i] = sigsq[i] / sigsqold[i + 1]
        qnew[i] = q[i] + a[i] * (qnew[i + 1] - qold[i + 1])
        signewsq[i] = sigsq[i] + a[i] ** 2 * (signewsq[i + 1] - sigsqold[i + 1])

    return qnew, signewsq, a


def em_bino(I, qnew, signewsq, a, muold, startflag):
    """
    Expectation-Maximization step to update process noise variance.

    Args:
        I: spike train data (unused in this function)
        qnew: smoothed posterior mode
        signewsq: smoothed posterior variance
        a: smoothing gain
        muold: not used here (kept for consistency)
        startflag: initialization strategy (0, 1, or 2)

    Returns:
        newsigsq: updated sigma_e squared (process noise variance)
    """
    N = qnew.shape[0]

    qnewt = qnew[2:N]
    qnewtm1 = qnew[1:N - 1]
    signewsqt = signewsq[2:N]
    a = a[1:]  # a(2:end) in MATLAB

    covcalc = signewsqt * a

    term1 = np.sum(qnewt ** 2) + np.sum(signewsqt)
    term2 = np.sum(qnewt ** 2) + np.sum(signewsqt)
    term3 = -2 * (np.sum(covcalc) + np.sum(qnewt * qnewtm1))

    if startflag == 1:
        term6 = 1.5 * qnew[1] ** 2 + 2.0 * signewsq[1] - qnew[-1] ** 2 - signewsq[-1]
    elif startflag == 0:
        term6 = 2.0 * qnew[1] ** 2 + 2.0 * signewsq[1] - qnew[-1] ** 2 - signewsq[-1]
    elif startflag == 2:
        term6 = 1.0 * qnew[1] ** 2 + 2.0 * signewsq[1] - qnew[-1] ** 2 - signewsq[-1]
        N = N - 1

    newsigsq = (term1 + term2 + term3 + term6) / N
    return newsigsq



def pdistn(q, s, muone, background_prob):
    """
    Computes the posterior distribution over probabilities.

    Args:
        q: posterior modes (array)
        s: posterior variances (array)
        muone: log background odds
        background_prob: prior background probability

    Returns:
        p05: 5th percentile
        p95: 95th percentile
        pmid: median
        pmode: mode
        pmatrix: cumulative posterior distribution values at background_prob
    """
    p05, p95, pmid, pmode = [], [], [], []
    pmatrix = []

    dels = 1e-4
    pr = np.arange(dels, 1 - dels, dels)

    for i in range(len(q)):
        qq = q[i]
        ss = s[i]

        fac = np.log(pr / (1 - pr) / np.exp(muone)) - qq
        fac = np.exp(-fac ** 2 / (2 * ss))
        pd = dels * (np.sqrt(1 / (2 * np.pi * ss)) * 1 / (pr * (1 - pr)) * fac)

        sumpd = cumtrapz(pd, pr, initial=0)

        lowlimit = np.searchsorted(sumpd, 0.05)
        highlimit = np.searchsorted(sumpd, 0.95)
        midlimit = np.searchsorted(sumpd, 0.5)

        p05.append(pr[lowlimit] if lowlimit < len(pr) else pr[0])
        p95.append(pr[highlimit - 1] if highlimit < len(pr) and highlimit > 0 else pr[-1])
        pmid.append(pr[midlimit] if midlimit < len(pr) else pr[-1])
        pmode.append(pr[np.argmax(pd)])
        
        inte = int(background_prob / dels)
        inte = min(max(inte, 0), len(sumpd) - 1)
        pmatrix.append(sumpd[inte])

    return np.array(p05), np.array(p95), np.array(pmid), np.array(pmode), np.array(pmatrix)


# Example data flag
exampledata = 0
startflag = 2  # 0: fix initial condition, 1: estimate initial condition, 2: remove x0 from likelihood

if exampledata == 0:
    mm = np.array([0, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1] + [1]*20)
    background_prob = 0.5
    ll = np.ones_like(mm)
else:
    mm = np.array([4, 3, 3, 3, 2, 3, 10, 11, 12, 3, 5, 6, 7, 8, 9, 12, 14, 15, 16, 18, 22, 25, 26])
    ll = 30 * np.ones_like(mm)
    background_prob = 0.1

I = np.vstack((mm, ll))

# Initial guess for sigma_epsilon squared
sige = np.sqrt(0.005)
sigsqguess = sige**2

# Set mu from the chance of correct
muone = np.log(background_prob / (1 - background_prob))

# Convergence criterion
cvgce_crit = 1e-8

# EM algorithm
qguess = 0
number_steps = 20000
newsigsq = []
qnew1save = []

for jk in range(number_steps):
    p, q, s, qold, sold = recfilter(I, sige, qguess, sigsqguess, muone)
    qnew, signewsq, a = backest(q, qold, s, sold)

    if startflag == 1:
        qnew[0] = 0.5 * qnew[1]
        signewsq[0] = sige**2
    elif startflag == 0:
        qnew[0] = 0
        signewsq[0] = sige**2
    elif startflag == 2:
        qnew[0] = qnew[1]
        signewsq[0] = signewsq[1]

    new_sigsq = em_bino(I, qnew, signewsq, a, muone, startflag)
    newsigsq.append(new_sigsq)
    qnew1save.append(qnew[0])

    if jk > 0:
        a1 = abs(newsigsq[jk] - newsigsq[jk - 1])
        a2 = abs(qnew1save[jk] - qnew1save[jk - 1])
        if a1 < cvgce_crit and a2 < cvgce_crit and startflag >= 1:
            print(f'EM estimates converged after {jk + 1} steps')
            break
        elif a1 < cvgce_crit and startflag == 0:
            print(f'EM estimate converged after {jk + 1} steps')
            break

    sige = np.sqrt(newsigsq[jk])
    qguess = qnew[0]
    sigsqguess = signewsq[0]
else:
    print(f'Failed to converge after {number_steps} steps; convergence criterion was {cvgce_crit}')

# Compute confidence limits
b05, b95, bmid, bmode, pmatrix = pdistn(qnew, signewsq, muone, background_prob)

# Find the last point where the 90% interval crosses chance
cback_indices = np.where(b05 < background_prob)[0]
if cback_indices.size > 0:
    if cback_indices[-1] < I.shape[1]:
        cback = cback_indices[-1]
    else:
        cback = np.nan
else:
    cback = np.nan

# Plotting
t = np.arange(1, len(p))

plt.figure(figsize=(10, 8))

# Subplot 1
plt.subplot(2, 1, 1)
plt.plot(t, bmode[1:], 'r-', label='Mode')
plt.plot(t, b05[1:], 'k--', label='5% CI')
plt.plot(t, b95[1:], 'k--', label='95% CI')
if exampledata == 0:
    y, x = np.where(mm > 0)
    plt.plot(x + 1, y + 0.05, 'ks', markerfacecolor='k')
    y, x = np.where(mm == 0)
    plt.plot(x + 1, y + 0.05, 'ks', markerfacecolor=[0.75, 0.75, 0.75])
    plt.axis([1, t[-1], 0, 1.05])
else:
    plt.plot(t, mm / ll, 'ko')
    plt.axis([1, t[-1], 0, 1])
plt.axhline(y=background_prob, color='gray', linestyle='--')
plt.title(f'IO(0.95) Learning Trial = {cback} RW variance = {sige**2:.4f}')
plt.xlabel('Trial Number')
plt.ylabel('Probability of a Correct Response')
plt.legend()

# Subplot 2
plt.subplot(2, 1, 2)
plt.plot(t, 1 - np.array(pmatrix[1:]), 'k')
plt.axhline(y=0.90, color='gray', linestyle='--')
plt.axhline(y=0.95, color='gray', linestyle='--')
plt.axhline(y=0.99, color='gray', linestyle='--')
plt.axis([1, t[-1], 0, 1])
plt.grid(True)
plt.xlabel('Trial Number')
plt.ylabel('Certainty')

plt.tight_layout()
plt.show()
