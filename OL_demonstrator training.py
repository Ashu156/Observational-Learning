# -*- coding: utf-8 -*-
"""
Created on Mon May 12 17:52:30 2025

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

def parse_individual_events(file_path):
    """
    Parses a logfile of behavioral poke/unpoke events,
    annotates with trajectory, goal well, and reward status.
    """
    # Read logfile
    with open(file_path, 'r') as f:
        log_lines = f.readlines()

    # Regex patterns
    poke_pattern = re.compile(r'(\d+)\s+Poke in well (\d) - (LEFT|CENTER|RIGHT)')
    unpoke_pattern = re.compile(r'(\d+)\s+Unpoke in well (\d) - (LEFT|CENTER|RIGHT)')
    reward_pattern = re.compile(r'(\d+)\s+count = (\d+)')

    entries = []
    reward_timestamps = []

    for line in log_lines:
        line = line.strip()

        # Match poke/unpoke
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

        # Match reward
        reward_match = reward_pattern.match(line)
        if reward_match:
            reward_ts = int(reward_match.group(1))  # keep as int to match poke timestamps
            reward_timestamps.append(reward_ts)

    # Convert to DataFrame and sort
    df = pd.DataFrame(entries).sort_values("timestamp").reset_index(drop=True)

    # Step 1: Remove redundant poke events (poke followed by poke)
    redundant_poke_indices = []
    i = 1
    while i < len(df):
        if df.loc[i - 1, "action"] == "poke" and df.loc[i, "action"] == "poke":
            redundant_poke_indices.append(i)
            # Find the first matching unpoke for this redundant poke
            for j in range(i + 1, len(df)):
                if df.loc[j, "action"] == "unpoke" and df.loc[j, "well_num"] == df.loc[i, "well_num"]:
                    redundant_poke_indices.append(j)
                    break
        i += 1
    df_cleaned = df.drop(index=redundant_poke_indices).reset_index(drop=True)

    # Step 2: Compute goal wells and trajectories
    goal_wells = []
    trajectories = []
    goal_trajectories = []
    visit_sequence = []
    last_well = None

    for idx, row in df_cleaned.iterrows():
        current = row["well_label"]
        traj = None
        goal_traj = None
        goal = None

        if row["action"] == "poke":
            # Goal trajectory logic
            if current == "CENTER" and last_well in ("LEFT", "RIGHT"):
                goal_traj = "inbound"
                traj = "inbound"
            elif (current == "LEFT" and last_well == "RIGHT") or (current == "RIGHT" and last_well == "LEFT"):
                goal_traj = "inbound"
                traj = "outbound"
            elif current in ("LEFT", "RIGHT") and last_well == "CENTER":
                goal_traj = "outbound"
                traj = "outbound"
            
            last_well = current  # Update only for pokes

            # Goal well logic
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

    df_cleaned["goal_well"] = goal_wells
    df_cleaned["trajectory"] = trajectories
    df_cleaned["goal_trajectory"] = goal_trajectories

    # Step 3: Forward-fill trajectory and goal well
    df_cleaned[["goal_well", "trajectory", "goal_trajectory"]] = df_cleaned[["goal_well", "trajectory", "goal_trajectory"]].ffill()

    # Step 4: Annotate reward status
    df_cleaned["rewarded"] = 0
    poke_indices = df_cleaned[df_cleaned["action"] == "poke"].index

    for ts in reward_timestamps:
        # Find the nearest poke event (could use a better method like bisection for efficiency)
        nearest_idx = (df_cleaned.loc[poke_indices, "timestamp"] - ts).abs().idxmin()
        df_cleaned.at[nearest_idx, "rewarded"] = 1

    return df_cleaned

#%% Function for extracting transition sequences 

def combine_consecutive_wells(df):
    """
    Function to extract the transition sequence of well
    visits from rat poking data. It also caluclates duration
    a rat remains / dwells at a well.
    
    """
    
    
    # Identify groups of consecutive rows with the same `thiswell`
    df['group'] = (df['well_num'] != df['well_num'].shift()).cumsum()
    
    # Group by `group` and aggregate
    result = (
        df.groupby('group')
        .agg(
            timestamp = ('timestamp', 'first'),                # First start time
            # end=('end', 'last'),                    # Last end time
            well_num = ('well_num', 'first'),         # First thiswell (same for the group)
            well_label = ('well_label', 'first'),
            goal_well = ('goal_well', 'first'),
            trajectory = ('trajectory', 'first'),
            goal_trajectory = ('goal_trajectory', 'first'),
            rewarded = ('rewarded', 'first'),       # Sum of dwell times in the group
        
        )
        .reset_index(drop=True)
    )
    
    return result

#%% Batch process individual rat experiments

# Define the dtype for the structured array (single rat)
dtype = np.dtype([
    ('name', 'U100'),        # File name
    ('folder', 'U255'),      # Folder path
    ('date', 'U50'),         # Extracted date
    ('cohortnum', 'U50'),    # Cohort number
    ('runum', 'U50'),        # Run number
    ('ratnum', 'U50'),       # Single rat number as string
    ('ratnames', 'U100'),     # Single rat name as string
    ('ratsamples', 'O'),     # Single sample (float)
    ('pokeData',  'O'),
    ('reward', 'f8'),        # Single reward value (float)
    ('nTransitions', 'f8'),  # Single transition count (float)
    ('perf', 'f8'),          # Performance metric (float)
    ('duration', 'f8')      # Session duration (float)

])



# Function to extract data from filename
def parse_individual_filename(filename):
    """
    Function to extract data from filename
    """
    match = re.search(r"log(\d{2}-\d{2}-\d{4})\((\d+)-([A-Z]+\d+)\)\.stateScriptLog", filename)
    
    if match:
        date, runum, rat1 = match.groups()
        return date, runum, f"{rat1}", [rat1]
    return None, None, None, None

# Function to load .stateScriptLog files
def process_individual_stateScriptLogs(base_dir):
    
    """
    Function to extract data from filename
    """
    
    struct_data = []
    
    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.endswith(".stateScriptLog"):
                full_path = os.path.join(root, file)
                date, runum, ratnums, ratnames = parse_individual_filename(file)

                # Dummy DataFrames for example
                df_rat = parse_individual_events(full_path)
                
                # match = df_rat['match'].sum()
                reward = df_rat['rewarded'].sum()
                
                
                session_length = (df_rat['timestamp'].max() - df_rat['timestamp'].min()) / (60*1000)
                
                rat_df = combine_consecutive_wells(df_rat)
                
                # count = count_valid_triplets(rat_df, column_name='well')
                
                # seq = rat_df['well'].dropna().astype(int).astype(str).str.cat()
                
                
                nTransitions = float(len(rat_df))
                
                
                # if nTransitions > 150:
                #     nTransitions = np.nan
                #     perf = np.nan
                # else:
                perf = 100*(reward / nTransitions)


                struct_data.append((
                    file,  # name
                    root,  # folder
                    date,  # date
                    root.split("/")[-1],  # cohortnum (assuming one level up)
                    runum,  # runum
                    str(ratnums),  # ratnums
                    str(ratnames),  # ratnames
                    df_rat,  # ratsamples (tuple of DataFrames)
                    rat_df,  # consolidated ratsamples
                    reward,  # rewards for rats
                    nTransitions,        # transitions for rats
                    perf,                # pair performance
                    session_length      # session duration
                    # seq,                 # transition sequences
                    # count                # triplet counts
                ))
                
                # Initialize an empty list to store the data
                structured_data = []

                for entry in struct_data:
                    data_entry = {
                        'name': entry[0],
                        'folder': entry[1],
                        'date': entry[2],
                        'cohortnum': entry[3],
                        'runum': entry[4],
                        'ratnum': entry[5],
                        'ratnames': entry[6],
                        'ratsamples': entry[7],  # DataFrame
                        'pokeData': entry[8],
                        'reward': entry[9],
                        'nTransitions': entry[10],
                        'perf': entry[11],
                        'duration': entry[12]
                        # 'transition sequence': entry[14],  # List
                        # 'triplet counts': entry[15]  # Dict
                    }
                    structured_data.append(data_entry)


    
    return structured_data

# Define base directory
base_dir = "E:/Jadhav lab data/Behavior/Observational learning/OL pilot/Data/Demonstrator training"

# Load the structured array
data = process_individual_stateScriptLogs(base_dir)

# Sort by date and run numbers
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

    df = entry['pokeData']
    df = assign_w_rule_logic(df)

#%% Dictionary to store performance data for each individual rat

Wrule_perf = {}
Wrule_reward_rate = {}

for entry in sorted_data:
    rat = entry['ratnames']  # entry['ratnames'] contains ['Rat1', 'Rat2']
    df = entry['pokeData']
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

sns.set(style='ticks')
sns.set_context('poster')

# Use the default matplotlib color cycle
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

# Plot performance
plt.figure()
handles = []  # Store legend handles
labels = []   # Store corresponding labels
lmbda = 5.0e-1

for i, rat in enumerate(Wrule_perf):  # ✅ Fixed syntax
    color = colors[i % len(colors)]  
    y0 = Wrule_perf[rat]
    x0 = np.linspace(0.0, len(y0), len(y0))
    
    # Raw data (scatter)
    scatter_handle = plt.plot(x0, y0, "x", color=color)[0]
    
    # Smooth fit
    basis, coeffs = smoothfit.fit1d(x0, y0, np.min(x0), np.max(x0), 1000, degree=1, lmbda=lmbda)
    line_handle = plt.plot(basis.mesh.p[0], coeffs[basis.nodal_dofs[0]], "-", color=color)[0]

    handles.append(scatter_handle)
    labels.append(f'{rat}')

plt.xlabel('Session')
plt.ylabel('% Correct choices')
plt.ylim((20, 100))
plt.legend(handles=handles, labels=labels, title='Rat')
plt.tight_layout()
plt.show()


  
#%% Get binary vectors for rewarded / unrewarded trials

demonstrator_reward_df = {}

for entry in sorted_data:
    rat = entry['ratnum']
    vec = entry['pokeData']['rewarded'].reset_index(drop=True)  # ensure clean indexing

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

#%% Reward vector by trajectory type

demonstrator_reward_by_trajectory = {}
demonstrator_accuracy_by_trajectory = {}

for entry in sorted_data:
    rat = entry['ratnum']
    poke_df = entry['pokeData'].reset_index(drop=True)

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




#%% Plot accuracy for indivdual rats (trajectory wise)

trajectory_types = ['inbound', 'outbound']

# Use the default matplotlib color cycle
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

lmbda = 5.0e-1
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

#%% Plot average across rats

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import sem

# Collect accuracy data across rats by trajectory and session
combined = []

for rat, traj_data in demonstrator_accuracy_by_trajectory.items():
    for traj_type, df in traj_data.items():
        df_copy = df.copy()
        df_copy['rat'] = rat
        df_copy['goal_trajectory'] = traj_type
        combined.append(df_copy)

# Concatenate all into one DataFrame
combined_df = pd.concat(combined, ignore_index=True)

# Group by session and trajectory, compute mean and SEM
agg_df = combined_df.groupby(['session', 'goal_trajectory'])['proportion_correct'].agg(
    mean='mean',
    sem=sem
).reset_index()

# Plot
plt.figure()

trajectory_types = agg_df['goal_trajectory'].unique()
colors = {'inbound': 'tab:orange', 'outbound': 'tab:purple'}

for traj in trajectory_types:
    df_traj = agg_df[agg_df['goal_trajectory'] == traj]
    plt.errorbar(
        df_traj['session'],
        df_traj['mean'],
        yerr=df_traj['sem'],
        label=f"{traj.capitalize()}",
        marker='o',
        capsize=5,
        color=colors.get(traj, None)
    )

plt.xlabel('Session')
plt.ylabel('Mean Proportion Correct')
plt.title('Average Accuracy Across Rats by Trajectory Type')
plt.ylim(0, 1.05)
plt.legend(title='Trajectory')
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
    plt.title('Binary Outcomes')
    plt.xlim(0, K)

    # Plot 2: smoothed probability
    plt.subplot(312)
    plt.plot(t, p_smth, 'r', linewidth=1.5)
    plt.fill_between(t, lcl_p, ucl_p, color=(1, 0, 127/255), alpha=0.3)
    plt.axhline(y=base_prob, linestyle='--', color='k')
    for s in session_changes:
        plt.axvline(x=s, linestyle='--', color='gray', linewidth=0.5)
    plt.ylabel('Smoothed p(k)')
    plt.ylim(0.1, 1.0)
    plt.xlim(0, K)
    plt.tick_params(labelbottom=False)

    # Plot 3: certainty
    plt.subplot(313)
    plt.fill_between([0, K], [0.9]*2, [1]*2, color='red', alpha=0.7)
    plt.fill_between([0, K], [0]*2, [0.1]*2, color='green', alpha=0.7)
    plt.plot(t, certainty, color=(138/255, 43/255, 226/255), linewidth=1.5)
    for s in session_changes:
        plt.axvline(x=s, linestyle='--', color='gray', linewidth=0.5)
    plt.ylabel('Certainty (HAI)')
    plt.xlabel('Trials')
    plt.xlim(0, K)
    plt.grid(True)

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

# Load only if variable doesn't already exist
if 'demonstrator_reward_df' not in globals():
    demonstrator_reward_df = {}
    demonstrator_reward_df['OL1'] = pd.read_excel("C:/Users/shukl/OneDrive/Documents/Demo_reward_vector_all_traj.xlsx", "OL1_all")
    demonstrator_reward_df['OL3'] = pd.read_excel("C:/Users/shukl/OneDrive/Documents/Demo_reward_vector_all_traj.xlsx", "OL3_all")

if 'demonstrator_reward_dfs_by_traj' not in globals():
    demonstrator_reward_dfs_by_traj = {
        'OL1': {
            'inbound': pd.read_excel("C:/Users/shukl/OneDrive/Documents/Demo_reward_vector_all_traj.xlsx", "OL1_inbound"),
            'outbound': pd.read_excel("C:/Users/shukl/OneDrive/Documents/Demo_reward_vector_all_traj.xlsx", "OL1_outbound")
        },
        'OL3': {
            'inbound': pd.read_excel("C:/Users/shukl/OneDrive/Documents/Demo_reward_vector_all_traj.xlsx", "OL3_inbound"),
            'outbound': pd.read_excel("C:/Users/shukl/OneDrive/Documents/Demo_reward_vector_all_traj.xlsx", "OL3_outbound")
        }
    }

if __name__ == "__main__":
    base_prob = 0.5
    condition = 'by_traj'  # or 'by_traj'
    traj_type = 'outbound'

    data = demonstrator_reward_df if condition == 'all' else demonstrator_reward_dfs_by_traj

    for rat in data:
        u, sessions = extract_reward_and_sessions(data, rat, condition, traj_type)
        K = len(u)
        n = (u > 0).astype(int)

        x_smth, v_smth, p_smth, lcl_x, ucl_x, lcl_p, ucl_p, certainty, ve_history = run_em_algorithm(
            n, K, max_iterations=20000, tol=1e-8, base_prob=base_prob, initial_ve=0.005
        )

        plot_em_output(n, p_smth, lcl_p, ucl_p, certainty, base_prob, sessions)
        plot_smoothed_probability(np.arange(K), p_smth, lcl_p, ucl_p, base_prob, sessions)
    