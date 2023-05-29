import pandas as pd
import collections
import matplotlib.pyplot as plt


class ProcessGameState:

    def __init__(self) -> None:
        pass


    # Handle file ingestion and ETL
    # param: filename - path to file
    # return: pandas dataframe
    def read_input(self, filename):
        return pd.read_parquet(filename)


    # Check whether or not each row falls within a provided boundary
    # param: row - one row in dataframe
    #        boundary - a list of vertices of the boundary
    # return: boolean
    def check_boundary(self, row, boundary):

        point = (int(row['x']), int(row['y']), int(row['z']))

        # Check z bound
        if point[2] < 285 or point[2] > 421:
            return False

        # Find all edges of the boundary
        edges = []
        for i in range(len(boundary) - 1):
            edge = (boundary[i], boundary[i + 1])
            edges.append(edge)
        edges.append((boundary[-1], boundary[0]))
        
        # Ray casting towards right infinity
        count = 0
        for edge in edges:
            v1_x, v1_y, v2_x, v2_y= edge[0][0], edge[0][1], edge[1][0], edge[1][1]
            if point[1] > min(v1_y, v2_y) and point[1] < max(v1_y, v2_y):
                if point[0] < max(v1_x, v2_x):
                    intersection = (point[1] - v1_y) * (v2_x - v1_x) / (v2_y - v1_y) + v1_x
                    if point[0] < intersection:
                        count += 1

        # If the count of hits on boundary edges is an odd number, then the point is in the boundary
        if count % 2 != 0:
            return True
        else:
            return False


    # Extract the weapon classes from the inventory json column
    # param: row - one row in dataframe
    # return: dictionary of {weapon class: count}
    def extract_weapon(self, row):
        if row['inventory'] is None:
            return None
        
        weapon_info = collections.defaultdict(int)
        for dict in row['inventory']:
            weapon_info[dict['weapon_class']] += 1
        
        return weapon_info
    

    # Extract totel number of certain weapon classes (e.g. ['Rifle', 'SMG'] asked in Question 2b)
    # param: row - one row in dataframe
    #        weapon_classes - weapon classes of interest
    # return: count of weapons
    def weapon_cnt(self, row, weapon_classes):
        count = 0
        if row['weapon_info'] is None:
            return count
        
        for weapon in weapon_classes:
            if weapon in row['weapon_info']:
                count += row['weapon_info'][weapon]
        return count


# ============================================================================================================
# Load the data
pgs = ProcessGameState()
data_df = pgs.read_input('data/game_state_frame_data.parquet')


# ==================================================== 2a ====================================================
# Is entering via the light blue boundary a common strategy used by Team2 on T (terrorist) side?

print('======================================================================================================')

# Extract rows for Team2 on T (terrorist) side
Team2_T = data_df[(data_df['team'] == 'Team2') & (data_df['side'] == 'T')].copy()

# Boundary vertices
boundary = [[-1735, 250], [-2024, 398], [-2806, 742], [-2472, 1233], [-1565, 580]]

# Apply check_boundary to each row and record result in a new column 'in_light_blue_zone'
Team2_T['in_light_blue'] = Team2_T.apply(lambda row: pgs.check_boundary(row, boundary), axis = 1)

# Group by player, round_num, and in_light_blue column
player_position = Team2_T.groupby(['player', 'round_num', 'in_light_blue']).size().unstack(fill_value = 0)
print(player_position)

# Summarize strategy for each player
false = true = 0
for i in range(len(player_position[0])):
    if player_position[0][i] > player_position[1][i]:
        false += 1
    else:
        true += 1

# Answer the question
if false > true:
    print('Entering via the light blue boundary is not a common strategy used by Team2 on T (terrorist) side.')
else:
    print('Entering via the light blue boundary is a common strategy used by Team2 on T (terrorist) side.')


# ==================================================== 2b ====================================================
# What is the average timer that Team2 on T (terrorist) side enters "BombsiteB" with least 2 rifles or SMGs?

print('======================================================================================================')

# Apply extract_weapon to each row and record result in a new column 'weapon_info'
Team2_T['weapon_info'] = Team2_T.apply(lambda row: pgs.extract_weapon(row), axis = 1)

# Apply weapon_cnt to each row and record result in a new column 'Rifle_SMG_count'
Team2_T['Rifle_SMG_count'] = Team2_T.apply(lambda row: pgs.weapon_cnt(row, ['Rifle', 'SMG']), axis = 1)

# Extract rows for area_name = 'BombsiteB' and Rifle_SMG_count >= 2
Team2_T_filtered = Team2_T[(Team2_T['area_name'] == 'BombsiteB') & (Team2_T['Rifle_SMG_count'] >= 2)].copy()

# Answer the question
if len(Team2_T_filtered) == 0:
    print(Team2_T_filtered)
    print('Team2 on T (terrorist) side has never entered "BombsiteB" with least 2 rifles or SMGs.')
else:
    Team2_T_filtered = Team2_T_filtered.groupby(['player', 'round_num']).agg({'clock_time': 'max'})
    print(Team2_T_filtered)

    avg_time = pd.to_datetime(Team2_T_filtered['clock_time'], infer_datetime_format = True).mean()
    hour = '0' + str(avg_time.hour)
    if len(str(avg_time.minute)) != 2:
        minute = '0' + str(avg_time.minute)
    else:
        minute = str(avg_time.minute)

    clock_time = hour + ':' + minute
    print('The average timer that Team2 on T (terrorist) side enters "BombsiteB" with least 2 rifles or SMGs is at clock time ' + clock_time + '.')


# ==================================================== 2c ====================================================
# Now that we’ve gathered data on Team2 T side, let's examine their CT (counter-terrorist) Side.
# Using the same data set, tell our coaching staff where you suspect them to be waiting inside "BombsiteB."

print('======================================================================================================')

# Extract rows for Team2 on CT (counter-terrorist) side with area_name = 'BombsiteB'
Team2_CT_BombsiteB = data_df[(data_df['team'] == 'Team2') & (data_df['side'] == 'CT') & (data_df['area_name'] == 'BombsiteB')].copy()

# Plot a heatmap
plt.hist2d(Team2_CT_BombsiteB['x'], Team2_CT_BombsiteB['y'])

# Answer the question
print('In the heatmap, the upper right region, especially the yellow grid, shows a relatively higher frequency of seeing CT players, so I suspect them to be waiting in that area inside "BombsiteB." The approximate coordinates for the yellow grid are [(-885, 410), (-815, 410), (-885, 495), (-815, 495)].')
plt.show()
