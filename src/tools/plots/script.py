import pandas as pd
import plotly.express as px
import colorsys # Added for color manipulation
import plotly.colors # Added for color conversion
from scipy.stats import ttest_1samp # Added for statistical test
import numpy as np # Added for mean/std calculation
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import wilcoxon  # Import Wilcoxon signed-rank test
from scipy.stats import mannwhitneyu  # Import Mann-Whitney U test


df = pd.read_csv("all_data.csv")

df['average_prediction'] = df[['prediction1', 'prediction2']].mean(axis=1)

# --- Add group_key based on name ---
df['group_key'] = df['name'].apply(lambda x: x.split('_')[0]) # e.g., 'ablation' or 'steering'


# Extract the reference value
reference_name_it = "steering_vector_0"
base_values = df[df["name"] == reference_name_it]["average_prediction"].values
top_10_base_values = sorted(base_values, reverse=True)[:10]
steering_vector_0_pos_mean = np.mean(top_10_base_values)



# Group by the combined column to prepare for plotting
# Keep 'group_key' and calculate list and mean of averages
grouped_df = df.groupby(['name', 'group_key'])['average_prediction'].agg(
    average_list=list,
    overall_average='mean' # Also calculate mean here for hover data
).reset_index()


# --- Perform Statistical Test and Determine Status ---
alpha = 0.05 # Significance level

def get_comparison_status(data_list, reference_val):
    n_points = len(data_list)
    top_10_data_list = sorted(data_list, reverse=True)[:10]
    mean_val = np.mean(top_10_data_list) if n_points > 0 else np.nan

    if n_points < 2:
        status = 'Insufficient Data'
        p_value = np.nan
        if n_points == 1:
             if mean_val > reference_val: status = 'Above (single point)'
             elif mean_val < reference_val: status = 'Below (single point)'
             else: status = 'Equal (single point)'
        return status, p_value, mean_val, n_points

    if np.std(data_list, ddof=1) == 0:
        p_value = 1.0 # Cannot be significant if no variance
        if mean_val > reference_val: status = 'Above (no variance)'
        elif mean_val < reference_val: status = 'Below (no variance)'
        else: status = 'Equal (no variance)'
        return status, p_value, mean_val, n_points

    # Create reference data for comparison (same value repeated)
    reference_data = [reference_val] * len(data_list)
    
    try:
        # Perform Mann-Whitney U test
        u_stat, two_sided_p = mannwhitneyu(data_list, reference_data, alternative='two-sided')
        
        # Determine which direction to test based on mean comparison
        if mean_val > reference_val:
            # One-sided p-value for testing if distribution is significantly greater
            u_stat, p_value = mannwhitneyu(data_list, reference_data, alternative='greater')
            if p_value < alpha:
                status = 'Significantly Above'
            else:
                status = 'Not Significantly Different'
        else:
            # One-sided p-value for testing if distribution is significantly less
            u_stat, p_value = mannwhitneyu(data_list, reference_data, alternative='less')
            if p_value < alpha:
                status = 'Significantly Below'
            else:
                status = 'Not Significantly Different'
                
    except Exception as e:
        # Handle any exceptions from the test
        print(f"Mann-Whitney U test error: {e}. Falling back to simple comparison.")
        p_value = np.nan
        if mean_val > reference_val:
            status = 'Above (test failed)'
        elif mean_val < reference_val:
            status = 'Below (test failed)'
        else:
            status = 'Equal (test failed)'
            
    return status, p_value, mean_val, n_points

# Apply the function to each group
results = grouped_df['average_list'].apply(lambda x: get_comparison_status(x, steering_vector_0_pos_mean))
grouped_df[['comparison_status', 'p_value', 'mean_value', 'n_points']] = pd.DataFrame(results.tolist(), index=grouped_df.index)

# Create the final color category (abbreviated for legend)
grouped_df['color_category_abbr'] = grouped_df['group_key'] + '_' + grouped_df['comparison_status'].replace({
    'Significantly Above': 'SigAbove',
    'Significantly Below': 'SigBelow',
    'Not Significantly Different': 'NotSig',
    'Above (single point)': 'NotSig',
    'Below (single point)': 'NotSig',
    'Equal (single point)': 'NotSig',
    'Above (no variance)': 'NotSig',
    'Below (no variance)': 'NotSig',
    'Equal (no variance)': 'NotSig',
    'Insufficient Data': 'NotSig'
})



# --- Define the color map dynamically using Pastel ---
unique_group_keys = sorted(grouped_df['group_key'].unique())
base_colors_palette = px.colors.qualitative.Pastel # Use Pastel palette
neutral_color = '#D3D3D3' # Light grey for non-significant

# Ensure we have enough base colors
num_keys = len(unique_group_keys)
extended_base_colors = [base_colors_palette[i % len(base_colors_palette)] for i in range(num_keys)]

color_discrete_map = {}

def rgb_string_to_hex(rgb_str):
    """Convert an RGB string like 'rgb(102, 197, 204)' to hex format '#66C5CC'"""
    try:
        # Extract the numbers from the rgb string
        rgb_values = rgb_str.strip('rgb()').split(',')
        r, g, b = [int(x.strip()) for x in rgb_values]
        return f'#{r:02x}{g:02x}{b:02x}'
    except Exception as e:
        print(f"Could not convert RGB string {rgb_str}: {e}")
        return "#CCCCCC"  # Fallback to grey

# Function to lighten a hex color (robust version)
def lighten_color(color, amount=0.3):
    try:
        # Convert RGB string to hex if needed
        if color.startswith('rgb'):
            color = rgb_string_to_hex(color)
        
        # Now process the hex color
        color = color.lstrip('#')
        # Convert hex to RGB (0-255)
        rgb_in = tuple(int(color[i:i+2], 16) for i in (0, 2, 4))
        # Normalize RGB to 0-1
        rgb_norm = [x / 255.0 for x in rgb_in]
        # Convert to HLS
        h, l, s = colorsys.rgb_to_hls(*rgb_norm)
        # Increase lightness
        new_l = min(1, l + amount)
        # Convert back to RGB (0-1)
        new_rgb_norm = colorsys.hls_to_rgb(h, new_l, s)
        # Denormalize RGB to 0-255
        new_rgb = tuple(int(x * 255) for x in new_rgb_norm)
        # Format back to hex
        return f'#{new_rgb[0]:02x}{new_rgb[1]:02x}{new_rgb[2]:02x}'
    except Exception as e:
        print(f"Could not lighten color {color}: {e}")
        return "#CCCCCC"  # Fallback to grey

# Assign colors: base for 'SigAbove', lighter for 'SigBelow', neutral for 'NotSig'
for i, key in enumerate(unique_group_keys):
    base_color = extended_base_colors[i]
    # Convert base color to hex if it's an RGB string
    if isinstance(base_color, str) and base_color.startswith('rgb'):
        base_color = rgb_string_to_hex(base_color)
    lighter_color = lighten_color(base_color)

    # Use the abbreviated status for map keys
    color_discrete_map[f"{key}_SigAbove"] = base_color
    color_discrete_map[f"{key}_SigBelow"] = lighter_color
    color_discrete_map[f"{key}_NotSig"] = neutral_color


# Explode the dataframe for plotting
# Merge necessary columns from grouped_df for hover info
plot_df = grouped_df.explode('average_list').rename(columns={'average_list': 'average_prediction'})
plot_df['average_prediction'] = pd.to_numeric(plot_df['average_prediction'])

# Add columns needed for hover data (ensure correct merge/transfer)
plot_df = pd.merge(plot_df[['name', 'average_prediction']],
                   grouped_df[['name', 'comparison_status', 'p_value', 'mean_value', 'n_points', 'color_category_abbr']],
                   on='name',
                   how='left')
print(plot_df.head())
print(grouped_df.head())
df['row_id'] = range(len(df))

# When we explode the grouped_df, we lose the connection to the original dataframe rows
# We need to reconstruct this connection to correctly map active_count values

# First, create an exploded dataframe with the original indices
exploded_indices = []
exploded_values = []

for idx, row in grouped_df.iterrows():
    for val in row['average_list']:
        exploded_indices.append(idx)
        exploded_values.append(val)

exploded_df = pd.DataFrame({
    'grouped_idx': exploded_indices,
    'average_prediction': exploded_values
})

# Now merge with the grouped_df to get name_it
exploded_df = pd.merge(
    exploded_df,
    grouped_df[['name','group_key','comparison_status','p_value','mean_value','n_points','color_category_abbr']].reset_index(),
    left_on='grouped_idx',
    right_on='index',
    how='left'
).drop(['index', 'grouped_idx'], axis=1)

# Now find matches in the original dataframe based on name_it and average_prediction
matches = []
for idx, row in exploded_df.iterrows():
    # Find all matching rows in original dataframe
    matching_rows = df[(df['name'] == row['name']) & 
                        (df['average_prediction'] == row['average_prediction'])]
    
    if len(matching_rows) > 0:
        # Just take the first match if there are multiple (should be rare)
        matches.append(matching_rows.iloc[0]['row_id'])
    else:
        matches.append(None)

exploded_df['original_row_id'] = matches

# Now merge to get active_count from the original dataframe
exploded_df = pd.merge(
    exploded_df,
    df[['row_id', 'active_count']],
    left_on='original_row_id',
    right_on='row_id',
    how='left'
).drop(['row_id', 'original_row_id'], axis=1)

# Finally update plot_df with the correct active_count values
plot_df = exploded_df

# Create marker shape mapping based on active_count
plot_df['marker_symbol'] = plot_df['active_count'].apply(
    lambda x: 'star' if x == -1 else ('square' if 0 <= x <= 10 else 'circle')
)



# Create a box plot with custom-shaped scatter points
fig = make_subplots()

# Track added box plots to avoid duplicates
added_boxes = set()

# First add the box plots for each group
for group in plot_df['color_category_abbr'].unique():
    group_data = plot_df[plot_df['color_category_abbr'] == group]
    
    for name_it in group_data['name'].unique():
        box_key = (group, name_it)
        
        if box_key not in added_boxes:
            added_boxes.add(box_key)
            name_data = group_data[group_data['name'] == name_it]
            
            # Add box plot
            box = go.Box(
                y=name_data['average_prediction'],
                x=[name_it] * len(name_data),
                name=group,
                marker_color=color_discrete_map.get(group, '#888888'),
                boxpoints=False,  # Hide default points
                showlegend=name_data['name'].iloc[0] == group_data['name'].unique()[0]  # Only show in legend once
            )
            fig.add_trace(box)

# Now add individual scatter points with their correct active_count values
for i, row in plot_df.iterrows():
    scatter = go.Scatter(
        y=[row['average_prediction']],
        x=[row['name']],
        mode='markers',
        marker=dict(
            symbol=row['marker_symbol'],
            color=color_discrete_map.get(row['color_category_abbr'], '#888888'),
            size=8,
            line=dict(width=1, color='black')
        ),
        showlegend=False,
        hovertemplate="<b>%{x}</b><br>" +
                      "Value: %{y:.3f}<br>" +
                      "Status: " + row['comparison_status'] + "<br>" +
                      "Active Count: " + str(row['active_count']) + "<br>" +
                      "Mean: " + str(row['mean_value']) + "<br>" +
                      "<extra></extra>",
    )
    fig.add_trace(scatter)

# Update layout
fig.update_layout(
    title=f"Distribution of Average Predictions by Group (Mann-Whitney U test: α={alpha}, Ref={steering_vector_0_pos_mean:.3f})",
    xaxis_title="Group (Name + Direction)",
    yaxis_title="Average Prediction",
    boxmode='group',
    template='plotly_white',
    xaxis={
        'categoryorder':'array',
        'categoryarray': sorted(grouped_df['name'].unique()),
        'tickangle': -45  # Rotate labels
    },
    legend_title_text='Group Significance Status',
    hoverlabel=dict(
        bgcolor="white",
        font_size=12
    ),
    margin=dict(b=150)  # Increase bottom margin for rotated labels
)

# Apply opacity to all violin traces
fig.update_traces(opacity=0.7)

fig.show()
fig.write_html("plot.html")