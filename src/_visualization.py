import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import _util
import _mapping
import regex as re
import os

def get_accuracy(filepath, output_type="immediate", print_stats=False):
    # Read the CSV file
    df = pd.read_csv(filepath)
    df['generated_text'] = df['generated_text'].astype(str)

    # Initialize a list to store correctness results
    factual_results = []
    counterfactual_results = []

    # Process each row
    for index, row in df.iterrows():
        try:
            # Get the base and source sums
            base_sum = row['base_sum']
            source_sum = row['source_sum']
            
            # Get the generated output from generated_text
            generated_output = _util.get_final_output(row['generated_text'], output_type=output_type)
            
            # Check if they match
            if 'factual_output' in row and pd.notna(row['factual_output']):
                is_factual = str(int(row['factual_output'])) == str(generated_output)
            else:
                is_factual = str(int(base_sum)) == str(generated_output)
            if 'counterfactual_output' in row and pd.notna(row['counterfactual_output']):
                is_counterfactual = str(int(row['counterfactual_output'])) == str(generated_output)
            elif 'counterfactual_sum' in row and pd.notna(row['counterfactual_sum']):
                is_counterfactual = str(int(row['counterfactual_sum'])) == str(generated_output)
            else:
                is_counterfactual = str(int(source_sum)) == str(generated_output)
            factual_results.append(is_factual)
            counterfactual_results.append(is_counterfactual)
            
        except (ValueError, KeyError) as e:
            # If there's an error processing the row, mark as incorrect
            factual_results.append(False)
            counterfactual_results.append(False)
            print(f"Error processing row {index}: {e}")

    # Add the correctness column to the dataframe
    df['is_factual'] = factual_results
    df['is_counterfactual'] = counterfactual_results

    # Display some statistics
    if print_stats: 
        print(f"Total rows: {len(df)}")
        print(f"Factual answers: {sum(factual_results)}")
        print(f"Counterfactual answers: {sum(counterfactual_results)}")
        print(f"Factual accuracy: {sum(factual_results) / len(factual_results) * 100:.2f}%")
        print(f"Counterfactual accuracy: {sum(counterfactual_results) / len(counterfactual_results) * 100:.2f}%")

    return df

def plot_intervention_success_rate_by_location(df, intervene_ids_dict=_mapping.intervene_ids_stepwise_3_digit):
    
    # Create single plot for intervention success rate
    plt.figure(figsize=(10, 4))
    
    # Plot: Intervention success rate by intervention_id
    factual_ratios = df.groupby('intervention_id')['is_factual'].mean() * 100
    success_ratios = 100 - factual_ratios  # Invert the factual ratio
    
    # Define colors for different intervention types
    colors = []
    for intervention_id in success_ratios.index:
        if intervention_id in intervene_ids_dict["restatement"]:
            colors.append('green')
        elif intervention_id in intervene_ids_dict["reasoning"]:
            colors.append('blue')
        elif intervention_id in intervene_ids_dict["result"]:
            colors.append('orange')
        else:
            colors.append('gray')
        # elif intervention_id in intervention_location_type["reasoning_copy"]:
        #     colors.append('blue')
        # elif intervention_id in intervention_location_type["scale"]:
        #     colors.append('purple')
        # elif intervention_id in intervention_location_type["intermediate_sums"]:
        #     colors.append('green')
    
    # Create bar positions and labels
    x_positions = range(len(success_ratios))
    x_labels = [str(idx) for idx in success_ratios.index]
    
    plt.bar(x_positions, factual_ratios.values, color=colors, alpha=0.7)
    plt.title(f'Factual Answer Rate by Intervention ID (n={len(df[df["intervention_id"] == df["intervention_id"][0]])} per token)', fontsize=16)
    plt.xlabel('Intervention Token ID', fontsize=14)
    plt.ylabel('Factual Answer Rate (%)', fontsize=14)
    plt.ylim(0, 110)
    plt.grid(True, alpha=0.3)
    
    # Set x-axis ticks and labels
    plt.xticks(x_positions, x_labels)
    
    # Add value labels on bars
    for i, v in enumerate(factual_ratios.values):
        plt.text(i, v + 1, f'{v:.1f}%', ha='center', va='bottom')
    
    # Create legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', alpha=0.7, label='Restatement'),
        # Patch(facecolor='blue', alpha=0.7, label='Copy'),
        # Patch(facecolor='purple', alpha=0.7, label='Scale'),
        # Patch(facecolor='green', alpha=0.7, label='Intermediate Sums'),
        Patch(facecolor='blue', alpha=0.7, label='Reasoning'),
        Patch(facecolor='orange', alpha=0.7, label='Result'),
        # Patch(facecolor='gray', alpha=0.7, label='Other')
    ]
    plt.legend(handles=legend_elements, loc='lower right', fontsize=12)
    
    plt.tight_layout()
    plt.show()
    
    # Print detailed statistics
    print("\nDetailed Statistics by Intervention ID:")
    print("Intervention success rate ratios:")
    for intervention_id, ratio in success_ratios.items():
        total_count = len(df[df['intervention_id'] == intervention_id])
        print(f"  Intervention ID {intervention_id}: {ratio:.1f}% intervention success rate (n={total_count})")

def plot_intervention_results_by_location(df):
    # Create two subplots side by side
    plt.figure(figsize=(15, 6))
    
    # Plot 1: Factual accuracy by intervention_id
    plt.subplot(1, 2, 1)
    factual_ratios = df.groupby('intervention_id')['is_factual'].mean() * 100
    
    # Create bar positions and labels for only existing indices
    x_positions = range(len(factual_ratios))
    x_labels = [str(idx) for idx in factual_ratios.index]
    
    plt.bar(x_positions, factual_ratios.values, color='blue', alpha=0.7)
    plt.title('Factual Accuracy by Intervention ID')
    plt.xlabel('Intervention ID')
    plt.ylabel('Factual Accuracy (%)')
    plt.ylim(0, 105)
    plt.grid(True, alpha=0.3)
    
    # Set x-axis ticks and labels
    plt.xticks(x_positions, x_labels)
    
    # Add value labels on bars
    for i, v in enumerate(factual_ratios.values):
        plt.text(i, v + 1, f'{v:.0f}%', ha='center', va='bottom')
    
    # Plot 2: Counterfactual accuracy by intervention_id
    plt.subplot(1, 2, 2)
    counterfactual_ratios = df.groupby('intervention_id')['is_counterfactual'].mean() * 100
    
    # Create bar positions and labels for only existing indices
    x_positions = range(len(counterfactual_ratios))
    x_labels = [str(idx) for idx in counterfactual_ratios.index]
    
    plt.bar(x_positions, counterfactual_ratios.values, color='red', alpha=0.7)
    plt.title('Counterfactual Accuracy by Intervention ID')
    plt.xlabel('Intervention ID')
    plt.ylabel('Counterfactual Accuracy (%)')
    plt.ylim(0, 105)
    plt.grid(True, alpha=0.3)
    
    # Set x-axis ticks and labels
    plt.xticks(x_positions, x_labels)
    
    # Add value labels on bars
    for i, v in enumerate(counterfactual_ratios.values):
        plt.text(i, v + 1, f'{v:.0f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()
    
    # Print detailed statistics
    print("\nDetailed Statistics by Intervention ID:")
    print("Factual accuracy ratios:")
    for intervention_id, ratio in factual_ratios.items():
        total_count = len(df[df['intervention_id'] == intervention_id])
        print(f"  Intervention ID {intervention_id}: {ratio:.1f}% factual accuracy (n={total_count})")
    
    print("\nCounterfactual accuracy ratios:")
    for intervention_id, ratio in counterfactual_ratios.items():
        total_count = len(df[df['intervention_id'] == intervention_id])
        print(f"  Intervention ID {intervention_id}: {ratio:.1f}% counterfactual accuracy (n={total_count})")

def plot_intervention_results_by_type(df):

    # Plot 1: Overall factual accuracy
    plt.figure(figsize=(12, 8))

    # Overall accuracy
    overall_factual_acc = df['is_factual'].mean() * 100
    overall_counterfactual_acc = df['is_counterfactual'].mean() * 100

    plt.subplot(2, 3, 1)
    plt.bar(['Factual', 'Counterfactual'], [overall_factual_acc, overall_counterfactual_acc], 
            color=['blue', 'red'], alpha=0.7)
    plt.title('Overall Accuracy')
    plt.ylabel('Accuracy (%)')
    plt.ylim(0, 100)

    # Add percentage labels on bars
    plt.text(0, overall_factual_acc + 2, f'{overall_factual_acc:.1f}%', ha='center')
    plt.text(1, overall_counterfactual_acc + 2, f'{overall_counterfactual_acc:.1f}%', ha='center')

    # Filter columns to analyze
    filter_columns = ['in_restatement', 'in_reasoning', 'in_result', 'in_copy', 'in_intermediate_sums']

    # Plot accuracy for each filter
    for i, col in enumerate(filter_columns, 2):
        if col in df.columns:
            # Filter data where the column is True
            filtered_df = df[df[col] == True]
            
            if len(filtered_df) > 0:
                factual_acc = filtered_df['is_factual'].mean() * 100
                counterfactual_acc = filtered_df['is_counterfactual'].mean() * 100
                
                plt.subplot(2, 3, i)
                plt.bar(['Factual', 'Counterfactual'], [factual_acc, counterfactual_acc], 
                        color=['blue', 'red'], alpha=0.7)
                plt.title(f'{col}\n(n={len(filtered_df)})')
                plt.ylabel('Accuracy (%)')
                plt.ylim(0, 100)
                
                # Add percentage labels on bars
                plt.text(0, factual_acc + 2, f'{factual_acc:.1f}%', ha='center')
                plt.text(1, counterfactual_acc + 2, f'{counterfactual_acc:.1f}%', ha='center')
            else:
                plt.subplot(2, 3, i)
                plt.text(0.5, 0.5, 'No data', ha='center', va='center', transform=plt.gca().transAxes)
                plt.title(f'{col}\n(n=0)')

    plt.tight_layout()
    plt.show()

    # Print detailed statistics
    print("\nDetailed Statistics:")
    print(f"Overall - Factual: {overall_factual_acc:.2f}%, Counterfactual: {overall_counterfactual_acc:.2f}%")

    for col in filter_columns:
        if col in df.columns:
            filtered_df = df[df[col] == True]
            if len(filtered_df) > 0:
                factual_acc = filtered_df['is_factual'].mean() * 100
                counterfactual_acc = filtered_df['is_counterfactual'].mean() * 100
                print(f"{col} - Factual: {factual_acc:.2f}%, Counterfactual: {counterfactual_acc:.2f}% (n={len(filtered_df)})")
            else:
                print(f"{col} - No data available")


def plot_probability_distributions(probability_filepath):

    # Load and plot probability distributions
    prob_df = pd.read_csv(probability_filepath)

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Plot factual label probability distribution
    axes[0, 0].hist(prob_df['factual_label_probability'], bins=30, alpha=0.7, color='blue', edgecolor='black')
    axes[0, 0].set_title('Distribution of Factual Label Probability')
    axes[0, 0].set_xlabel('Factual Label Probability')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axvline(prob_df['factual_label_probability'].mean(), color='red', linestyle='--', 
                       label=f'Mean: {prob_df["factual_label_probability"].mean():.3f}')
    axes[0, 0].legend()

    # Plot counterfactual (source) label probability distribution
    axes[0, 1].hist(prob_df['counterfactual_label_probability'], bins=30, alpha=0.7, color='orange', edgecolor='black')
    axes[0, 1].set_title('Distribution of Counterfactual Label Probability')
    axes[0, 1].set_xlabel('Counterfactual Label Probability')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].axvline(prob_df['counterfactual_label_probability'].mean(), color='red', linestyle='--',
                       label=f'Mean: {prob_df["counterfactual_label_probability"].mean():.3f}')
    axes[0, 1].legend()

    # Filter data where factual probability is higher
    factual_higher = prob_df[prob_df['factual_label_probability'] > prob_df['counterfactual_label_probability']]
    
    # Plot distributions conditioned on factual probability being higher
    axes[1, 0].hist([factual_higher['factual_label_probability'], factual_higher['counterfactual_label_probability']], 
                    bins=20, alpha=0.7, color=['blue', 'orange'], edgecolor='black', 
                    label=['Factual', 'Counterfactual'])
    axes[1, 0].set_title(f'Probability Distributions when Factual > Counterfactual\n(n={len(factual_higher)})')
    axes[1, 0].set_xlabel('Probability')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()

    # Filter data where counterfactual probability is higher
    counterfactual_higher = prob_df[prob_df['counterfactual_label_probability'] > prob_df['factual_label_probability']]
    
    # Plot distributions conditioned on counterfactual probability being higher
    axes[1, 1].hist([counterfactual_higher['factual_label_probability'], counterfactual_higher['counterfactual_label_probability']], 
                    bins=20, alpha=0.7, color=['blue', 'orange'], edgecolor='black', 
                    label=['Factual', 'Counterfactual'])
    axes[1, 1].set_title(f'Probability Distributions when Counterfactual > Factual\n(n={len(counterfactual_higher)})')
    axes[1, 1].set_xlabel('Probability')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()

    plt.tight_layout()
    plt.show()

def plot_probability_distributions_simplified_1(probability_filepath):

    # Load and plot probability distributions
    prob_df = pd.read_csv(probability_filepath)

    # Create figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 5), sharey=True)

    # Filter data where factual probability is higher
    factual_higher = prob_df[prob_df['factual_label_probability'] > prob_df['counterfactual_label_probability']]
    
    # Plot distributions conditioned on factual probability being higher
    axes[0].hist([factual_higher['factual_label_probability'], factual_higher['counterfactual_label_probability']], 
                    bins=20, alpha=0.7, color=['blue', 'orange'], edgecolor='black', 
                    label=['Factual', 'Counterfactual'])
    axes[0].set_title(f'Distributions when Factual > Counterfactual (n={len(factual_higher)})', fontsize=20)
    axes[0].set_xlabel('Probability', fontsize=16)
    axes[0].set_ylabel('Frequency', fontsize=16)
    axes[0].set_ylim(0, 30)
    axes[0].grid(True, alpha=0.3)
    axes[0].tick_params(axis='both', which='major', labelsize=14)
    # axes[0].legend(fontsize=12)

    # Filter data where counterfactual probability is higher
    counterfactual_higher = prob_df[prob_df['counterfactual_label_probability'] >= prob_df['factual_label_probability']]
    
    # Plot distributions conditioned on counterfactual probability being higher
    axes[1].hist([counterfactual_higher['factual_label_probability'], counterfactual_higher['counterfactual_label_probability']], 
                    bins=20, alpha=0.7, color=['blue', 'orange'], edgecolor='black', 
                    label=['Factual', 'Counterfactual'])
    axes[1].set_title(f'Distributions when Counterfactual ≥ Factual (n={len(counterfactual_higher)})', fontsize=20)
    axes[1].set_xlabel('Probability', fontsize=16)
    axes[1].set_ylim(0, 30)
    axes[1].grid(True, alpha=0.3)
    axes[1].tick_params(axis='both', which='major', labelsize=14)
    axes[1].legend(fontsize=14)

    plt.tight_layout()
    plt.show()

def plot_probability_distributions_simplified_2(probability_filepath):

    # Load and plot probability distributions
    prob_df = pd.read_csv(probability_filepath)

    # Create figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=True)

    # Plot factual label probability distribution
    axes[0].hist(prob_df['factual_label_probability'], bins=30, alpha=0.7, color='blue', edgecolor='black')
    axes[0].set_title('Distribution of Factual Label Probability', fontsize=20)
    axes[0].set_xlabel('Factual Label Probability', fontsize=14)
    axes[0].set_ylabel('Frequency', fontsize=14)
    axes[0].grid(True, alpha=0.3)
    axes[0].axvline(prob_df['factual_label_probability'].mean(), color='red', linestyle='--', 
                   label=f'Mean: {prob_df["factual_label_probability"].mean():.3f}')
    axes[0].legend(fontsize=15)
    axes[0].tick_params(axis='both', which='major', labelsize=14)

    # Plot counterfactual (source) label probability distribution
    axes[1].hist(prob_df['counterfactual_label_probability'], bins=30, alpha=0.7, color='orange', edgecolor='black')
    axes[1].set_title('Distribution of Counterfactual Label Probability', fontsize=20)
    axes[1].set_xlabel('Counterfactual Label Probability', fontsize=14)
    axes[1].grid(True, alpha=0.3)
    axes[1].axvline(prob_df['counterfactual_label_probability'].mean(), color='red', linestyle='--',
                   label=f'Mean: {prob_df["counterfactual_label_probability"].mean():.3f}')
    axes[1].legend(fontsize=15)
    axes[1].tick_params(axis='both', which='major', labelsize=14)

    plt.tight_layout()
    plt.show()

def plot_prediction_probability_histogram_integrated(probability_filepath, title):

    # Load and plot probability histogram
    prob_df = pd.read_csv(probability_filepath)

    # Create figure with single plot
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))

    # Calculate histogram data for both probabilities
    factual_counts, factual_bins = np.histogram(prob_df['factual_label_probability'], bins=40, density=True)
    counterfactual_counts, counterfactual_bins = np.histogram(prob_df['counterfactual_label_probability'], bins=40, density=True)
    
    # Calculate bin centers for line plots
    factual_bin_centers = (factual_bins[:-1] + factual_bins[1:]) / 2
    counterfactual_bin_centers = (counterfactual_bins[:-1] + counterfactual_bins[1:]) / 2

    # Plot factual label probability histogram as line with filled area
    ax.plot(factual_bin_centers, factual_counts, color='blue', linewidth=2, label='Original')
    ax.fill_between(factual_bin_centers, factual_counts, alpha=0.3, color='blue')

    # Plot counterfactual label probability distribution as line with filled area
    ax.plot(counterfactual_bin_centers, counterfactual_counts, color='orange', linewidth=2, label='Counterfactual')
    ax.fill_between(counterfactual_bin_centers, counterfactual_counts, alpha=0.3, color='orange')

    # Set labels and title
    ax.set_title(f'Histogram for {title}', fontsize=20)
    ax.set_xlabel('Probability', fontsize=14)
    ax.set_ylabel('Frequency', fontsize=14)
    ax.grid(True, alpha=0.3)
    
    # Add mean lines
    ax.axvline(prob_df['factual_label_probability'].mean(), color='blue', linestyle='--', alpha=0.8,
               label=f'Original Mean: {prob_df["factual_label_probability"].mean():.3f}')
    ax.axvline(prob_df['counterfactual_label_probability'].mean(), color='orange', linestyle='--', alpha=0.8,
               label=f'Counterfactual Mean: {prob_df["counterfactual_label_probability"].mean():.3f}')
    
    ax.legend(fontsize=12)
    ax.tick_params(axis='both', which='major', labelsize=14)

    plt.tight_layout()
    plt.show()


def plot_prediction_probability_distributions_integrated_layerwise(probability_filepath, layers=None):

    # Load and plot probability distributions
    prob_df = pd.read_csv(probability_filepath)
    # Create 'other_label_probability' column
    prob_df['other_label_probability'] = 1 - prob_df['factual_label_probability'] - prob_df['counterfactual_label_probability']
    
    # If num_layers is not provided, determine it from the data
    if layers is None:
        if 'layer' in prob_df.columns:
            layers = range(prob_df['layer'].max() + 1)
        else:
            layers = [0]
    
    # Check if we have layer information
    if 'layer' not in prob_df.columns:
        prob_df['layer'] = 0  # Add layer column with default value
    
    # Create 3D plot
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # Colors for factual and counterfactual
    factual_color = 'blue'
    counterfactual_color = 'orange'
    other_color = 'green'
    
    for layer in layers:
        # Filter data for current layer
        layer_df = prob_df[prob_df['layer'] == layer]
        
        if len(layer_df) == 0:
            continue
            
        # Calculate histogram data for both distributions
        factual_counts, factual_bins = np.histogram(layer_df['factual_label_probability'], bins=20, range=(0, 1), density=False)
        counterfactual_counts, counterfactual_bins = np.histogram(layer_df['counterfactual_label_probability'], bins=20, range=(0, 1), density=False)
        
        # Calculate bin centers for line plots
        factual_bin_centers = (factual_bins[:-1] + factual_bins[1:]) / 2
        counterfactual_bin_centers = (counterfactual_bins[:-1] + counterfactual_bins[1:]) / 2

        # Create y-coordinates (layer positions)
        factual_y = np.full_like(factual_bin_centers, layer)
        counterfactual_y = np.full_like(counterfactual_bin_centers, layer)

        # Plot factual distribution
        ax.plot(factual_bin_centers, factual_y, factual_counts, 
                color=factual_color, linewidth=2, alpha=0.8)
        
        # Plot counterfactual distribution  
        ax.plot(counterfactual_bin_centers, counterfactual_y, counterfactual_counts,
                color=counterfactual_color, linewidth=2, alpha=0.8)
        
        # Fill the area under the curves
        # Create vertices for filled polygons
        factual_verts = [(factual_bin_centers[0], layer, 0)]
        for i in range(len(factual_bin_centers)):
            factual_verts.append((factual_bin_centers[i], layer, factual_counts[i]))
        factual_verts.append((factual_bin_centers[-1], layer, 0))
        
        counterfactual_verts = [(counterfactual_bin_centers[0], layer, 0)]
        for i in range(len(counterfactual_bin_centers)):
            counterfactual_verts.append((counterfactual_bin_centers[i], layer, counterfactual_counts[i]))
        counterfactual_verts.append((counterfactual_bin_centers[-1], layer, 0))
        
        # Add filled polygons
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        factual_poly = Poly3DCollection([factual_verts], alpha=0.3, facecolor=factual_color)
        counterfactual_poly = Poly3DCollection([counterfactual_verts], alpha=0.3, facecolor=counterfactual_color)
        ax.add_collection3d(factual_poly)
        ax.add_collection3d(counterfactual_poly)

        # # Plot other distribution
        # other_counts, other_bins = np.histogram(layer_df['other_label_probability'], bins=20, density=False)
        # other_bin_centers = (other_bins[:-1] + other_bins[1:]) / 2
        # other_y = np.full_like(other_bin_centers, layer)

        # # Plot other distribution
        # ax.plot(other_bin_centers, other_y, other_counts,
        #         color=other_color, linewidth=2, alpha=0.8)

        # other_verts = [(other_bin_centers[0], layer, 0)]
        # for i in range(len(other_bin_centers)):
        #     other_verts.append((other_bin_centers[i], layer, other_counts[i]))
        # other_verts.append((other_bin_centers[-1], layer, 0))

        # other_poly = Poly3DCollection([other_verts], alpha=0.3, facecolor=other_color)
        # ax.add_collection3d(other_poly)

    
    # Set labels and title
    ax.set_title('Distribution of Label Probabilities Across Layers', fontsize=16)
    ax.set_xlabel('Probability', fontsize=12)
    ax.set_ylabel('Layer', fontsize=12)
    ax.set_zlabel('Density', fontsize=12)
    
    # Set x-axis (probability) limits from 0 to 1
    ax.set_xlim(0, 1)
    
    # Set y-axis (layer) ticks to integers only
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    
    # Create custom legend
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], color=factual_color, lw=2, label='Factual'),
                      Line2D([0], [0], color=counterfactual_color, lw=2, label='Counterfactual')]
    ax.legend(handles=legend_elements, fontsize=12)
    
    plt.tight_layout()
    plt.show()


def plot_two_prediction_probability_distributions_integrated(probability_filepath_1, probability_filepath_2):

    # Load and plot probability distributions
    prob_df_1 = pd.read_csv(probability_filepath_1)
    prob_df_2 = pd.read_csv(probability_filepath_2)

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Plot for prob_df_1
    # Calculate histogram data for both distributions
    factual_counts_1, factual_bins_1 = np.histogram(prob_df_1['factual_label_probability'], bins=30, density=True)
    counterfactual_counts_1, counterfactual_bins_1 = np.histogram(prob_df_1['counterfactual_label_probability'], bins=30, density=True)
    
    # Calculate bin centers for line plots
    factual_bin_centers_1 = (factual_bins_1[:-1] + factual_bins_1[1:]) / 2
    counterfactual_bin_centers_1 = (counterfactual_bins_1[:-1] + counterfactual_bins_1[1:]) / 2

    # Plot factual label probability distribution as line with filled area
    ax1.plot(factual_bin_centers_1, factual_counts_1, color='blue', linewidth=2, label='Factual')
    ax1.fill_between(factual_bin_centers_1, factual_counts_1, alpha=0.3, color='blue')

    # Plot counterfactual label probability distribution as line with filled area
    ax1.plot(counterfactual_bin_centers_1, counterfactual_counts_1, color='orange', linewidth=2, label='Counterfactual')
    ax1.fill_between(counterfactual_bin_centers_1, counterfactual_counts_1, alpha=0.3, color='orange')

    # Set labels and title for first plot
    ax1.set_title('Non-Intervened Attention Pattern', fontsize=16)
    ax1.set_xlabel('Probability', fontsize=14)
    ax1.set_ylabel('Density', fontsize=14)
    ax1.grid(True, alpha=0.3)
    
    # Add mean lines for first plot
    ax1.axvline(prob_df_1['factual_label_probability'].mean(), color='blue', linestyle='--', alpha=0.8,
               label=f'Factual Mean: {prob_df_1["factual_label_probability"].mean():.3f}')
    ax1.axvline(prob_df_1['counterfactual_label_probability'].mean(), color='orange', linestyle='--', alpha=0.8,
               label=f'Counterfactual Mean: {prob_df_1["counterfactual_label_probability"].mean():.3f}')
    
    ax1.legend(fontsize=12)
    ax1.tick_params(axis='both', which='major', labelsize=14)

    # Plot for prob_df_2
    # Calculate histogram data for both distributions
    factual_counts_2, factual_bins_2 = np.histogram(prob_df_2['factual_label_probability'], bins=30, density=True)
    counterfactual_counts_2, counterfactual_bins_2 = np.histogram(prob_df_2['counterfactual_label_probability'], bins=30, density=True)
    
    # Calculate bin centers for line plots
    factual_bin_centers_2 = (factual_bins_2[:-1] + factual_bins_2[1:]) / 2
    counterfactual_bin_centers_2 = (counterfactual_bins_2[:-1] + counterfactual_bins_2[1:]) / 2

    # Plot factual label probability distribution as line with filled area
    ax2.plot(factual_bin_centers_2, factual_counts_2, color='blue', linewidth=2, label='Factual')
    ax2.fill_between(factual_bin_centers_2, factual_counts_2, alpha=0.3, color='blue')

    # Plot counterfactual label probability distribution as line with filled area
    ax2.plot(counterfactual_bin_centers_2, counterfactual_counts_2, color='orange', linewidth=2, label='Counterfactual')
    ax2.fill_between(counterfactual_bin_centers_2, counterfactual_counts_2, alpha=0.3, color='orange')

    # Set labels and title for second plot
    ax2.set_title('Intervened Attention Pattern', fontsize=16)
    ax2.set_xlabel('Probability', fontsize=14)
    ax2.set_ylabel('Density', fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    # Add mean lines for second plot
    ax2.axvline(prob_df_2['factual_label_probability'].mean(), color='blue', linestyle='--', alpha=0.8,
               label=f'Factual Mean: {prob_df_2["factual_label_probability"].mean():.3f}')
    ax2.axvline(prob_df_2['counterfactual_label_probability'].mean(), color='orange', linestyle='--', alpha=0.8,
               label=f'Counterfactual Mean: {prob_df_2["counterfactual_label_probability"].mean():.3f}')
    
    ax2.legend(fontsize=12)
    ax2.tick_params(axis='both', which='major', labelsize=14)

    plt.tight_layout()
    plt.show()

    
def has_wait(generated_text):
    return "?" in generated_text.lower()

def has_wait_before_answer(generated_text, answer):
    if answer in generated_text:
        if generated_text.split(answer)[0].lower().count("?") == 0:
            return 0
        else:
            return 1
    else:
        if generated_text.count("?") != 0:
            print("Has wait but no answer in generated text: ", generated_text)
        return 2

def plot_wait_ratio(df, filter_columns=['in_restatement', 'in_reasoning', 'in_result']):

    # Add wait analysis columns to accuracy_results
    df['has_wait'] = df['generated_text'].apply(has_wait)
    df['has_wait_before_answer'] = df.apply(
        lambda row: has_wait_before_answer(row['generated_text'], str(row['base_sum'])), axis=1
    )

    # Plot wait analysis
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle('Percentage of Responses with "Wait" by Category', fontsize=16)

    # Overall statistics
    overall_wait_pct = df['has_wait'].mean() * 100
    overall_wait_before_answer_0_pct = (df['has_wait_before_answer'] == 0).mean() * 100
    overall_wait_before_answer_1_pct = (df['has_wait_before_answer'] == 1).mean() * 100
    overall_wait_before_answer_2_pct = (df['has_wait_before_answer'] == 2).mean() * 100

    # Plot overall - has_wait
    plt.subplot(2, 4, 1)
    plt.bar(['Has Wait', 'No Wait'], 
            [overall_wait_pct, 100 - overall_wait_pct], 
            color=['green', 'gray'], alpha=0.7)
    plt.title(f'Overall - Has Wait\n(n={len(df)})')
    plt.ylabel('Percentage (%)')
    plt.ylim(0, 110)
    plt.text(0, overall_wait_pct + 2, f'{overall_wait_pct:.1f}%', ha='center')
    plt.text(1, (100 - overall_wait_pct) + 2, f'{100 - overall_wait_pct:.1f}%', ha='center')

    # Plot overall - has_wait_before_answer
    plt.subplot(2, 4, 5)
    plt.bar(['No', 'Yes', 'No Correct Answer'], 
            [overall_wait_before_answer_0_pct, overall_wait_before_answer_1_pct, overall_wait_before_answer_2_pct], 
            color=['green', 'gray'], alpha=0.7)
    plt.title(f'Overall - Wait Before Correct Answer\n(n={len(df)})')
    plt.ylabel('Percentage (%)')
    plt.ylim(0, 110)
    plt.text(0, overall_wait_before_answer_0_pct + 2, f'{overall_wait_before_answer_0_pct:.1f}%', ha='center')
    plt.text(1, overall_wait_before_answer_1_pct + 2, f'{overall_wait_before_answer_1_pct:.1f}%', ha='center')
    plt.text(2, overall_wait_before_answer_2_pct + 2, f'{overall_wait_before_answer_2_pct:.1f}%', ha='center')

    # Plot for each filter column
    for i, col in enumerate(filter_columns):
        if col in df.columns:
            filtered_df = df[df[col] == True]
            if len(filtered_df) > 0:
                wait_pct = filtered_df['has_wait'].mean() * 100
                wait_before_answer_0_pct = (filtered_df['has_wait_before_answer'] == 0).mean() * 100
                wait_before_answer_1_pct = (filtered_df['has_wait_before_answer'] == 1).mean() * 100
                wait_before_answer_2_pct = (filtered_df['has_wait_before_answer'] == 2).mean() * 100
                
                # Plot has_wait
                plt.subplot(2, 4, i + 2)
                plt.bar(['Has Wait', 'No Wait'], 
                        [wait_pct, 100 - wait_pct], 
                        color=['green', 'gray'], alpha=0.7)
                plt.title(f'{col} - Has Wait\n(n={len(filtered_df)})')
                plt.ylabel('Percentage (%)')
                plt.ylim(0, 110)
                plt.text(0, wait_pct + 2, f'{wait_pct:.1f}%', ha='center')
                plt.text(1, (100 - wait_pct) + 2, f'{100 - wait_pct:.1f}%', ha='center')
                
                # Plot has_wait_before_answer
                plt.subplot(2, 4, i + 6)
                plt.bar(['No', 'Yes', 'No Correct Answer'], 
                        [wait_before_answer_0_pct, wait_before_answer_1_pct, wait_before_answer_2_pct], 
                        color=['green', 'gray'], alpha=0.7)
                plt.title(f'{col} - Wait Before Answer\n(n={len(filtered_df)})')
                plt.ylabel('Percentage (%)')
                plt.ylim(0, 110)
                plt.text(0, wait_before_answer_0_pct + 2, f'{wait_before_answer_0_pct:.1f}%', ha='center')
                plt.text(1, wait_before_answer_1_pct + 2, f'{wait_before_answer_1_pct:.1f}%', ha='center')
                plt.text(2, wait_before_answer_2_pct + 2, f'{wait_before_answer_2_pct:.1f}%', ha='center')
            else:
                # Plot has_wait - no data
                plt.subplot(2, 4, i + 2)
                plt.text(0.5, 0.5, 'No data', ha='center', va='center', transform=plt.gca().transAxes)
                plt.title(f'{col} - Has Wait\n(n=0)')
                
                # Plot has_wait_before_answer - no data
                plt.subplot(2, 4, i + 6)
                plt.text(0.5, 0.5, 'No data', ha='center', va='center', transform=plt.gca().transAxes)
                plt.title(f'{col} - Wait Before Answer\n(n=0)')

    plt.tight_layout()
    plt.show()

def plot_wait_ratio_simplified(df, filter_columns=['in_restatement', 'in_reasoning', 'in_result']):

    # Add wait analysis columns to accuracy_results
    # df['has_wait'] = df['generated_text'].apply(has_wait)
    df['has_wait_before_answer'] = df.apply(
        lambda row: has_wait_before_answer(row['generated_text'], str(row['base_sum'])), axis=1
    )

    # Plot wait analysis
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    # fig.suptitle('Percentage of Responses with "Wait" - Overall Statistics', fontsize=16)

    # Overall statistics
    overall_wait_before_answer_0_pct = (df['has_wait_before_answer'] == 0).mean() * 100
    overall_wait_before_answer_1_pct = (df['has_wait_before_answer'] == 1).mean() * 100
    overall_wait_before_answer_2_pct = (df['has_wait_before_answer'] == 2).mean() * 100

    # Plot overall - has_wait_before_answer
    plt.subplot(1, 2, 1)
    plt.bar(['Has Correction Before Answer', 'No Correction Before Answer'], 
            [overall_wait_before_answer_1_pct, 100 - overall_wait_before_answer_1_pct], 
            color=['blue', 'orange'], alpha=0.7)
    plt.title(f'Has Correction Before Answer\n(n={len(df)})', fontsize=18)
    plt.ylabel('Percentage (%)', fontsize=14)
    plt.ylim(0, 110)
    plt.tick_params(axis='x', labelsize=12)
    plt.text(0, overall_wait_before_answer_1_pct + 2, f'{overall_wait_before_answer_1_pct:.1f}%', ha='center', fontsize=14)
    plt.text(1, (100 - overall_wait_before_answer_1_pct) + 2, f'{100 - overall_wait_before_answer_1_pct:.1f}%', ha='center', fontsize=14)

    # Plot overall - gets_correct_answer
    scaled_overall_wait_before_answer_0_pct = overall_wait_before_answer_0_pct / (overall_wait_before_answer_0_pct + overall_wait_before_answer_2_pct) * 100
    scaled_overall_wait_before_answer_2_pct = overall_wait_before_answer_2_pct / (overall_wait_before_answer_0_pct + overall_wait_before_answer_2_pct) * 100
    plt.subplot(1, 2, 2)
    plt.bar(['Correct Answer', 'Incorrect Answer', ], 
            [scaled_overall_wait_before_answer_0_pct, scaled_overall_wait_before_answer_2_pct], 
            color=['blue', 'orange'], alpha=0.7)
    plt.title(f'Gets Correct Answer (Given No Correction)\n(n={len(df[(df["has_wait_before_answer"] == 0) | (df["has_wait_before_answer"] == 2)])})', fontsize=18)
    plt.ylim(0, 110)
    plt.tick_params(axis='x', labelsize=12)
    plt.text(0, scaled_overall_wait_before_answer_0_pct + 2, f'{scaled_overall_wait_before_answer_0_pct:.1f}%', ha='center', fontsize=14)
    plt.text(1, scaled_overall_wait_before_answer_2_pct + 2, f'{scaled_overall_wait_before_answer_2_pct:.1f}%', ha='center', fontsize=14)

    plt.tight_layout()
    plt.show()


def plot_intervention_success_rate_layerwise(df):

    # Group by layer and calculate the mean of is_factual and is_counterfactual
    layer_stats = df.groupby('layer')[['is_factual', 'is_counterfactual']].mean() * 100

    # Create single plot
    fig, ax = plt.subplots(figsize=(5, 2.25))

    # Layer-wise analysis
    x = range(len(layer_stats.index))
    
    # Calculate the 'Other' proportion (1 - factual - counterfactual)
    other_proportion = 100 - layer_stats['is_factual'] - layer_stats['is_counterfactual']

    # Create stacked bars
    counterfactual_bars = ax.bar(x, layer_stats['is_counterfactual'], label='Counterfactual', color='#ff7f0e')
    factual_bars = ax.bar(x, layer_stats['is_factual'], bottom=layer_stats['is_counterfactual'], label='Factual', color='#1f77b4')
    other_bars = ax.bar(x, other_proportion, bottom=layer_stats['is_factual'] + layer_stats['is_counterfactual'], label='Other', color='#808080')

    # Print the height of counterfactual bars
    print("Counterfactual bar heights by layer:")
    for i, (layer, height) in enumerate(zip(layer_stats.index, layer_stats['is_counterfactual'])):
        print(f"Layer {layer}: {height:.2f}%")

    ax.set_xlabel('Layer')
    ax.set_ylabel('Output Frequency (%)')
    ax.set_xticks(x)
    ax.set_xticklabels(layer_stats.index, fontsize=7)
    ax.legend(fontsize=10, loc='upper right')
    # ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 110)

    # Overall title - adjust position to reduce gap
    fig.suptitle('Post-Intervention Output by Intervention Layer',fontsize=12.5)

    # Save the figure as SVG (before plt.show() to avoid saving empty figure)
    fig.savefig('../figures/default_layerwise.svg', format='svg', bbox_inches='tight')

    plt.tight_layout()
    plt.show()


def plot_intervention_effect_heatmap(directory, keyword):
    """
    Scan directory for files with name containing given keyword and starting with 'filtered'.
    Each file encodes x, y value in its name: ..._sum_{x}_intervention_effect_var20_l{y}.csv
    For each such file, compute the proportion of rows with is_counterfactual as the heatmap value.
    """
    # Prepare axis
    x_values = list(range(-4, 5))  # Now from -4 to 4
    y_values = list(range(0, 25))
    heatmap = np.full((len(y_values), len(x_values)), np.nan)  # rows: y, cols: x

    # File pattern: ..._sum_{x}_intervention_effect_var20_l{y}.csv
    fname_regex = re.compile(
        r'^filtered.*?_sum_([-+]?\d+)_.*' + re.escape(keyword) + r'.*?_l(\d+)\.csv$'
    )

    files = [
        f for f in os.listdir(directory)
        if (f.startswith("filtered") and (keyword in f))
    ]

    for fname in files:
        m = fname_regex.match(fname)
        if not m:
            continue
        x = int(m.group(1))
        y = int(m.group(2))
        if (x not in x_values) or (y not in y_values):
            continue
        file_path = os.path.join(directory, fname)
        # Use supplied function to get accuracy (returns a DataFrame with an is_counterfactual column)
        accuracy_results = get_accuracy(file_path, output_type="immediate")
        if "is_counterfactual" not in accuracy_results.columns:
            continue
        num_cf = accuracy_results["is_counterfactual"].sum()
        total = len(accuracy_results)
        if total > 0:
            ratio = num_cf / total
            heatmap[y, x+4] = ratio  # offset x so -4 is column 0

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(14, 8))
    # extent: [left, right, bottom, top], offset -4 instead of -10, and 4 instead of 10
    im = ax.imshow(heatmap, aspect='auto', origin='lower', cmap='viridis', extent=[-4-0.5, 4+0.5, 0-0.5, 24+0.5])

    # Set axis
    ax.set_xticks(x_values)
    ax.set_yticks(y_values)
    ax.set_xlabel('Intervention Sum (x)', fontsize=16)
    ax.set_ylabel('Layer (y)', fontsize=16)
    ax.set_title('Heatmap of Counterfactual Ratio by Intervention (x) and Layer (y)', fontsize=18)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Counterfactual Ratio", fontsize=15)

    # Annotate cells with value
    for y in y_values:
        for i, x in enumerate(x_values):
            val = heatmap[y, i]
            if not np.isnan(val):
                ax.text(x, y, f"{val:.2f}", ha='center', va='center', color='w', fontsize=10)

    plt.tight_layout()
    plt.show()