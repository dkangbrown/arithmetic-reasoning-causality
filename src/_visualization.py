import matplotlib.pyplot as plt
import pandas as pd
import _util
import _mapping

def get_accuracy(filepath):
    # Read the CSV file
    df = pd.read_csv(filepath)

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
            generated_output = _util.get_final_output(row['generated_text'])
            
            # Check if they match
            is_factual = str(base_sum) == str(generated_output)
            if 'counterfactual_sum' in row:
                is_counterfactual = str(row['counterfactual_sum']) == str(generated_output)
            else:
                is_counterfactual = str(source_sum) == str(generated_output)
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
    print(f"Total rows: {len(df)}")
    print(f"Factual answers: {sum(factual_results)}")
    print(f"Counterfactual answers: {sum(counterfactual_results)}")
    print(f"Factual accuracy: {sum(factual_results) / len(factual_results) * 100:.2f}%")
    print(f"Counterfactual accuracy: {sum(counterfactual_results) / len(counterfactual_results) * 100:.2f}%")

    return df

def plot_intervention_success_rate_by_location(df):
    # Load intervention location types
    intervention_location_type = _mapping.stepwise_intervene_loc_3_digit
    
    # Create single plot for intervention success rate
    plt.figure(figsize=(10, 6))
    
    # Plot: Intervention success rate by intervention_id
    factual_ratios = df.groupby('intervention_id')['is_factual'].mean() * 100
    success_ratios = 100 - factual_ratios  # Invert the factual ratio
    
    # Define colors for different intervention types
    colors = []
    for intervention_id in success_ratios.index:
        if intervention_id in intervention_location_type["restatement"]:
            colors.append('red')
        # elif intervention_id in intervention_location_type["reasoning_copy"]:
        #     colors.append('blue')
        # elif intervention_id in intervention_location_type["scale"]:
        #     colors.append('purple')
        # elif intervention_id in intervention_location_type["intermediate_sums"]:
        #     colors.append('green')
        if intervention_id in intervention_location_type["reasoning"]:
            colors.append('blue')
        elif intervention_id in intervention_location_type["result"]:
            colors.append('yellow')
        else:
            colors.append('gray')
    
    plt.bar(success_ratios.index, success_ratios.values, color=colors, alpha=0.7)
    plt.title('Intervention Success Rate by Intervention ID')
    plt.xlabel('Intervention ID')
    plt.ylabel('Intervention Success Rate (%)')
    plt.ylim(0, 100)
    plt.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, v in enumerate(success_ratios.values):
        plt.text(success_ratios.index[i], v + 1, f'{v:.1f}%', ha='center', va='bottom')
    
    # Create legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='red', alpha=0.7, label='Restatement'),
        # Patch(facecolor='blue', alpha=0.7, label='Copy'),
        # Patch(facecolor='purple', alpha=0.7, label='Scale'),
        # Patch(facecolor='green', alpha=0.7, label='Intermediate Sums'),
        Patch(facecolor='blue', alpha=0.7, label='Reasoning'),
        Patch(facecolor='orange', alpha=0.7, label='Result'),
        Patch(facecolor='gray', alpha=0.7, label='Other')
    ]
    plt.legend(handles=legend_elements, loc='upper right')
    
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
    plt.bar(factual_ratios.index, factual_ratios.values, color='blue', alpha=0.7)
    plt.title('Factual Accuracy by Intervention ID')
    plt.xlabel('Intervention ID')
    plt.ylabel('Factual Accuracy (%)')
    plt.ylim(0, 100)
    plt.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, v in enumerate(factual_ratios.values):
        plt.text(factual_ratios.index[i], v + 1, f'{v:.1f}%', ha='center', va='bottom')
    
    # Plot 2: Counterfactual accuracy by intervention_id
    plt.subplot(1, 2, 2)
    counterfactual_ratios = df.groupby('intervention_id')['is_counterfactual'].mean() * 100
    plt.bar(counterfactual_ratios.index, counterfactual_ratios.values, color='red', alpha=0.7)
    plt.title('Counterfactual Accuracy by Intervention ID')
    plt.xlabel('Intervention ID')
    plt.ylabel('Counterfactual Accuracy (%)')
    plt.ylim(0, 100)
    plt.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, v in enumerate(counterfactual_ratios.values):
        plt.text(counterfactual_ratios.index[i], v + 1, f'{v:.1f}%', ha='center', va='bottom')
    
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
    axes[0, 1].set_title('Distribution of Source Label Probability')
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

    
