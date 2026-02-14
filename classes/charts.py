import numpy as np
from collections import defaultdict
from typing import List, Dict
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math

class Charts:
    
    def __init__(self, factuality_weights=None, robustness_weights=None):
        """
        Initialize Charts with configurable metric weights.
        
        Args:
            factuality_weights: Dict with keys 'is_correct', 'score', 'meteor' 
                               Default: {'is_correct': 0.5, 'score': 0.2, 'meteor': 0.3}
            robustness_weights: Dict with keys 'is_correct', 'similarity_score', 'f1_score'
                               Default: {'is_correct': 0.6, 'similarity_score': 0.2, 'f1_score': 0.2}
        """
        self.df = None
        
        # Set default weights for factuality
        self.factuality_weights = factuality_weights or {
            'is_correct': 0.5,
            'score': 0.2,
            'meteor': 0.3
        }
        
        # Set default weights for robustness
        self.robustness_weights = robustness_weights or {
            'is_correct': 0.6,
            'similarity_score': 0.2,
            'f1_score': 0.2
        }
    
    def _format_group_key(self, group_key):
        """
        Format group_key appropriately based on value.
        - Values between 0-1: format as percentage (poison ratios)
        - Values > 1: format as integer (distractor counts)
        """
        if isinstance(group_key, (int, float)):
            if 0 <= group_key <= 1:
                return f"{group_key:.0%}"
            else:
                return str(int(group_key))
        return str(group_key)
    
    def analyze_results(self, results: List[Dict]) -> pd.DataFrame:
        """Convert results to a pandas DataFrame for easier analysis."""
        
        rows = []
        
        for result in results:
            # Get experiment type directly from the result
            experiment_type = result.get("experiment", "unknown")
            
            # Determine the group key based on what's present
            has_poison_ratio = result.get("poison_ratio") is not None
            
            if has_poison_ratio:
                group_key = result["poison_ratio"]
            else:
                group_key = result.get("distractors_numbers", "Unknown")
            
            # Extract evaluation data
            if 'evaluation' not in result:
                continue
                
            eval_data = result['evaluation']
            
            for model, model_strategies in eval_data.items():
                for strategy, categories in model_strategies.items():
                    
                    # Extract factuality metrics
                    if "factuality" in categories:
                        f_metrics = categories["factuality"]
                        
                        # Safely get values with default 0, ensuring they're floats
                        is_correct = float(f_metrics.get("is_correct", 0) or 0)
                        score = float(f_metrics.get("score", 0) or 0)
                        meteor = float(f_metrics.get("meteor", 0) or 0)
                        
                        # Use configurable weights
                        factuality_score = (
                            self.factuality_weights['is_correct'] * is_correct + 
                            self.factuality_weights['score'] * score + 
                            self.factuality_weights['meteor'] * meteor
                        )
                        
                        rows.append({
                            'experiment': experiment_type,
                            'model': model,
                            'group_key': group_key,
                            'strategy': strategy,
                            'metric': 'factuality',
                            'value': float(factuality_score)
                        })
                    
                    # Extract robustness metrics
                    if "robustness" in categories:
                        r_metrics = categories["robustness"]
                        
                        # Safely get values with default 0, ensuring they're floats
                        is_correct = float(r_metrics.get("is_correct", 0) or 0)
                        similarity = float(r_metrics.get("similarity_score", 0) or 0)
                        f1 = float(r_metrics.get("f1_score", 0) or 0)
                        
                        # Use configurable weights
                        robustness_score = (
                            self.robustness_weights['is_correct'] * is_correct + 
                            self.robustness_weights['similarity_score'] * similarity + 
                            self.robustness_weights['f1_score'] * f1
                        )
                        
                        rows.append({
                            'experiment': experiment_type,
                            'model': model,
                            'group_key': group_key,
                            'strategy': strategy,
                            'metric': 'robustness',
                            'value': float(robustness_score)
                        })
                    
                    # Extract accountability metrics
                    if "accountability" in categories:
                        a_metrics = categories["accountability"]
                        
                        # Safely get citation_f1
                        citation_f1 = float(a_metrics.get("citation_f1", 0) or 0)
                        
                        rows.append({
                            'experiment': experiment_type,
                            'model': model,
                            'group_key': group_key,
                            'strategy': strategy,
                            'metric': 'accountability_f1',
                            'value': float(citation_f1)
                        })
        
        self.df = pd.DataFrame(rows)
        # Ensure value column is numeric
        if not self.df.empty:
            self.df['value'] = pd.to_numeric(self.df['value'], errors='coerce')
            self.df = self.df.dropna(subset=['value'])
        
        return self.df

    def aggregate_results(self) -> pd.DataFrame:
        """Aggregate results by taking the mean of each metric."""
        
        if self.df is None or self.df.empty:
            print("No data to aggregate. Run analyze_results first.")
            return pd.DataFrame()
        
        # Group by all dimensions and calculate mean
        aggregated = self.df.groupby(
            ['experiment', 'model', 'group_key', 'strategy', 'metric']
        )['value'].mean().reset_index()
        
        return aggregated

    def print_summary(self, config=None):
        """Print a clean, organized summary for each experiment type."""
        
        if self.df is None or self.df.empty:
            print("No data available.")
            return
        
        df = self.aggregate_results()
        
        models = sorted(df['model'].unique())
        print(f"Models evaluated: {models}")
        
        # Define which metrics to show for each experiment type
        experiment_configs = {
            'factuality': {
                'title': 'FACTUALITY EXPERIMENT',
                'metrics': [('Factuality Score', 'factuality', '{:.1%}')]
            },
            'robustness': {
                'title': 'ROBUSTNESS EXPERIMENT',
                'metrics': [
                    ('Robustness Score', 'robustness', '{:.1%}'),
                ]
            },
            'accountability': {
                'title': 'ACCOUNTABILITY EXPERIMENT',
                'metrics': [
                    ('Citation F1', 'accountability_f1', '{:.3f}')
                ]
            }
        }
        
        # Process each experiment type
        for exp_type in ['factuality', 'robustness', 'accountability']:
            exp_df = df[df['experiment'] == exp_type]
            
            if exp_df.empty:
                continue
            
            config_info = experiment_configs.get(exp_type, {
                'title': f'{exp_type.upper()} EXPERIMENT',
                'metrics': []
            })
            
            print(f"\n{'='*10} {config_info['title']} {'='*10}")
            
            # Get metrics for this experiment
            metrics_to_show = config_info.get('metrics', [])
            
            for metric_name, metric_key, fmt in metrics_to_show:
                metric_df = exp_df[exp_df['metric'] == metric_key]
                
                if metric_df.empty:
                    continue
                
                # Create pivot table
                pivot = metric_df.pivot_table(
                    index=['group_key', 'strategy'],
                    columns='model',
                    values='value',
                    aggfunc='mean'
                )
                
                if pivot.empty:
                    continue
                
                print(f"\n{metric_name}:")
                
                # Print header
                header = f"{'Input':<10} {'Strategy':<15}"
                for model in pivot.columns:
                    header += f" {model:>15}"
                print(header)
                print("─" * len(header))
                
                # Print rows
                for idx, row in pivot.iterrows():
                    group_key, strategy = idx
                    
                    # Format group_key (handle both float percentages and integers)
                    # If it's a float between 0 and 1, format as percentage
                    # Otherwise, format as integer (distractor count)
                    if isinstance(group_key, (int, float)):
                        if 0 <= group_key <= 1:
                            group_str = f"{group_key:<10.0%}"
                        else:
                            group_str = f"{int(group_key):<10}"
                    else:
                        group_str = f"{str(group_key):<10}"
                    
                    row_str = f"{group_str} {strategy:<15}"
                    
                    for model in pivot.columns:
                        value = row[model]
                        if pd.notna(value):
                            formatted_value = fmt.format(value)
                        else:
                            formatted_value = ""
                        row_str += f" {formatted_value:>15}"
                    
                    print(row_str)
                
                print("─" * len(header))

    def plot_metric_barchart(self, metric_name, experiment_type=None):
        """
        Plot a bar chart for a specific metric.
        
        Args:
            metric_name: Name of the metric to plot (e.g., 'factuality', 'robustness', 'accountability_f1')
            experiment_type: Optional filter for experiment type ('factuality', 'robustness', 'accountability')
        """
        if self.df is None or self.df.empty:
            print("No data available. Run analyze_results first.")
            return
        
        # Get aggregated data
        df = self.aggregate_results()
        
        # Filter for the specific metric
        plot_data = df[df['metric'] == metric_name].copy()
        
        # Optionally filter by experiment type
        if experiment_type:
            plot_data = plot_data[plot_data['experiment'] == experiment_type]
        
        # Remove any NaN values
        plot_data = plot_data.dropna(subset=['value'])
        
        if plot_data.empty:
            print(f"No data found for metric '{metric_name}'" + 
                  (f" in experiment '{experiment_type}'" if experiment_type else ""))
            return
        
        # Ensure value column is float
        plot_data['value'] = plot_data['value'].astype(float)
        
        # Format group_key for display using helper method
        plot_data['Parameter'] = plot_data['group_key'].apply(self._format_group_key)
        
        # Get unique parameters for subplot arrangement
        n_params = plot_data['Parameter'].nunique()
        
        # Create the chart
        try:
            g = sns.catplot(
                data=plot_data,
                x='model',
                y='value',
                hue='strategy',
                col='Parameter',
                kind='bar',
                col_wrap=min(3, n_params),  # Max 3 columns
                height=5,
                aspect=1.2,
                palette='Set2'
            )
            
            # Adjust titles and labels
            title = f'Comparison of {metric_name}'
            if experiment_type:
                title += f' ({experiment_type} experiment)'
            g.fig.suptitle(title, y=1.02, fontsize=16, fontweight='bold')
            g.set_axis_labels("Model", "Score")
            
            # Rotate model names if they overlap
            for ax in g.axes.flat:
                for label in ax.get_xticklabels():
                    label.set_rotation(45)
                    label.set_ha('right')
                ax.grid(axis='y', alpha=0.3)
            
            plt.show()
        except Exception as e:
            print(f"Error creating plot: {e}")
            print(f"Data shape: {plot_data.shape}")
            print(f"Available columns: {plot_data.columns.tolist()}")

    def plot_experiment_overview(self, experiment_type):
        """
        Plot all metrics for a specific experiment type in a single figure.
        
        Args:
            experiment_type: 'factuality', 'robustness', or 'accountability'
        """
        if self.df is None or self.df.empty:
            print("No data available.")
            return
        
        df = self.aggregate_results()
        exp_data = df[df['experiment'] == experiment_type].copy()
        
        if exp_data.empty:
            print(f"No data found for experiment '{experiment_type}'")
            return
        
        # Get unique metrics for this experiment
        metrics = sorted(exp_data['metric'].unique())
        n_metrics = len(metrics)
        
        if n_metrics == 0:
            print("No metrics found")
            return
        
        # Format group_key for display using helper method
        exp_data['Parameter'] = exp_data['group_key'].apply(self._format_group_key)
        
        # Create subplots
        fig, axes = plt.subplots(1, n_metrics, figsize=(6*n_metrics, 6))
        
        if n_metrics == 1:
            axes = [axes]
        
        for idx, metric in enumerate(metrics):
            metric_data = exp_data[exp_data['metric'] == metric]
            
            # Create bar plot
            ax = axes[idx]
            
            # Get unique models and strategies for proper grouping
            models = sorted(metric_data['model'].unique())
            strategies = sorted(metric_data['strategy'].unique())
            
            # Prepare data for grouped bar chart
            x = np.arange(len(models))
            width = 0.8 / len(strategies)
            
            for i, strategy in enumerate(strategies):
                strategy_data = metric_data[metric_data['strategy'] == strategy]
                values = [strategy_data[strategy_data['model'] == m]['value'].mean() 
                         for m in models]
                
                offset = (i - len(strategies)/2) * width + width/2
                ax.bar(x + offset, values, width, label=strategy, alpha=0.8)
            
            ax.set_xlabel('Model', fontweight='bold')
            ax.set_ylabel('Score', fontweight='bold')
            ax.set_title(f'{metric}', fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(models, rotation=45, ha='right')
            ax.legend()
            ax.grid(axis='y', alpha=0.3)
            
            # Format y-axis as percentage for certain metrics
            if metric in ['factuality', 'robustness']:
                ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
        
        plt.suptitle(f'{experiment_type.upper()} Experiment Overview', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()

    def plot_strategy_comparison(self, metric_name, experiment_type=None):
        """
        Plot a heatmap comparing strategies across models and parameters.
        
        Args:
            metric_name: Name of the metric to plot
            experiment_type: Optional filter for experiment type
        """
        if self.df is None or self.df.empty:
            print("No data available.")
            return
        
        df = self.aggregate_results()
        plot_data = df[df['metric'] == metric_name].copy()
        
        if experiment_type:
            plot_data = plot_data[plot_data['experiment'] == experiment_type]
        
        if plot_data.empty:
            print(f"No data found for metric '{metric_name}'" + 
                  (f" in experiment '{experiment_type}'" if experiment_type else ""))
            return
        
        # Get unique strategies
        strategies = sorted(plot_data['strategy'].unique())
        n_strategies = len(strategies)
        
        if n_strategies == 0:
            print("No strategies found")
            return
        
        # Create subplots for each strategy
        fig, axes = plt.subplots(1, n_strategies, figsize=(6*n_strategies, 5))
        
        if n_strategies == 1:
            axes = [axes]
        
        for idx, strategy in enumerate(strategies):
            strategy_data = plot_data[plot_data['strategy'] == strategy]
            
            # Create pivot table for heatmap
            pivot = strategy_data.pivot_table(
                index='group_key',
                columns='model',
                values='value',
                aggfunc='mean'
            )
            
            if pivot.empty:
                axes[idx].set_visible(False)
                continue
            
            # Format index for display using helper method
            pivot.index = [self._format_group_key(x) for x in pivot.index]
            
            # Create heatmap
            sns.heatmap(
                pivot,
                annot=True,
                fmt='.3f',
                cmap='YlGnBu',
                ax=axes[idx],
                cbar_kws={'label': 'Score'},
                linewidths=1,
                linecolor='white'
            )
            
            axes[idx].set_title(f'{strategy}', fontweight='bold')
            axes[idx].set_xlabel('Model', fontweight='bold')
            axes[idx].set_ylabel('Parameter', fontweight='bold')
        
        title = f'Strategy Comparison - {metric_name}'
        if experiment_type:
            title += f' ({experiment_type})'
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()

    def plot_line_comparison(self, experiment_type=None):
        """
        Plot line charts comparing strategies within each model across different parameters.
        
        Args:
            experiment_type: Optional filter for experiment type
        """
        if self.df is None or self.df.empty:
            print("No data available.")
            return
        
        df = self.aggregate_results()
        plot_data = df.copy()
        
        if experiment_type:
            plot_data = plot_data[plot_data['experiment'] == experiment_type]
        
        if plot_data.empty:
            print("No data to plot")
            return
        
        # Get unique metrics and models
        metrics = sorted(plot_data['metric'].unique())
        models = sorted(plot_data['model'].unique())
        
        n_metrics = len(metrics)
        n_models = len(models)
        
        if n_metrics == 0 or n_models == 0:
            print("No metrics or models found")
            return
        
        # Create a grid of subplots (rows=metrics, cols=models)
        fig, axes = plt.subplots(n_metrics, n_models, 
                                 figsize=(5*n_models, 4*n_metrics))
        
        if n_metrics == 1 and n_models == 1:
            axes = np.array([[axes]])
        elif n_metrics == 1:
            axes = axes.reshape(1, -1)
        elif n_models == 1:
            axes = axes.reshape(-1, 1)
        
        # Color palette for strategies
        strategies = sorted(plot_data['strategy'].unique())
        colors = plt.cm.Set2(np.linspace(0, 1, len(strategies)))
        strategy_colors = dict(zip(strategies, colors))
        
        for i, metric in enumerate(metrics):
            for j, model in enumerate(models):
                ax = axes[i, j]
                
                # Filter data for this metric and model
                subset = plot_data[
                    (plot_data['metric'] == metric) & 
                    (plot_data['model'] == model)
                ].copy()
                
                if subset.empty:
                    ax.set_visible(False)
                    continue
                
                # Sort by group_key for line plot
                subset = subset.sort_values('group_key')
                
                # Plot line for each strategy
                for strategy in strategies:
                    strategy_data = subset[subset['strategy'] == strategy]
                    if not strategy_data.empty:
                        ax.plot(strategy_data['group_key'].values, 
                               strategy_data['value'].values,
                               marker='o', 
                               label=strategy, 
                               linewidth=2.5, 
                               markersize=7,
                               color=strategy_colors[strategy])
                
                ax.set_title(f'{model}\n{metric}', fontweight='bold', fontsize=10)
                ax.set_xlabel('Parameter', fontweight='bold', fontsize=9)
                ax.set_ylabel('Score', fontweight='bold', fontsize=9)
                ax.legend(fontsize=8, loc='best')
                ax.grid(True, alpha=0.3)
                
                # Format x-axis based on group_key type
                x_values = subset['group_key'].unique()
                if len(x_values) > 0:
                    # Check if these are ratios (0-1) or counts (>1)
                    if all(0 <= x <= 1 for x in x_values):
                        # Poison ratios - format as percentages
                        ax.xaxis.set_major_formatter(
                            plt.FuncFormatter(lambda x, _: f'{x:.0%}')
                        )
                    else:
                        # Distractor counts - format as integers
                        ax.set_xticks(sorted(x_values))
                        ax.set_xticklabels([str(int(x)) for x in sorted(x_values)])
                
                # Format y-axis as percentage for certain metrics
                if metric in ['factuality', 'robustness']:
                    ax.yaxis.set_major_formatter(
                        plt.FuncFormatter(lambda y, _: f'{y:.0%}')
                    )
        
        title = 'Strategy Comparison by Model'
        if experiment_type:
            title += f' - {experiment_type.upper()}'
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()

    def plot_model_comparison_lines(self, experiment_type=None):
        """
        Plot line charts comparing models within each strategy across different parameters.
        Alternative to plot_line_comparison - shows model differences instead of strategy differences.
        
        Args:
            experiment_type: Optional filter for experiment type
        """
        if self.df is None or self.df.empty:
            print("No data available.")
            return
        
        df = self.aggregate_results()
        plot_data = df.copy()
        
        if experiment_type:
            plot_data = plot_data[plot_data['experiment'] == experiment_type]
        
        if plot_data.empty:
            print("No data to plot")
            return
        
        # Get unique metrics and strategies
        metrics = sorted(plot_data['metric'].unique())
        strategies = sorted(plot_data['strategy'].unique())
        
        n_metrics = len(metrics)
        n_strategies = len(strategies)
        
        if n_metrics == 0 or n_strategies == 0:
            print("No metrics or strategies found")
            return
        
        # Create a grid of subplots (rows=metrics, cols=strategies)
        fig, axes = plt.subplots(n_metrics, n_strategies, 
                                 figsize=(5*n_strategies, 4*n_metrics))
        
        if n_metrics == 1 and n_strategies == 1:
            axes = np.array([[axes]])
        elif n_metrics == 1:
            axes = axes.reshape(1, -1)
        elif n_strategies == 1:
            axes = axes.reshape(-1, 1)
        
        # Color palette for models
        models = sorted(plot_data['model'].unique())
        colors = plt.cm.tab10(np.linspace(0, 1, len(models)))
        model_colors = dict(zip(models, colors))
        
        for i, metric in enumerate(metrics):
            for j, strategy in enumerate(strategies):
                ax = axes[i, j]
                
                # Filter data for this metric and strategy
                subset = plot_data[
                    (plot_data['metric'] == metric) & 
                    (plot_data['strategy'] == strategy)
                ].copy()
                
                if subset.empty:
                    ax.set_visible(False)
                    continue
                
                # Sort by group_key for line plot
                subset = subset.sort_values('group_key')
                
                # Plot line for each model
                for model in models:
                    model_data = subset[subset['model'] == model]
                    if not model_data.empty:
                        ax.plot(model_data['group_key'].values, 
                               model_data['value'].values,
                               marker='s', 
                               label=model, 
                               linewidth=2.5, 
                               markersize=7,
                               color=model_colors[model])
                
                ax.set_title(f'{strategy}\n{metric}', fontweight='bold', fontsize=10)
                ax.set_xlabel('Parameter', fontweight='bold', fontsize=9)
                ax.set_ylabel('Score', fontweight='bold', fontsize=9)
                ax.legend(fontsize=8, loc='best')
                ax.grid(True, alpha=0.3)
                
                # Format x-axis based on group_key type
                x_values = subset['group_key'].unique()
                if len(x_values) > 0:
                    # Check if these are ratios (0-1) or counts (>1)
                    if all(0 <= x <= 1 for x in x_values):
                        # Poison ratios - format as percentages
                        ax.xaxis.set_major_formatter(
                            plt.FuncFormatter(lambda x, _: f'{x:.0%}')
                        )
                    else:
                        # Distractor counts - format as integers
                        ax.set_xticks(sorted(x_values))
                        ax.set_xticklabels([str(int(x)) for x in sorted(x_values)])
                
                # Format y-axis as percentage for certain metrics
                if metric in ['factuality', 'robustness']:
                    ax.yaxis.set_major_formatter(
                        plt.FuncFormatter(lambda y, _: f'{y:.0%}')
                    )
        
        title = 'Model Comparison by Strategy'
        if experiment_type:
            title += f' - {experiment_type.upper()}'
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()


    def plot_dimension_heatmap_by_strategy(self):
        """
        Plot a heatmap comparing Factuality, Robustness, and Accountability across strategies.
        Aggregates across all models and parameters to show overall strategy performance.
        """
        if self.df is None or self.df.empty:
            print("No data available.")
            return
        
        df = self.aggregate_results()
        
        # Map metric names to dimension categories
        dimension_map = {
            'factuality': 'Factuality',
            'robustness': 'Robustness',
            'accountability_f1': 'Accountability'
        }
        
        # Filter for the three main dimensions
        df_filtered = df[df['metric'].isin(dimension_map.keys())].copy()
        
        if df_filtered.empty:
            print("No dimension data found.")
            return
        
        # Map metric names to dimension names
        df_filtered['dimension'] = df_filtered['metric'].map(dimension_map)
        
        # Aggregate across all models and parameters
        pivot = df_filtered.groupby(['strategy', 'dimension'])['value'].mean().reset_index()
        pivot_table = pivot.pivot(index='strategy', columns='dimension', values='value')
        
        # Reorder columns to consistent order
        desired_order = ['Factuality', 'Robustness', 'Accountability']
        pivot_table = pivot_table[[col for col in desired_order if col in pivot_table.columns]]
        
        # Sort strategies alphabetically
        pivot_table = pivot_table.sort_index()
        
        # Create heatmap
        plt.figure(figsize=(10, 6))
        
        sns.heatmap(
            pivot_table,
            annot=True,
            fmt='.3f',
            cmap='RdYlGn',
            center=0.5,
            vmin=0,
            vmax=1,
            linewidths=2,
            linecolor='white',
            cbar_kws={'label': 'Score'},
            square=True
        )
        
        plt.title('Performance Dimensions by Strategy\n(Averaged across all models and parameters)', 
                 fontsize=14, fontweight='bold', pad=15)
        plt.xlabel('Dimension', fontsize=12, fontweight='bold')
        plt.ylabel('Strategy', fontsize=12, fontweight='bold')
        plt.xticks(rotation=0)
        plt.yticks(rotation=0)
        
        plt.tight_layout()
        plt.show()
    
    def plot_dimension_heatmap_by_model(self):
        """
        Plot a heatmap comparing Factuality, Robustness, and Accountability across models.
        Aggregates across all strategies and parameters to show overall model performance.
        """
        if self.df is None or self.df.empty:
            print("No data available.")
            return
        
        df = self.aggregate_results()
        
        # Map metric names to dimension categories
        dimension_map = {
            'factuality': 'Factuality',
            'robustness': 'Robustness',
            'accountability_f1': 'Accountability'
        }
        
        # Filter for the three main dimensions
        df_filtered = df[df['metric'].isin(dimension_map.keys())].copy()
        
        if df_filtered.empty:
            print("No dimension data found.")
            return
        
        # Map metric names to dimension names
        df_filtered['dimension'] = df_filtered['metric'].map(dimension_map)
        
        # Aggregate across all strategies and parameters
        pivot = df_filtered.groupby(['model', 'dimension'])['value'].mean().reset_index()
        pivot_table = pivot.pivot(index='model', columns='dimension', values='value')
        
        # Reorder columns to consistent order
        desired_order = ['Factuality', 'Robustness', 'Accountability']
        pivot_table = pivot_table[[col for col in desired_order if col in pivot_table.columns]]
        
        # Sort models alphabetically
        pivot_table = pivot_table.sort_index()
        
        # Create heatmap
        plt.figure(figsize=(10, 6))
        
        sns.heatmap(
            pivot_table,
            annot=True,
            fmt='.3f',
            cmap='RdYlGn',
            center=0.5,
            vmin=0,
            vmax=1,
            linewidths=2,
            linecolor='white',
            cbar_kws={'label': 'Score'},
            square=True
        )
        
        plt.title('Performance Dimensions by Model\n(Averaged across all strategies and parameters)', 
                 fontsize=14, fontweight='bold', pad=15)
        plt.xlabel('Dimension', fontsize=12, fontweight='bold')
        plt.ylabel('Model', fontsize=12, fontweight='bold')
        plt.xticks(rotation=0)
        plt.yticks(rotation=0)
        
        plt.tight_layout()
        plt.show()
    
    def plot_dimension_comparison_combined(self):
        """
        Plot both strategy and model dimension heatmaps side by side.
        """
        if self.df is None or self.df.empty:
            print("No data available.")
            return
        
        df = self.aggregate_results()
        
        # Map metric names to dimension categories
        dimension_map = {
            'factuality': 'Factuality',
            'robustness': 'Robustness',
            'accountability_f1': 'Accountability'
        }
        
        # Filter for the three main dimensions
        df_filtered = df[df['metric'].isin(dimension_map.keys())].copy()
        
        if df_filtered.empty:
            print("No dimension data found.")
            return
        
        # Map metric names to dimension names
        df_filtered['dimension'] = df_filtered['metric'].map(dimension_map)
        
        # Create figure with two subplots
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Left: Strategy comparison
        pivot_strategy = df_filtered.groupby(['strategy', 'dimension'])['value'].mean().reset_index()
        pivot_table_strategy = pivot_strategy.pivot(index='strategy', columns='dimension', values='value')
        
        desired_order = ['Factuality', 'Robustness', 'Accountability']
        pivot_table_strategy = pivot_table_strategy[[col for col in desired_order if col in pivot_table_strategy.columns]]
        pivot_table_strategy = pivot_table_strategy.sort_index()
        
        sns.heatmap(
            pivot_table_strategy,
            annot=True,
            fmt='.3f',
            cmap='RdYlGn',
            center=0.5,
            vmin=0,
            vmax=1,
            linewidths=2,
            linecolor='white',
            cbar_kws={'label': 'Score'},
            square=True,
            ax=axes[0]
        )
        
        axes[0].set_title('By Strategy\n(Avg across models & parameters)', 
                         fontsize=12, fontweight='bold')
        axes[0].set_xlabel('Dimension', fontsize=11, fontweight='bold')
        axes[0].set_ylabel('Strategy', fontsize=11, fontweight='bold')
        axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=0)
        axes[0].set_yticklabels(axes[0].get_yticklabels(), rotation=0)
        
        # Right: Model comparison
        pivot_model = df_filtered.groupby(['model', 'dimension'])['value'].mean().reset_index()
        pivot_table_model = pivot_model.pivot(index='model', columns='dimension', values='value')
        
        pivot_table_model = pivot_table_model[[col for col in desired_order if col in pivot_table_model.columns]]
        pivot_table_model = pivot_table_model.sort_index()
        
        sns.heatmap(
            pivot_table_model,
            annot=True,
            fmt='.3f',
            cmap='RdYlGn',
            center=0.5,
            vmin=0,
            vmax=1,
            linewidths=2,
            linecolor='white',
            cbar_kws={'label': 'Score'},
            square=True,
            ax=axes[1]
        )
        
        axes[1].set_title('By Model\n(Avg across strategies & parameters)', 
                         fontsize=12, fontweight='bold')
        axes[1].set_xlabel('Dimension', fontsize=11, fontweight='bold')
        axes[1].set_ylabel('Model', fontsize=11, fontweight='bold')
        axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=0)
        axes[1].set_yticklabels(axes[1].get_yticklabels(), rotation=0)
        
        plt.suptitle('Performance Dimension Comparison', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.show()
