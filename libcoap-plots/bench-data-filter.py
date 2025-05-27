#!/usr/bin/env python3
"""
Complete CV-based outlier filtering with proper statistics recalculation.
Removes timeout iterations and recalculates all aggregated statistics properly.
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Optional
from collections import Counter
import os
import glob

class CompleteOutlierFilter:
    """
    Complete outlier filtering that:
    1. Identifies timeout iterations using CV analysis
    2. Removes timeout-affected iterations
    3. Recalculates aggregated statistics using proper variance pooling
    4. Updates all metrics: mean, std, mode, below/above, min/max/range
    """
    
    def __init__(self, 
                 cv_timeout_threshold: float = 3.0,
                 cv_retransmission_threshold: float = 1.5,
                 min_iterations_to_keep: int = 5):
        """Initialize the complete outlier filter."""
        self.cv_timeout_threshold = cv_timeout_threshold
        self.cv_retransmission_threshold = cv_retransmission_threshold
        self.min_iterations = min_iterations_to_keep
        
        # Track filtering statistics
        self.filtering_stats = {
            'files_processed': 0,
            'iterations_analyzed': 0,
            'iterations_filtered': 0,
            'timeout_iterations': 0,
            'retransmission_iterations': 0,
            'normal_iterations': 0
        }
    
    def analyze_iteration_cv(self, mean_row: pd.Series, std_row: pd.Series) -> Dict[str, float]:
        """Calculate coefficient of variation for key metrics."""
        cv_analysis = {}
        
        # Duration CV (primary indicator)
        if 'duration' in mean_row.index and 'duration' in std_row.index:
            mean_duration = pd.to_numeric(mean_row['duration'], errors='coerce')
            std_duration = pd.to_numeric(std_row['duration'], errors='coerce')
            
            if pd.notna(mean_duration) and pd.notna(std_duration) and mean_duration > 0:
                cv_analysis['duration_cv'] = std_duration / mean_duration
            else:
                cv_analysis['duration_cv'] = 0.0
        
        # Energy CV (secondary validation)
        energy_col = 'Energy (Wh)'
        if energy_col in mean_row.index and energy_col in std_row.index:
            mean_energy = pd.to_numeric(mean_row[energy_col], errors='coerce')
            std_energy = pd.to_numeric(std_row[energy_col], errors='coerce')
            
            if pd.notna(mean_energy) and pd.notna(std_energy) and mean_energy > 0:
                cv_analysis['energy_cv'] = std_energy / mean_energy
            else:
                cv_analysis['energy_cv'] = 0.0
        
        return cv_analysis
    
    def classify_iteration(self, cv_analysis: Dict[str, float]) -> str:
        """Classify iteration based on CV patterns."""
        duration_cv = cv_analysis.get('duration_cv', 0.0)
        
        if duration_cv >= self.cv_timeout_threshold:
            return 'timeout'
        elif duration_cv >= self.cv_retransmission_threshold:
            return 'retransmission'
        else:
            return 'normal'
    
    def extract_iteration_data(self, df: pd.DataFrame) -> Tuple[List[Dict], Optional[int]]:
        """
        Extract iteration blocks and find the main aggregated section.
        
        The actual structure is:
        - Multiple iteration blocks: 5 data rows + 1 separator row (semicolons)
        - Main separator: dashes (------------)
        - Aggregated statistics
        
        Returns:
            Tuple of (iteration_list, main_separator_index)
        """
        iterations = []
        
        # Debug: Print structure to understand the pattern
        #print("DEBUG: Analyzing data structure...")
        semicolon_separators = []
        dash_separators = []
        
        for i, row in df.iterrows():
            first_col = str(row.iloc[0]) if not pd.isna(row.iloc[0]) else ""
            
            # Look for semicolon separators (iteration block separators)
            if first_col == '' and len(row.dropna()) == 0:
                # Check if this is a row of semicolons by looking at the pattern
                row_str = ';'.join([str(x) if not pd.isna(x) else '' for x in row])
                if row_str.count(';') >= 8 and row_str.replace(';', '').strip() == '':
                    semicolon_separators.append(i)
                    
            # Look for dash separators (main separator)  
            elif '------------' in first_col or ('---' in first_col and len(first_col) > 5):
                dash_separators.append(i)
        
        #print(f"DEBUG: Found {len(semicolon_separators)} semicolon separators at: {semicolon_separators}")
        #print(f"DEBUG: Found {len(dash_separators)} dash separators at: {dash_separators}")
        
        # The main separator should be the dash separator
        if not dash_separators:
            print("ERROR: No main separator (dashes) found")
            return [], None
            
        main_separator_idx = dash_separators[0]  # Should be only one
        
        # Extract iteration blocks based on semicolon separators
        if not semicolon_separators:
            print("WARNING: No semicolon separators found - treating as single iteration")
            # Single iteration case (everything before main separator)
            iteration_data = df.iloc[:main_separator_idx]
            iterations.append({
                'data': iteration_data,
                'separator_idx': len(iteration_data) - 1,  # Last row before main separator
                'iteration_number': 1,
                'start_idx': 0
            })
        else:
            #print(f"DEBUG: Processing {len(semicolon_separators)} iteration blocks")
            
            # Process each iteration block
            current_start = 0
            for i, sep_idx in enumerate(semicolon_separators):
                # Each iteration block: rows from current_start to sep_idx (inclusive)
                iteration_data = df.iloc[current_start:sep_idx + 1]
                
                # The separator within this block is at the end
                local_sep_idx = len(iteration_data) - 1
                
                #print(f"DEBUG: Iteration {i+1}: rows {current_start}:{sep_idx+1} (separator at local row {local_sep_idx})")
                
                iterations.append({
                    'data': iteration_data,
                    'separator_idx': local_sep_idx,
                    'iteration_number': i + 1,
                    'start_idx': current_start
                })
                
                current_start = sep_idx + 1
                
                # Stop if we've reached the main separator
                if current_start >= main_separator_idx:
                    break
        
        #print(f"DEBUG: Successfully extracted {len(iterations)} iteration blocks")
        #print(f"DEBUG: Main separator at row {main_separator_idx}")
        
        return iterations, main_separator_idx
    
    def recalculate_aggregated_statistics(self, kept_iterations: List[Dict], original_columns: List[str]) -> pd.DataFrame:
        """
        Recalculate aggregated statistics from filtered iterations using proper statistical methods.
        
        Args:
            kept_iterations: List of iteration dictionaries that were kept after filtering
            original_columns: Column names from original dataset
            
        Returns:
            DataFrame with recalculated aggregated statistics in correct format
        """
        if not kept_iterations:
            return pd.DataFrame()
        
        k = len(kept_iterations)  # Number of iterations
        print(f"Recalculating aggregated statistics from {k} filtered iterations")
        
        # Extract statistics from each kept iteration
        all_means = []
        all_stds = []
        all_modes = []
        all_belows = []
        all_aboves = []
        
        for iteration in kept_iterations:
            iter_data = iteration['data']
            iter_num = iteration['iteration_number']
            
            # Handle header for first iteration
            if iter_num == 1:
                # Skip header row
                mean_row = iter_data.iloc[1]
                std_row = iter_data.iloc[2]
                mode_row = iter_data.iloc[3]
                below_row = iter_data.iloc[4]
                above_row = iter_data.iloc[5]
            else:
                # No header
                mean_row = iter_data.iloc[0]
                std_row = iter_data.iloc[1]
                mode_row = iter_data.iloc[2]
                below_row = iter_data.iloc[3]
                above_row = iter_data.iloc[4]
            
            all_means.append(mean_row)
            all_stds.append(std_row)
            all_modes.append(mode_row)
            all_belows.append(below_row)
            all_aboves.append(above_row)
        
        # Convert to DataFrames for easier calculation
        means_df = pd.DataFrame(all_means)
        stds_df = pd.DataFrame(all_stds)
        modes_df = pd.DataFrame(all_modes)
        belows_df = pd.DataFrame(all_belows)
        aboves_df = pd.DataFrame(all_aboves)
        
        print(f"  Processing {k} iterations for aggregated statistics")
        
        # Create aggregated statistics rows
        aggregated_rows = []
        
        # 1. DASH SEPARATOR ROW
        separator_data = {}
        for col in original_columns:
            separator_data[col] = '------------'
        aggregated_rows.append(separator_data)
        
        # 2. AGGREGATED MEAN
        mean_data = {}
        for col in original_columns:
            if col in means_df.columns:
                col_means = pd.to_numeric(means_df[col], errors='coerce').dropna()
                if len(col_means) > 0:
                    mean_data[col] = np.mean(col_means)
                else:
                    mean_data[col] = 0
            else:
                mean_data[col] = 0
        aggregated_rows.append(mean_data)
        
        # 3. AGGREGATED STANDARD DEVIATION
        # Use the same formula from bench-data-manager.py adapted for filtered data
        n = 25  # samples per iteration (constant)
        # k = number of kept iterations (variable based on filtering)
        N = n * k  # total samples after filtering
        
        std_data = {}
        for col in original_columns:
            if col in means_df.columns and col in stds_df.columns:
                col_means = pd.to_numeric(means_df[col], errors='coerce').dropna()
                col_stds = pd.to_numeric(stds_df[col], errors='coerce').dropna()
                
                if len(col_means) > 0:
                    if col == 'cpu_cycles':
                        # CPU cycles: single measurement per iteration, no within-iteration std
                        # Aggregated std = std of the iteration means (between-iteration variance)
                        if len(col_means) > 1:
                            std_data[col] = np.std(col_means, ddof=1)
                        else:
                            std_data[col] = 0
                    else:
                        # For other metrics, use correct pooled variance formula
                        if N > 1:
                            if n > 1:
                                # within_variance = average of variances from kept iterations
                                within_variance = np.mean(col_stds ** 2)
                                # between_variance = variance of means from kept iterations
                                if len(col_means) > 1:
                                    between_variance = np.var(col_means, ddof=1)
                                else:
                                    between_variance = 0
                                
                                # Correct pooled variance formula: ((n-1) * within_var + n * between_var) / (N-1)
                                total_variance = ((n-1) * within_variance + n * between_variance) / (N-1)
                                std_data[col] = np.sqrt(max(0, total_variance))
                            else:
                                # n = 1 case, use variance of means
                                if len(col_means) > 1:
                                    std_data[col] = np.std(col_means, ddof=1)
                                else:
                                    std_data[col] = 0
                        else:
                            # N = 1 case
                            std_data[col] = np.mean(col_stds)
                else:
                    std_data[col] = 0
            else:
                std_data[col] = 0
        aggregated_rows.append(std_data)
        
        # 4. AGGREGATED MODE
        mode_data = {}
        for col in original_columns:
            if col in modes_df.columns:
                col_modes = pd.to_numeric(modes_df[col], errors='coerce').dropna()
                if len(col_modes) > 0:
                    mode_counts = Counter(col_modes)
                    mode_data[col] = mode_counts.most_common(1)[0][0]
                else:
                    mode_data[col] = '--'
            else:
                mode_data[col] = '--'
        aggregated_rows.append(mode_data)
        
        # 5. AGGREGATED BELOW MODE
        below_data = {}
        for col in original_columns:
            if col in belows_df.columns:
                col_belows = pd.to_numeric(belows_df[col], errors='coerce').dropna()
                if len(col_belows) > 0:
                    below_data[col] = round(np.mean(col_belows))
                else:
                    below_data[col] = 0
            else:
                below_data[col] = 0
        aggregated_rows.append(below_data)
        
        # 6. AGGREGATED ABOVE MODE
        above_data = {}
        for col in original_columns:
            if col in aboves_df.columns:
                col_aboves = pd.to_numeric(aboves_df[col], errors='coerce').dropna()
                if len(col_aboves) > 0:
                    above_data[col] = round(np.mean(col_aboves))
                else:
                    above_data[col] = 0
            else:
                above_data[col] = 0
        aggregated_rows.append(above_data)
        
        # 7. EQUALS SEPARATOR (for min/max section)
        equals_data = {}
        for col in original_columns:
            equals_data[col] = '======'
        aggregated_rows.append(equals_data)
        
        # 8. MIN VALUES (for discrete metrics only)
        min_data = {}
        discrete_metrics = ['frames_sent', 'bytes_sent', 'frames_received', 
                          'bytes_received', 'total_frames', 'total_bytes']
        for col in original_columns:
            if col in discrete_metrics and col in modes_df.columns:
                col_modes = pd.to_numeric(modes_df[col], errors='coerce').dropna()
                if len(col_modes) > 0:
                    min_data[col] = np.min(col_modes)
                else:
                    min_data[col] = '--'
            else:
                min_data[col] = '--'
        aggregated_rows.append(min_data)
        
        # 9. MAX VALUES (for discrete metrics only)
        max_data = {}
        for col in original_columns:
            if col in discrete_metrics and col in modes_df.columns:
                col_modes = pd.to_numeric(modes_df[col], errors='coerce').dropna()
                if len(col_modes) > 0:
                    max_data[col] = np.max(col_modes)
                else:
                    max_data[col] = '--'
            else:
                max_data[col] = '--'
        aggregated_rows.append(max_data)
        
        # 10. RANGE VALUES (for discrete metrics only)
        range_data = {}
        for col in original_columns:
            if col in discrete_metrics and col in modes_df.columns:
                col_modes = pd.to_numeric(modes_df[col], errors='coerce').dropna()
                if len(col_modes) > 0:
                    range_data[col] = np.max(col_modes) - np.min(col_modes)
                else:
                    range_data[col] = '--'
            else:
                range_data[col] = '--'
        aggregated_rows.append(range_data)
        
        # Create DataFrame from the data dictionaries
        result_df = pd.DataFrame(aggregated_rows)
        
        # Ensure columns are in the correct order
        result_df = result_df[original_columns]
        
        print(f"  Generated aggregated statistics with {len(result_df)} rows")
        return result_df
    
    def filter_aggregated_csv(self, csv_file_path: str, output_path: str = None) -> bool:
        """
        Complete filtering with proper statistics recalculation.
        
        Args:
            csv_file_path: Input CSV file path
            output_path: Output CSV file path
            
        Returns:
            Success boolean
        """
        try:
            # Read original data
            df = pd.read_csv(csv_file_path, sep=';')
            original_columns = df.columns.tolist()
            
            print(f"\nProcessing {os.path.basename(csv_file_path)}")
            print(f"Original shape: {df.shape}")
            
            # Extract iteration structure
            iterations, main_separator_idx = self.extract_iteration_data(df)
            
            if not iterations:
                print("No iteration structure found - copying original file")
                if output_path:
                    df.to_csv(output_path, sep=';', index=False)
                return True
            
            print(f"Found {len(iterations)} iteration blocks")
            
            # Analyze and classify each iteration
            iteration_classifications = []
            
            for iteration in iterations:
                iter_data = iteration['data']
                sep_idx = iteration['separator_idx']
                iter_num = iteration['iteration_number']
                
                if len(iter_data) >= 3:
                    # Handle header row for first iteration only
                    if iter_num == 1:
                        # First iteration has header row
                        # Row 0: Header  
                        # Row 1: Mean values
                        # Row 2: Std values
                        # Row 3: Mode values
                        # Row 4: Above mode counts
                        # Row 5: Below mode counts
                        # Row 6: Separator (on its own row)
                        mean_row = iter_data.iloc[1]  # Mean values (skip header)
                        std_row = iter_data.iloc[2]   # Std values
                    else:
                        # Subsequent iterations don't have header
                        # Row 0: Mean values
                        # Row 1: Std values  
                        # Row 2: Mode values
                        # Row 3: Above mode counts
                        # Row 4: Below mode counts
                        # Row 5: Separator (on its own row)
                        mean_row = iter_data.iloc[0]  # Mean values
                        std_row = iter_data.iloc[1]   # Std values
                    
                    cv_analysis = self.analyze_iteration_cv(mean_row, std_row)
                    classification = self.classify_iteration(cv_analysis)
                    
                    # Extract raw values for detailed debugging
                    duration_mean = pd.to_numeric(mean_row.get('duration', 0), errors='coerce')
                    duration_std = pd.to_numeric(std_row.get('duration', 0), errors='coerce')
                    energy_mean = pd.to_numeric(mean_row.get('Energy (Wh)', 0), errors='coerce')
                    energy_std = pd.to_numeric(std_row.get('Energy (Wh)', 0), errors='coerce')
                    
                    #print(f"  DEBUG: Iteration {iter_num} raw values:")
                    #print(f"    Mean row duration: {duration_mean}")
                    #print(f"    Std row duration: {duration_std}")
                    #print(f"    Mean row energy: {energy_mean}")
                    #print(f"    Std row energy: {energy_std}")
                    
                    iteration_classifications.append({
                        'iteration': iter_num,
                        'classification': classification,
                        'cv_analysis': cv_analysis,
                        'data': iteration
                    })
                    
                    # Print detailed analysis
                    duration_cv = cv_analysis.get('duration_cv', 0)
                    energy_cv = cv_analysis.get('energy_cv', 0)
                    
                    #print(f"  Iteration {iter_num}: {classification.upper()}")
                    #print(f"    Duration: {duration_mean:.3f}s (mean), {duration_std:.3f}s (std)")
                    #print(f"    Duration CV: {duration_cv:.3f}")
                    #if energy_cv > 0:
                        #print(f"    Energy: {energy_mean:.6f} Wh (mean)")
                        #print(f"    Energy CV: {energy_cv:.3f}")
                                                                            
                    # Show why it was classified this way
                    #if duration_cv >= self.cv_timeout_threshold:
                    #    print(f"    -> TIMEOUT (CV {duration_cv:.3f} >= {self.cv_timeout_threshold})")
                    #elif duration_cv >= self.cv_retransmission_threshold:
                    #    print(f"    -> RETRANSMISSION (CV {duration_cv:.3f} >= {self.cv_retransmission_threshold})")
                    #else:
                    #    print(f"    -> NORMAL (CV {duration_cv:.3f} < {self.cv_retransmission_threshold})")
            
            # Determine which iterations to keep
            timeout_iterations = [item for item in iteration_classifications if item['classification'] == 'timeout']
            kept_iterations = [item for item in iteration_classifications if item['classification'] != 'timeout']
            
            # Safety check
            if len(kept_iterations) < self.min_iterations and len(iteration_classifications) >= self.min_iterations:
                print(f"Warning: Would keep only {len(kept_iterations)} iterations. Keeping best {self.min_iterations}.")
                # Sort by duration CV and keep the best ones
                sorted_iterations = sorted(iteration_classifications, 
                                         key=lambda x: x['cv_analysis'].get('duration_cv', 0))
                kept_iterations = sorted_iterations[:self.min_iterations]
                timeout_iterations = sorted_iterations[self.min_iterations:]
            
            # Summary
            classifications = [item['classification'] for item in iteration_classifications]
            normal_count = classifications.count('normal')
            retransmission_count = classifications.count('retransmission')
            timeout_count = len(timeout_iterations)
            
            print(f"\nClassification Summary:")
            print(f"  Normal: {normal_count}")
            print(f"  Retransmission: {retransmission_count}")
            print(f"  Timeout (filtered): {timeout_count}")
            print(f"  Kept: {len(kept_iterations)}")
            
            # Build filtered dataset
            if timeout_count == 0:
                print("No timeout iterations detected - keeping original data")
                filtered_df = df
            else:
                print(f"Filtering {timeout_count} timeout iterations and recalculating statistics...")
                
                # Combine kept iteration data
                filtered_sections = []
                
                # Add kept iterations
                for item in kept_iterations:
                    filtered_sections.append(item['data']['data'])
                
                # Recalculate aggregated statistics
                aggregated_stats = self.recalculate_aggregated_statistics(
                    [item['data'] for item in kept_iterations], 
                    original_columns
                )
                
                if not aggregated_stats.empty:
                    filtered_sections.append(aggregated_stats)
                
                # Combine all sections
                if filtered_sections:
                    filtered_df = pd.concat(filtered_sections, ignore_index=True)
                else:
                    filtered_df = df
            
            # Update statistics
            self.filtering_stats['files_processed'] += 1
            self.filtering_stats['iterations_analyzed'] += len(iterations)
            self.filtering_stats['iterations_filtered'] += timeout_count
            self.filtering_stats['timeout_iterations'] += timeout_count
            self.filtering_stats['retransmission_iterations'] += retransmission_count
            self.filtering_stats['normal_iterations'] += normal_count
            
            # Save result
            if output_path is None:
                base_name = csv_file_path.rsplit('.', 1)[0]
                output_path = f"{base_name}_filtered.csv"
            
            filtered_df.to_csv(output_path, sep=';', index=False)
            print(f"Saved complete filtered data to {os.path.basename(output_path)}")
            
            return True
            
        except Exception as e:
            print(f"Error processing {csv_file_path}: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def print_summary_statistics(self):
        """Print comprehensive filtering summary."""
        print(f"\n{'='*60}")
        print("COMPLETE FILTERING SUMMARY")
        print(f"{'='*60}")
        print(f"Files processed: {self.filtering_stats['files_processed']}")
        print(f"Iterations analyzed: {self.filtering_stats['iterations_analyzed']}")
        print(f"  Normal: {self.filtering_stats['normal_iterations']}")
        print(f"  Retransmission: {self.filtering_stats['retransmission_iterations']}")
        print(f"  Timeout: {self.filtering_stats['timeout_iterations']}")
        print(f"Iterations filtered: {self.filtering_stats['iterations_filtered']}")
        
        if self.filtering_stats['iterations_analyzed'] > 0:
            filter_rate = (self.filtering_stats['iterations_filtered'] / 
                          self.filtering_stats['iterations_analyzed']) * 100
            print(f"Filtering rate: {filter_rate:.1f}%")
        
        print(f"\nRecalculated aggregated statistics using:")
        print(f"  - Proper variance pooling across iterations")
        print(f"  - Within-iteration + between-iteration variance")
        print(f"  - Updated mode, min/max, range for discrete metrics")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Complete CV-based filtering with statistics recalculation")
    parser.add_argument("input", help="CSV file or directory")
    parser.add_argument("--cv-threshold", type=float, default=3.0,
                      help="CV threshold for timeout detection")
    parser.add_argument("--file-pattern", default="*.csv", help="File pattern for directory")
    
    args = parser.parse_args()
    
    outlier_filter = CompleteOutlierFilter(cv_timeout_threshold=args.cv_threshold)
    
    if os.path.isfile(args.input):
        # Single file
        success = outlier_filter.filter_aggregated_csv(args.input)
        outlier_filter.print_summary_statistics()
    elif os.path.isdir(args.input):
        # Directory processing
        pattern = os.path.join(args.input, args.file_pattern)
        csv_files = glob.glob(pattern)
        
        if not csv_files:
            print(f"No CSV files found matching: {pattern}")
        else:
            print(f"Processing {len(csv_files)} files...")
            for csv_file in sorted(csv_files):
                outlier_filter.filter_aggregated_csv(csv_file)
            
            outlier_filter.print_summary_statistics()
    else:
        print(f"Error: {args.input} not found")