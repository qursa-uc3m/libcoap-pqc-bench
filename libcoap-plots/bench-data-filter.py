#!/usr/bin/env python3
"""
Complete CV-based outlier filtering with proper statistics recalculation.
Removes timeout iterations and recalculates all aggregated statistics properly.
FIXED: Proper iteration extraction and bounds checking.
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
                cv_analysis['energy_mwh'] = mean_energy * 1000  # Convert to mWh
            else:
                cv_analysis['energy_cv'] = 0.0
                cv_analysis['energy_mwh'] = 0.0
        
        return cv_analysis
    
    def classify_iteration(self, cv_analysis: Dict[str, float]) -> str:
        """Classify iteration based on CV patterns and absolute bounds."""
        duration_cv = cv_analysis.get('duration_cv', 0.0)
        energy_mwh = cv_analysis.get('energy_mwh', 0.0)
        
        # ABSOLUTE BOUNDS CHECK - CRITICAL FIX
        if energy_mwh > 15.0:  # Energy above 15 mWh is clearly an outlier
            return 'timeout'
        
        # ORIGINAL CV LOGIC
        if duration_cv >= self.cv_timeout_threshold:
            return 'timeout'
        elif duration_cv >= self.cv_retransmission_threshold:
            return 'retransmission'
        else:
            return 'normal'

    def extract_iteration_data(self, df: pd.DataFrame) -> Tuple[List[Dict], Optional[int]]:
        """
        FIXED: Extract iteration blocks with proper header handling.
        """
        iterations = []
        
        # Find semicolon separators (iteration block separators)
        semicolon_separators = []
        dash_separators = []
        
        for i, row in df.iterrows():
            # Convert entire row to string for analysis
            row_str = ';'.join([str(x) if not pd.isna(x) else '' for x in row])
            
            # Look for semicolon separators (at least 8 consecutive semicolons)
            if row_str.count(';') >= 8 and len(row_str.replace(';', '').strip()) == 0:
                semicolon_separators.append(i)
                
            # Look for dash separators (main separator)  
            first_col = str(row.iloc[0]) if not pd.isna(row.iloc[0]) else ""
            if '---' in first_col and len(first_col) > 10:  # More strict dash detection
                dash_separators.append(i)
        
        print(f"DEBUG: Found {len(semicolon_separators)} semicolon separators at: {semicolon_separators}")
        print(f"DEBUG: Found {len(dash_separators)} dash separators at: {dash_separators}")
        
        # The main separator should be the dash separator
        if not dash_separators:
            print("ERROR: No main separator (dashes) found")
            return [], None
            
        main_separator_idx = dash_separators[0]
        
        # Extract iteration blocks based on semicolon separators
        if not semicolon_separators:
            print("WARNING: No semicolon separators found - treating as single iteration")
            iteration_data = df.iloc[:main_separator_idx]
            iterations.append({
                'data': iteration_data,
                'separator_idx': len(iteration_data) - 1,
                'iteration_number': 1,
                'start_idx': 0,
                'has_header': False  # First iteration always has header
            })
        else:
            # FIXED: Process each iteration block correctly with header detection
            current_start = 0
            for i, sep_idx in enumerate(semicolon_separators):
                # Stop if we've reached the main separator
                if sep_idx >= main_separator_idx:
                    break
                    
                # Each iteration block: rows from current_start to sep_idx (exclusive)
                iteration_data = df.iloc[current_start:sep_idx]
                
                # First iteration has header, others don't
                #has_header = (i == 0)
                has_header = False
                
                # FIXED: More flexible validation - just need mean, std, mode rows
                min_rows_needed = 6 if has_header else 5  # header + 5 data rows OR just 5 data rows
                
                if len(iteration_data) >= min_rows_needed:
                    iterations.append({
                        'data': iteration_data,
                        'separator_idx': len(iteration_data) - 1,
                        'iteration_number': i + 1,
                        'start_idx': current_start,
                        'has_header': has_header
                    })
                    print(f"DEBUG: Added iteration {i+1}: rows {current_start} to {sep_idx-1} ({len(iteration_data)} rows, header={has_header})")
                else:
                    print(f"WARNING: Iteration {i+1} has only {len(iteration_data)} rows (need {min_rows_needed}), skipping")
                
                current_start = sep_idx + 1
        
        print(f"DEBUG: Successfully extracted {len(iterations)} valid iteration blocks")
        return iterations, main_separator_idx
    
    def recalculate_aggregated_statistics(self, kept_iterations: List[Dict], original_columns: List[str]) -> pd.DataFrame:
        """
        FIXED: Recalculate aggregated statistics with proper header handling.
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
            has_header = iteration.get('has_header', iter_num == 1)
            
            print(f"DEBUG: Processing iteration {iter_num}, has_header={has_header}, rows={len(iter_data)}")
            
            # FIXED: Proper row indexing based on header presence
            if has_header:
                # First iteration with header: skip header row
                if len(iter_data) < 6:  # Need header + 5 data rows
                    print(f"Warning: Header iteration {iter_num} has only {len(iter_data)} rows, expected at least 6")
                    continue
                    
                mean_row = iter_data.iloc[1]   # Row after header
                std_row = iter_data.iloc[2]
                mode_row = iter_data.iloc[3]
                below_row = iter_data.iloc[4]
                above_row = iter_data.iloc[5]
            else:
                # Subsequent iterations without header
                if len(iter_data) < 5:  # Need 5 data rows
                    print(f"Warning: Non-header iteration {iter_num} has only {len(iter_data)} rows, expected at least 5")
                    continue
                    
                mean_row = iter_data.iloc[0]
                std_row = iter_data.iloc[1]
                mode_row = iter_data.iloc[2]
                below_row = iter_data.iloc[3]
                above_row = iter_data.iloc[4]
            
            # Add to lists
            all_means.append(mean_row)
            all_stds.append(std_row)
            all_modes.append(mode_row)
            all_belows.append(below_row)
            all_aboves.append(above_row)
            
            print(f"DEBUG: Successfully extracted statistics from iteration {iter_num}")
            
        # If no valid iterations after bounds checking, return empty
        if not all_means:
            print("Error: No valid iterations found after bounds checking")
            return pd.DataFrame()
        
        # Convert to DataFrames for easier calculation
        means_df = pd.DataFrame(all_means)
        stds_df = pd.DataFrame(all_stds)
        modes_df = pd.DataFrame(all_modes)
        belows_df = pd.DataFrame(all_belows)
        aboves_df = pd.DataFrame(all_aboves)
        
        # Update k to reflect actual number of valid iterations
        k = len(all_means)
        print(f"  Processing {k} valid iterations for aggregated statistics")
        
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
        n = 25  # samples per iteration (constant)
        N = n * k  # total samples after filtering
        
        std_data = {}
        for col in original_columns:
            if col in means_df.columns and col in stds_df.columns:
                col_means = pd.to_numeric(means_df[col], errors='coerce').dropna()
                col_stds = pd.to_numeric(stds_df[col], errors='coerce').dropna()
                
                if len(col_means) > 0:
                    if col == 'cpu_cycles':
                        # CPU cycles: single measurement per iteration
                        if len(col_means) > 1:
                            std_data[col] = np.std(col_means, ddof=1)
                        else:
                            std_data[col] = 0
                    else:
                        # For other metrics, use correct pooled variance formula
                        if N > 1:
                            if n > 1:
                                within_variance = np.mean(col_stds ** 2)
                                if len(col_means) > 1:
                                    between_variance = np.var(col_means, ddof=1)
                                else:
                                    between_variance = 0
                                
                                total_variance = ((n-1) * within_variance + n * between_variance) / (N-1)
                                std_data[col] = np.sqrt(max(0, total_variance))
                            else:
                                if len(col_means) > 1:
                                    std_data[col] = np.std(col_means, ddof=1)
                                else:
                                    std_data[col] = 0
                        else:
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
        
        # 8-10. MIN/MAX/RANGE VALUES (for discrete metrics only)
        discrete_metrics = ['frames_sent', 'bytes_sent', 'frames_received', 
                          'bytes_received', 'total_frames', 'total_bytes']
        
        for stat_name in ['min', 'max', 'range']:
            stat_data = {}
            for col in original_columns:
                if col in discrete_metrics and col in modes_df.columns:
                    col_modes = pd.to_numeric(modes_df[col], errors='coerce').dropna()
                    if len(col_modes) > 0:
                        if stat_name == 'min':
                            stat_data[col] = np.min(col_modes)
                        elif stat_name == 'max':
                            stat_data[col] = np.max(col_modes)
                        else:  # range
                            stat_data[col] = np.max(col_modes) - np.min(col_modes)
                    else:
                        stat_data[col] = '--'
                else:
                    stat_data[col] = '--'
            aggregated_rows.append(stat_data)
        
        # Create DataFrame from the data dictionaries
        result_df = pd.DataFrame(aggregated_rows)
        
        # Ensure columns are in the correct order
        result_df = result_df[original_columns]
        
        print(f"  Generated aggregated statistics with {len(result_df)} rows")
        return result_df
    
    def filter_aggregated_csv(self, csv_file_path: str, output_path: str = None) -> bool:
        """
        FIXED: Complete filtering with proper statistics recalculation.
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
                iter_num = iteration['iteration_number']
                has_header = iteration.get('has_header', iter_num == 1)
                
                # FIXED: Proper row access based on header presence
                try:
                    if has_header:
                        if len(iter_data) < 3:  # Need at least header + mean + std
                            print(f"Warning: Iteration {iter_num} too short for analysis")
                            continue
                        mean_row = iter_data.iloc[1]  # Skip header
                        std_row = iter_data.iloc[2]
                    else:
                        if len(iter_data) < 2:  # Need at least mean + std
                            print(f"Warning: Iteration {iter_num} too short for analysis")
                            continue
                        mean_row = iter_data.iloc[0]
                        std_row = iter_data.iloc[1]
                    
                    cv_analysis = self.analyze_iteration_cv(mean_row, std_row)
                    classification = self.classify_iteration(cv_analysis)
                    
                    iteration_classifications.append({
                        'iteration': iter_num,
                        'classification': classification,
                        'cv_analysis': cv_analysis,
                        'data': iteration
                    })
                except Exception as e:
                    print(f"Error analyzing iteration {iter_num}: {e}")
                    continue
            
            # Add the detailed iteration analysis output
            print(f"\nDETAILED ITERATION ANALYSIS:")
            for item in iteration_classifications:
                iter_num = item['iteration']
                classification = item['classification']
                cv_analysis = item['cv_analysis']
                
                duration_cv = cv_analysis.get('duration_cv', 0)
                energy_mwh = cv_analysis.get('energy_mwh', 0)
                
                status = "KEEP" if classification != 'timeout' else "REMOVE"
                print(f"  Iteration {iter_num}: {classification.upper()} ({status})")
                print(f"    Duration CV: {duration_cv:.3f}")
                print(f"    Energy: {energy_mwh:.1f} mWh")
                
                if classification == 'timeout':
                    if energy_mwh > 15.0:
                        print(f"    -> TIMEOUT due to energy bounds ({energy_mwh:.1f} > 15.0 mWh)")
                    elif duration_cv >= self.cv_timeout_threshold:
                        print(f"    -> TIMEOUT due to duration CV ({duration_cv:.3f} >= {self.cv_timeout_threshold})")
                print()
            
            # Determine which iterations to keep
            timeout_iterations = [item for item in iteration_classifications if item['classification'] == 'timeout']
            kept_iterations = [item for item in iteration_classifications if item['classification'] != 'timeout']
            
            # Safety check
            if len(kept_iterations) < self.min_iterations and len(iteration_classifications) >= self.min_iterations:
                print(f"Warning: Would keep only {len(kept_iterations)} iterations. Keeping best {self.min_iterations}.")
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
                filtered_df = self._build_filtered_dataset(kept_iterations, original_columns)
            
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

    def _build_filtered_dataset(self, kept_iterations: List[Dict], original_columns: List[str]) -> pd.DataFrame:
        """
        FIXED: Build filtered dataset by combining kept iterations and recalculated stats.
        """
        try:
            result_sections = []
            
            max_iter = max(item['iteration'] for item in kept_iterations)
            # Add all kept iterations
            for item in kept_iterations:
                iter_data = item['data']['data']
                result_sections.append(iter_data)
                print(f"DEBUG: Added kept iteration {item['iteration']} ({len(iter_data)} rows)")
                
                if item['iteration'] < max_iter:
                    sep_row = pd.DataFrame([{col: np.nan for col in original_columns}])
                    result_sections.append(sep_row)
                    print(f"DEBUG: Inserted semicolon separator after iteration {item['iteration']}")
            
            # Add recalculated aggregated statistics
            aggregated_stats = self.recalculate_aggregated_statistics(
                [item['data'] for item in kept_iterations], 
                original_columns
            )
            
            if not aggregated_stats.empty:
                result_sections.append(aggregated_stats)
                print(f"DEBUG: Added recalculated aggregated statistics ({len(aggregated_stats)} rows)")
            else:
                print("ERROR: Failed to recalculate aggregated statistics")
                return pd.DataFrame()
            
            # Combine all sections
            if result_sections:
                final_df = pd.concat(result_sections, ignore_index=True)
                print(f"DEBUG: Final DataFrame has {len(final_df)} rows")
                
                # VALIDATION: Check filtering results
                energy_col = 'Energy (Wh)'
                if energy_col in final_df.columns:
                    energy_values = pd.to_numeric(final_df[energy_col], errors='coerce').dropna()
                    if len(energy_values) > 0:
                        max_energy_mwh = energy_values.max() * 1000
                        outlier_count = (energy_values * 1000 > 15.0).sum()
                        
                        print(f"DEBUG: Maximum energy in filtered file: {max_energy_mwh:.1f} mWh")
                        print(f"DEBUG: Energy values > 15.0 mWh: {outlier_count}")
                        
                        if outlier_count > 0:
                            print(f"WARNING: {outlier_count} energy outliers still remain")
                        else:
                            print("SUCCESS: All energy outliers filtered out")
                
                return final_df
            else:
                print("ERROR: No sections to combine")
                return pd.DataFrame()
                
        except Exception as e:
            print(f"ERROR in _build_filtered_dataset: {e}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()
    
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