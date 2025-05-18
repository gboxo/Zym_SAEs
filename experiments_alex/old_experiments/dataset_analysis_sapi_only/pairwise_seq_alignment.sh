#!/bin/bash

set -eo pipefail

output_dir="/home/woody/b114cb/b114cb23/boxo/dataset_analysis_sapi_only/pairwise_alignment_results"
mkdir -p "$output_dir"

# Collect FASTA files (exclude subdirectories)
fasta_files=(/home/woody/b114cb/b114cb23/DPO_clean_amylase_run_SAPI_only_gerard/*.fasta)
if [ ${#fasta_files[@]} -eq 0 ]; then
    echo "No FASTA files found in current directory"
    exit 1
fi

total_pairs=$((${#fasta_files[@]} * (${#fasta_files[@]} - 1)))
current_pair=1

# Process all ordered pairs
for query in "${fasta_files[@]}"; do
    for target in "${fasta_files[@]}"; do
        if [ "$query" == "$target" ]; then
            continue
        fi
        
        # Create unique output name
        clean_query=$(basename "$query" .fasta | tr '.' '_')
        clean_target=$(basename "$target" .fasta | tr '.' '_')
        outfile="${output_dir}/${clean_query}_vs_${clean_target}.m8"
        
        # Skip existing results
        if [ -f "$outfile" ]; then
            echo "[${current_pair}/${total_pairs}] Skipping existing: $outfile"
            ((current_pair++))
            continue
        fi

        echo "[${current_pair}/${total_pairs}] Processing: $query vs $target"
        
        # Create temporary workspace
        tmp_dir=$(mktemp -d)
        
        # Create databases
        mmseqs createdb "$query" "${tmp_dir}/query_db"
        mmseqs createdb "$target" "${tmp_dir}/target_db"
        
        # Run search and convert results
        mmseqs search \
            "${tmp_dir}/query_db" \
            "${tmp_dir}/target_db" \
            "${tmp_dir}/results" \
            "${tmp_dir}/work" \
            --threads 16 
        
        mmseqs convertalis \
            "${tmp_dir}/query_db" \
            "${tmp_dir}/target_db" \
            "${tmp_dir}/results" \
            "$outfile"
        
        # Cleanup
        rm -rf "$tmp_dir"
        ((current_pair++))
    done
done

echo "All pairwise comparisons completed. Results saved to: $output_dir"
