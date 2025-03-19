






    python activity_prediction.py --iteration_num $i --label $label
    # Fold the sequences with ESM fold
    echo Folding started
    python ESM_Fold.py --iteration_num $i  --label $label
    # Calculate TM Score
    echo foldseek started for 4is2
    export PATH=/home/woody/b114cb/b114cb23/foldseek/bin/:$PATH
    foldseek easy-search output_iteration$((i))/PDB  '4is2.pdb' ${label}_TM_iteration$((i)) tm --format-output "query,target,alntmscore,qtmscore,ttmscore,alnlen" --exhaustive-search 1 -e inf --tmscore-threshold 0.0
   
     # Calculate aligment and clusters
    echo Aligments and cluster 
    export PATH=/home/woody/b114cb/b114cb23/mmseqs/bin/:$PATH
    mmseqs easy-cluster seq_gen_${label}_iteration$((i)).fasta clustering/clustResult_0.9_seq_gen_${label}_iteration$((i)) tmp --min-seq-id 0.9
    mmseqs easy-cluster seq_gen_${label}_iteration$((i)).fasta clustering/clustResult_0.5_seq_gen_${label}_iteration$((i)) tmp --min-seq-id 0.5
    mmseqs easy-search  seq_gen_${label}_iteration$((i)).fasta /home/woody/b114cb/b114cb23/brenda_dataset/database_${target}.fasta alignment/alnResult_seq_gen_${target}_iteration$((i)).m8 tmp    
    #mmseqs easy-search  seq_gen_${label}_iteration$((i)).fasta /home/woody/b114cb/b114cb23/brenda_dataset/database_${label}.fasta alignment/alnResult_seq_gen_${label}_iteration$((i)).m8 tmp    
   
    