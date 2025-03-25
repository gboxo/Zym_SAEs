#!/bin/bash
for file in src/training/*.py; do
        aider --model deepseek/deepseek-chat --message "Add pydoc comments" $file
done    
