## Code refactoring


How I want to refactor the code for simplicity:



### Rules

1) All the experiments should be inside the experiments folder
2) All the data should be saved to /home/woody/b114cb/b114cb23/boxo
3) Each new experiments should be in a separate folder (with the same name as the folder containing the final data)
4) The experiment folders should have python code that cannot be reused
5) Specific configuration / bash scripts should be inside the experiments folder
6) Scripts inside src/tools should be compatible with yaml configuration files
7) The output format of generators/oracles/etc should be the same across implementations

