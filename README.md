# PSO (Particle Swarm Optimization)
This repository is PSO program on python. This program can search cec2013 benchmarks.

## Setup
### Environments
- Cython: 3.0.10
- matplotlib: 3.8.4
- numpy: 1.26.4
- cec2013single: 0.1
### pip
```
pip install Cython
```
### bash
```
cd PSO
git clone https://github.com/dmolina/cec2013single.git
cd cec2013single
python3 setup.py install --user
cd ..
cp -r cec2013single/cec2013single/cec2013_data .
```

## Run
```
python3 main.py -f (any_number)
```

## Reference
- cec2013single: https://github.com/dmolina/cec2013single
- Problem Definitions and Evaluation Criteria for the CEC 2013 Special Session on Real-Parameter Optimization: https://www.al-roomi.org/multimedia/CEC_Database/CEC2013/RealParameterOptimization/CEC2013_RealParameterOptimization_TechnicalReport.pdf
