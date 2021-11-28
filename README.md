# PACVP
PACVP: A Stacking-Based Learning Strategy to Predict Anti-coronavirus peptides Using Effective Feature Representation
<img width="4001" alt="演示文稿1" src="https://user-images.githubusercontent.com/74239672/143756907-e3a76693-7300-4b80-b8dd-54dcf6726eb8.png">

- Requirement

  - `Windows` ：Windows7 or later
  
  Python：
  
  - `Python` >= 3.6
  
- Download `PACOVP`to your computer

  ```bash
  git clone https://github.com/GEHAH/PACVP.git
  ```
  ## How to Use
  
* `process_data.py`: Processing of data sets, including partitioning of training sets and test sets, and average partitioning of positive samples  .  
* `base_classifiers.py`: Construct a baseline model and obtain its probabilistic predictive value.
* `fea_extract.py`: Feature Descriptors.
* `PACOVP.py`: Building mate-model, and the performance of the model's performance. 
