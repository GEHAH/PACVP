# PACVP
PACVP: A Stacking-Based Learning Strategy to Predict Anti-coronavirus peptides Using Effective Feature Representation
Due to the global outbreak of COVID-19 and its variants, antiviral peptides with anti-coronavirus activity (ACVPs) represent a promising new drug candidate for the treatment of coronavirus infection. At present, several computational tools have been developed to identify ACVPs, but the overall prediction performance is still not enough to meet the actual therapeutic application. In this study, we constructed an efficient and reliable prediction model PACVP (Prediction of Anti-CoronaVirus Peptides) for identifying ACVPs based on effective feature representation and a two-layer stacking learning framework. In the first layer, we use nine feature encoding methods with different feature representation angles to characterize the rich sequence information and fuse them into a feature matrix. Secondly, data normalization and unbalanced data processing are carried out. Next, 12 baseline models are constructed by combining three feature selection methods and four machine learning classification algorithms. In the second layer, we input the optimal probability features into the logistic regression algorithm (LR) to train the final model PACVP. The experiments show that PACVP achieves favorable prediction performance on independent test dataset, with ACC of 0.9208 and AUC of 0.9465. We hope that PACVP will become a useful method for identifying, annotating and characterizing novel ACVPs.
<img width="4001" alt="演示文稿1" src="https://user-images.githubusercontent.com/74239672/143756907-e3a76693-7300-4b80-b8dd-54dcf6726eb8.png">

- Requirement

  - `Windows` ：Windows7 or later
  
  Python：
  
  - `Python` >= 3.6
  
- Download `PACOVP`to your computer

  ```bash
  git clone https://github.com/GEHAH/PACVP.git
  ```

