# Classical ML Experiments Repo
A collection of small classicial ML experiments for educational implementation built from scratch. Most of these projects were built to better understand different fields of machine learning. 

---
## 1. Hopfield Network
### Overview

This project implements Hopfield Network from scratch using Numpy. The model, given the data; 
* mounts the data through weights,
* destorys the image by given corruption rate,
* reconstruct the image based on the Hopfield update rule,
* and shows the initial and final images after updates.



### How to Run
```
pip install -r requirements.txt
git clone https://github.com/markna627/ml_experiments.git
cd ml_experiments/hopfield_network
train_hopfield.py --corruption_rate 0.25 --n_iter 1000

```
Available Arguments:
```
--corruption_rate   (float)  amount of noise to add, e.g. 0.2  
--n_iter            (int)    number of Hopfield updates  
```
Colab demo is available:
[Here](https://colab.research.google.com/drive/1BywK8P9n4dc02KBK_xsI7k1_YT2fkqUC?usp=sharing)


### Example Run:

```
#Corrupted digit 3, with corruption rate = 0.2
    ████        ██            ██
    ██████          ██  ██    ██
        ██████████████  ██  ██  
        ████        ████  ██    
    ██              ████      ██
██      ██  ██    ████        ██
██████        ██  ██            
        ██  ██  ██████          
      ██        ██  ████    ██  
  ██          ██    ████        
        ██          ██      ██  
    ██  ████        ██      ██  
    ██  ████  ██  ████        ██
            ██████████          
██    ██            ██  ████    
            ██      ██  ██  "    


#The reconstructed image after mounting the original image:           
          ████████████         
        ████        ████       
                    ████      
                  ████         
              ██████          
            ██████████        
                ██  ████       
                    ████      
                    ████      
        ████        ████      
          ████████████        
            ████████ 
```

### Notes
* The demo is specifically designed to be run on a digit 3, and to help visualize binary values were mapped to █ and space. 
* The update rule follows the hopfield network's update rule. 

### Related Works
* J. J. Hopfield. “Neural networks and physical systems with emergent collective computational abilities.”
Proceedings of the National Academy of Sciences, 1982. [Link](https://www.pnas.org/doi/10.1073/pnas.79.8.2554)

---
---
## 2. Gaussian Mixture Model
### Overview
The Gaussian Mixture Model was implemented and a synthetic dataset was given to estimate its means and covariances.
* E-step/M-step were implemented from scratch.
* The estimated means and covariances of the three clusters were confirmed to be very accurate.


### How to Run

```
pip install -r requirements.txt
git clone https://github.com/markna627/ml_experiments.git
cd ml_experiments/gmm
training.py
```

Colab demo is available: [Here](https://colab.research.google.com/drive/1RdnTLRYJJY5pN3rsd6lbkywh3aNDXXWK?usp=sharing)




