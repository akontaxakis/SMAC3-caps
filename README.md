

# CAPS–SMAC Integration

This repository contains the integration of **CAPS** with **SMAC**. 
This if forked repository from automl/SMAC3

The main CAPS repository, including instructions on how to integrate CAPS with other AutoML tools, can be found here:  
*https://github.com/akontaxakis/CAPS.git*.

---

## Overview

**CAPS** acts as a middleware between the **generation** and **evaluation** stages of an AutoML system.  
Its purpose is to select which pipelines should be evaluated by using a **weighted ratio between expected performance and expected cost**, controlled by a parameter **λ (lambda)**.

### How CAPS Works

1. **Cost & Performance Estimation**  
   CAPS trains models that predict:
   - the expected **execution cost** of a generated pipeline  
   - the expected **performance** of the same pipeline  

2. **Pipeline Selection (NP-hard problem)**  
   Selecting the optimal set of pipelines is NP-hard (see our paper for details).  
   CAPS provides two approximation algorithms:
   - **Greedy**
   - **Beam Search**
   
---

## Testing CAPS–SMAC

An example run using the **Dionis** dataset is included in this repository called ***smac_caps_integration_test.py***.

### Additionally

This repository includes the following examples: 
- SMAC integrated with CAPS ***file: smac_caps_integration_test.py***
- SMAC integration with CAPS and Hyperband ***file: smac_caps_hyperband_integration_test.py***
- SMAC integration with CAPS and metalearning ***file: smac_caps_with_metalearning_integration_test.py***


All datasets used in our experiments can be found here:  
*https://automl.chalearn.org/data*.

---

## CAPS–SMAC Parameterization

Below is a typical configuration snippet for CAPS inside TPOT:

```python
mode = "CAPS"                        # CAPS-specific parameter if set to any other values CAPS will not be used
sel_algo = "caps-greedy"           # options: caps-greedy, caps-beam_search, flaML-like, ratio
lamda = 0.5                      # used with caps-greedy and caps-beam_search
selection = 8                  # CAPS-specific parameter that identifies how many pipelines to be selected
N = 20                      #parameter taht specifies how many pipelines to be generated
data_id = "jannis"           # unique identifier for using a dataset and for history graph + logging files
```

### Contact

For any questions don't hesitate to ask:

Antonios Kontaxakis, antonios.kontaxakis-ATNOSPAM-ulb.be
