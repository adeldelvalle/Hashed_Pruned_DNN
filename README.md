# Hashed Neural Networks for Efficient Weight Pruning and Regularization

Research Idea by Adel del Valle at New York University. 

Research Idea inspired from SLIDE & Mongoose Algorithms. 


## Project Description
This project explores the use of hashing-based algorithms to reduce redundancy in neural network weights. The approach involves creating fingerprints of weights, grouping similar weights into buckets using Locality-Sensitive Hashing (LSH), and aggregating weights for computational efficiency. The project integrates regularization, compression, and pruning into a single adaptive framework.

---

## Milestones

### Milestone 1: Finishing LSH Algorithm ✅ 
#### 1.2: LSH with Maximum Inner Product Search ✅ 
#### 1.3: Representatives ✅ 

### Milestone 2:  Layer dimensionality reshaping
**Completion Status:** ✅ Completed  
- After hashing and grouping weights by similarity, achieved to reshape the current layer output dim and the next layer input dim.

### Milestone 3:  Weight Representatives 
**Completion Status:** ✅ Completed  
- Collected activations, summed the activations and did a weighted average of the grouped weights per layer to get the representative.



### Milestone 4: Synthetic Data Testing  
**Completion Status:** ✅ Completed  
- Tested the prototype on synthetic datasets, achieving competitive accuracy and reduced training times.

### Milestone 5: Initial Prototype  
**Completion Status:** ✅ Completed  
- Developed a basic prototype for weight hashing and pruning.

### Milestone 6: Evaluation on Real-World Datasets  
**Completion Status:** ⏳ In Progress  


### Milestone 7: Implementation of Learnable Hash Functions  
**Completion Status:** ⏳ Pending  
- Planned feature to adaptively learn hash functions for improved performance.

### Milestone 9: Per-Layer rehashing 
**Completion Status:** ⏳ Pending

### Milestone 10: Metric to schedule layer rehash 
**Completion Status:** ⏳ Pending

### Milestone 8: Pass it to ResNet
**Completion Status:** Failed
- Bugs to fix and features to implement before jumping to another architecture. 



---

## Repository and Code Structure

### Directory Structure

```
project
│   README.md
│   cupy_kernel.py
│   hashedFC.py   #  Class for the Initial Hashed Fully Connected Layer. 
│   lsh.py      # LSH construction and call of Cython module
│   main.py   # Experiments being down, build of HashedNN and Vanilla NN 
│   simHash.py    # Kernel for building fingerprints and producing the random projections
│   utils.py     # Utils functions for main
│
└───clsh
│   │   LSH.cpp    # All the dynamics of LSH with multi-threading 
│   │   LSH.h  
│   │   Makefile
│   │   clsh.pyx    # Modularized LSH interaction with Python
│   │   cupy_kernel.py
│   │   matrix_simhash.py
│   │   query_mul_interface.py
│   │   setup.py    # Set-up the Cython module
```

To run the project, first you need to setup the Cython module:

```
python setup.py build_ext --inplace
```

This will produce a .so file. If you have problems importing clsh, export the variable to that .so file.

Then, you can simply run 

```python main.py```

and experiment from there. 

# Prelimiinary Results

| Network  | FLOPs       | Training acc  | Testing acc | Running time | Speedup  | Total parameters |
|------------|------------|------------|------------|------------|------------|------------|
| Hashed NN | 4,435,310  | 69.39%-75.10% |68.25%-70.60%| 14.21 (s)|2.42x|4,439,333 |
| Vanilla NN| 456,030,000| 69.39%  | 68.25% |34.53 (s)|1 |456,075,002|

## Models epoch loss per step
![W&B Chart](assets/W%26B%20Chart%2012_11_2024,%206_27_08%20PM.png)

## Runtime per step 

![W&B Chart](assets/W%26B%20Chart%2012_11_2024,%206_27_01%20PM.png)

We can observe how after the first rehashing, the runtime of the Hashed NN doesn't increase as the Vanilla one. 

## FC1 Weight norm 

![W&B Chart](assets/W%26B%20Chart%2012_11_2024,%206_49_55%20PM-2.png)

As we can see, each hashing brings the Hashed NN weights norm lower, which is more suited for more stable updates and better generalization.

## Final gradient weight norms

![W&B Chart](assets/W%26B%20Chart%2012_11_2024,%206_26_53%20PM.png)


