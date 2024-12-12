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

 ├── main.py # Entry point for training and evaluation 