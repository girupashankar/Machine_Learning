# Machine_Learning
- Listing all machine learning algorithms is a challenging task due to the vast number of methods and the continuous development in the field. However, here is a comprehensive list of many well-known machine learning algorithms, categorized into different types:

### Supervised Learning Algorithms

#### **Regression**
1. **Linear Regression**
2. **Ridge Regression**
3. **Lasso Regression**
4. **Polynomial Regression**
5. **Elastic Net Regression**
6. **Bayesian Regression**
7. **Support Vector Regression (SVR)**
8. **Decision Tree Regression**
9. **Random Forest Regression**
10. **Gradient Boosting Regression**
11. **AdaBoost Regression**
12. **XGBoost Regression**
13. **LightGBM Regression**
14. **CatBoost Regression**
15. **K-Nearest Neighbors Regression**
16. **Neural Network Regression (e.g., MLPRegressor)**

#### **Classification**
1. **Logistic Regression**
2. **Naive Bayes**
   - Gaussian Naive Bayes
   - Multinomial Naive Bayes
   - Bernoulli Naive Bayes
3. **K-Nearest Neighbors (KNN)**
4. **Support Vector Machine (SVM)**
5. **Decision Tree**
6. **Random Forest**
7. **Gradient Boosting Machines (GBM)**
   - Gradient Boosting Classifier
   - AdaBoost
   - XGBoost
   - LightGBM
   - CatBoost
8. **Neural Networks**
   - Multi-Layer Perceptron (MLP)
   - Convolutional Neural Networks (CNN)
   - Recurrent Neural Networks (RNN)
9. **Linear Discriminant Analysis (LDA)**
10. **Quadratic Discriminant Analysis (QDA)**
11. **Bagging Classifier**
12. **Extra Trees Classifier**
13. **Stacking Classifier**

### Unsupervised Learning Algorithms

#### **Clustering**
1. **K-Means Clustering**
2. **Hierarchical Clustering**
3. **DBSCAN (Density-Based Spatial Clustering of Applications with Noise)**
4. **OPTICS (Ordering Points To Identify the Clustering Structure)**
5. **Mean Shift Clustering**
6. **Gaussian Mixture Models (GMM)**
7. **Agglomerative Clustering**
8. **BIRCH (Balanced Iterative Reducing and Clustering using Hierarchies)**
9. **Spectral Clustering**

#### **Dimensionality Reduction**
1. **Principal Component Analysis (PCA)**
2. **Kernel PCA**
3. **Linear Discriminant Analysis (LDA)**
4. **t-Distributed Stochastic Neighbor Embedding (t-SNE)**
5. **Independent Component Analysis (ICA)**
6. **Factor Analysis**
7. **Non-Negative Matrix Factorization (NMF)**
8. **Latent Dirichlet Allocation (LDA)**
9. **UMAP (Uniform Manifold Approximation and Projection)**

#### **Association Rule Learning**
1. **Apriori Algorithm**
2. **Eclat Algorithm**
3. **FP-Growth Algorithm**

#### **Anomaly Detection**
1. **Isolation Forest**
2. **One-Class SVM**
3. **Local Outlier Factor (LOF)**
4. **Elliptic Envelope**
5. **Autoencoders**

### Semi-Supervised Learning Algorithms
1. **Self-Training**
2. **Co-Training**
3. **Multi-View Learning**
4. **Graph-Based Semi-Supervised Learning**

### Reinforcement Learning Algorithms
1. **Q-Learning**
2. **SARSA (State-Action-Reward-State-Action)**
3. **Deep Q-Network (DQN)**
4. **Double DQN**
5. **Dueling DQN**
6. **Policy Gradient Methods**
   - REINFORCE
   - Actor-Critic
7. **Proximal Policy Optimization (PPO)**
8. **Trust Region Policy Optimization (TRPO)**
9. **A3C (Asynchronous Advantage Actor-Critic)**
10. **A2C (Advantage Actor-Critic)**
11. **Deep Deterministic Policy Gradient (DDPG)**
12. **Soft Actor-Critic (SAC)**
13. **Twin Delayed DDPG (TD3)**

### Ensemble Learning Algorithms
1. **Bagging**
   - Bagged Decision Trees
   - Random Forest
2. **Boosting**
   - AdaBoost
   - Gradient Boosting Machines (GBM)
   - XGBoost
   - LightGBM
   - CatBoost
3. **Stacking**
4. **Blending**
5. **Voting**

### Neural Network Algorithms
1. **Feedforward Neural Networks (FNN)**
2. **Convolutional Neural Networks (CNN)**
3. **Recurrent Neural Networks (RNN)**
   - Long Short-Term Memory Networks (LSTM)
   - Gated Recurrent Unit (GRU)
4. **Autoencoders**
   - Variational Autoencoders (VAE)
   - Denoising Autoencoders
5. **Generative Adversarial Networks (GAN)**
   - DCGAN (Deep Convolutional GAN)
   - WGAN (Wasserstein GAN)
6. **Transformer Models**
   - BERT (Bidirectional Encoder Representations from Transformers)
   - GPT (Generative Pre-trained Transformer)
   - T5 (Text-To-Text Transfer Transformer)

### Probabilistic Graphical Models
1. **Bayesian Networks**
2. **Markov Networks (Markov Random Fields)**
3. **Hidden Markov Models (HMM)**

### Evolutionary Algorithms
1. **Genetic Algorithms**
2. **Genetic Programming**
3. **Evolution Strategies**
4. **Particle Swarm Optimization**
5. **Ant Colony Optimization**

### Other Algorithms
1. **Decision Trees (CART, ID3, C4.5)**
2. **Rule-Based Classifiers (RIPPER)**
3. **K-Nearest Neighbors (KNN)**
4. **Support Vector Machines (SVM)**

This comprehensive list covers many of the machine learning algorithms in use today. The field of machine learning is dynamic, and new algorithms and variations are continually being developed, so staying updated with the latest research is crucial.

# Evaluation Metrics

Evaluation metrics in machine learning are critical tools that help assess the performance of a model. The choice of metric depends on the specific problem type, such as classification, regression, clustering, or ranking. Here’s a comprehensive overview of the most common evaluation metrics used in machine learning:

### Classification Metrics

1. **Accuracy**: The ratio of correctly predicted instances to the total instances.  
   ![image](https://github.com/girupashankar/Machine_Learning/assets/164706869/fd295a38-3d5c-4afb-bc32-55906e7035c3)
   - **TP**: True Positives
   - **TN**: True Negatives
   - **FP**: False Positives
   - **FN**: False Negatives


2. **Precision**: The ratio of true positive predictions to the total predicted positives.  
   ![image](https://github.com/girupashankar/Machine_Learning/assets/164706869/2affc25e-6bbe-4acd-a982-c8f5c7a64a2c)


3. **Recall (Sensitivity or True Positive Rate)**: The ratio of true positive predictions to the total actual positives.  
   ![image](https://github.com/girupashankar/Machine_Learning/assets/164706869/2b220ffa-5844-4395-ab97-b7a64921ff85)


4. **F1 Score**: The harmonic mean of precision and recall.  
   ![image](https://github.com/girupashankar/Machine_Learning/assets/164706869/7255e4b4-5201-4cdd-b74f-aeb2a1db7dd5)


5. **Specificity (True Negative Rate)**: The ratio of true negative predictions to the total actual negatives.  
   ![image](https://github.com/girupashankar/Machine_Learning/assets/164706869/f992f3b3-865f-4838-9dd4-00593459f210)


6. **ROC Curve (Receiver Operating Characteristic Curve)**: A plot of the true positive rate (recall) against the false positive rate (1-specificity).

 
7. **AUC (Area Under the ROC Curve)**: Represents the degree of separability between classes. A higher AUC indicates better performance.


8. **Logarithmic Loss (Log Loss)**: Measures the performance of a classification model where the output is a probability value between 0 and 1.  
   ![image](https://github.com/girupashankar/Machine_Learning/assets/164706869/ce5bde2b-df6d-4cd8-ab41-debe0e173484)


9. **Confusion Matrix**: A table used to describe the performance of a classification model by displaying the true positives, false positives, true negatives, and false negatives.

### Regression Metrics

1. **Mean Absolute Error (MAE)**: The average of the absolute differences between predicted and actual values.  
   ![image](https://github.com/girupashankar/Machine_Learning/assets/164706869/0b30e7b5-7f2b-4fdb-af10-cfc02586ec36)


2. **Mean Squared Error (MSE)**: The average of the squared differences between predicted and actual values.  
   ![image](https://github.com/girupashankar/Machine_Learning/assets/164706869/1217eb07-91ca-4e19-8863-ebe6bf9d1768)


3. **Root Mean Squared Error (RMSE)**: The square root of the mean squared error.  
   ![image](https://github.com/girupashankar/Machine_Learning/assets/164706869/6340f3d0-eb61-412a-9d60-5fd7f137f758)


4. **R-squared (Coefficient of Determination)**: Indicates the proportion of the variance in the dependent variable that is predictable from the independent variables.
   ![image](https://github.com/girupashankar/Machine_Learning/assets/164706869/79fc8513-2922-4b00-88d2-48895ef3fb96)
   where SSres is the sum of squares of residuals and SStot is the total sum of squares.

5. **Mean Absolute Percentage Error (MAPE)**: The average of the absolute percentage errors between predicted and actual values.  
   ![image](https://github.com/girupashankar/Machine_Learning/assets/164706869/2af222ed-d7da-45e2-bebb-48265db56310)

### Clustering Metrics

1. **Silhouette Score**: Measures how similar an object is to its own cluster compared to other clusters. It ranges from -1 to 1.  
   ![image](https://github.com/girupashankar/Machine_Learning/assets/164706869/972d6b62-7d3d-4127-a80b-a765fbc7c374)
   where \(a\) is the mean intra-cluster distance and \(b\) is the mean nearest-cluster distance.

2. **Adjusted Rand Index (ARI)**: Measures the similarity between two data clusterings by considering all pairs of samples and counting pairs that are assigned in the same or different clusters in the predicted and true clusterings.

3. **Davies-Bouldin Index**: Represents the average similarity ratio of each cluster with the cluster that is most similar to it. Lower values indicate better clustering.

4. **Calinski-Harabasz Index**: Also known as the Variance Ratio Criterion, it is the ratio of the sum of between-cluster dispersion and within-cluster dispersion.

### Ranking Metrics

1. **Mean Reciprocal Rank (MRR)**: The average of the reciprocal ranks of results for a sample of queries.  
   ![image](https://github.com/girupashankar/Machine_Learning/assets/164706869/52dffadf-1c28-4285-a9c1-e9e99df47692)

2. **Normalized Discounted Cumulative Gain (NDCG)**: Measures the effectiveness of a ranking algorithm based on the graded relevance of the results.  
   ![image](https://github.com/girupashankar/Machine_Learning/assets/164706869/5385d351-844a-4c26-a443-71d862334776)
   where DCG is the discounted cumulative gain and IDCG is the ideal discounted cumulative gain.

3. **Precision at k (P@k)**: The proportion of relevant items in the top \(k\) retrieved items.  
  ![image](https://github.com/girupashankar/Machine_Learning/assets/164706869/bed565ab-e7ad-44ed-9215-1ac8ff89d1ae)

4. **Mean Average Precision (MAP)**: The mean of average precision scores for a set of queries.  
   ![image](https://github.com/girupashankar/Machine_Learning/assets/164706869/692a2731-4074-483f-8d1c-28a9c3cfe9e0)

### Summary

Choosing the appropriate evaluation metric is essential for correctly interpreting the performance of a machine learning model. It depends on the specific problem being addressed, the data, and the goals of the analysis. Understanding and using the right metrics can significantly impact the success of a machine learning project.

# Hyperparameter Tuninng 

Hyperparameter tuning is a crucial step in the machine learning pipeline, aimed at finding the optimal set of hyperparameters for a model. Here are the main hyperparameter tuning methods used in machine learning:

### 1. Grid Search
- **Definition**: A brute-force approach that involves exhaustively searching through a specified subset of hyperparameters.
- **Procedure**:
  1. Define a set of hyperparameter values.
  2. Train and evaluate the model for every combination of hyperparameters.
  3. Select the combination that results in the best performance based on a chosen evaluation metric.
- **Pros**: Simple to implement and guarantees finding the best combination within the specified grid.
- **Cons**: Computationally expensive, especially with large datasets and many hyperparameters.

### 2. Random Search
- **Definition**: A method that randomly samples hyperparameter values from a predefined distribution.
- **Procedure**:
  1. Define a distribution for each hyperparameter.
  2. Randomly sample combinations of hyperparameters.
  3. Train and evaluate the model for each sampled combination.
  4. Select the combination that results in the best performance.
- **Pros**: More efficient than grid search as it can find good hyperparameters with fewer evaluations.
- **Cons**: May still be computationally expensive and might miss the optimal combination.

### 3. Bayesian Optimization
- **Definition**: An approach that models the performance of the hyperparameters as a probabilistic function and uses this model to select the most promising hyperparameters to evaluate next.
- **Procedure**:
  1. Use a surrogate model (e.g., Gaussian Process) to approximate the objective function.
  2. Optimize the acquisition function to select the next set of hyperparameters to evaluate.
  3. Update the surrogate model with the new results.
  4. Repeat until convergence or a budget is exhausted.
- **Pros**: More sample-efficient than grid or random search and can handle a large number of hyperparameters.
- **Cons**: More complex to implement and computationally intensive.

### 4. Gradient-Based Optimization
- **Definition**: Uses gradients to guide the search for optimal hyperparameters.
- **Procedure**:
  1. Define a differentiable objective function.
  2. Use gradient descent or its variants to update hyperparameters iteratively.
- **Pros**: Efficient for certain types of hyperparameters (e.g., continuous and differentiable ones).
- **Cons**: Not suitable for discrete hyperparameters and may get stuck in local minima.

### 5. Evolutionary Algorithms
- **Definition**: Inspired by the process of natural selection, these algorithms use mechanisms such as mutation, crossover, and selection to evolve a population of hyperparameter sets.
- **Procedure**:
  1. Initialize a population of hyperparameter sets.
  2. Evaluate each set in the population.
  3. Select the best-performing sets to form a new population.
  4. Apply crossover and mutation to create new sets.
  5. Repeat the process for a specified number of generations.
- **Pros**: Good for exploring large, complex search spaces and can handle both continuous and discrete hyperparameters.
- **Cons**: Computationally expensive and requires careful tuning of algorithm parameters (e.g., mutation rate).

### 6. Hyperband
- **Definition**: Combines random search with early stopping to efficiently allocate resources.
- **Procedure**:
  1. Sample a large number of hyperparameter configurations.
  2. Allocate a small budget to each configuration and evaluate.
  3. Retain the top configurations and allocate more budget to them.
  4. Repeat the process, gradually increasing the budget for the best-performing configurations.
- **Pros**: Efficient in terms of both computation and time by focusing resources on promising configurations.
- **Cons**: Relies on the ability to evaluate configurations quickly with a small budget.

### 7. Successive Halving
- **Definition**: A simplified version of Hyperband that progressively increases the budget for a subset of configurations.
- **Procedure**:
  1. Initialize a large set of configurations.
  2. Allocate a small budget to each configuration and evaluate.
  3. Halve the number of configurations, retaining only the top performers.
  4. Increase the budget and evaluate the remaining configurations.
  5. Repeat until a single configuration remains.
- **Pros**: More efficient than traditional grid or random search.
- **Cons**: May discard configurations that perform poorly with a small budget but could excel with more resources.

### 8. Reinforcement Learning (RL) Based Methods
- **Definition**: Uses reinforcement learning to model the hyperparameter tuning process as a sequential decision-making problem.
- **Procedure**:
  1. Define a state space (hyperparameter values), action space (changes to hyperparameters), and a reward function (model performance).
  2. Train an RL agent to select hyperparameter values based on the observed performance.
- **Pros**: Can learn complex patterns in hyperparameter space and adapt over time.
- **Cons**: Complex to implement and requires significant computational resources.

### Summary

Selecting the right hyperparameter tuning method depends on the specific problem, computational resources, and the nature of the hyperparameters. Each method has its strengths and weaknesses, and often a combination of methods might be employed to achieve the best results.
