# CSE 515 Group Project Phase 2

## How To Execute
To run each of the tasks please execute the command "python3 task_x.py" where x is an integer between 1-9. Once this is done each task will prompt you to enter some input through the terminal which will lead to the execute of the specified task. Please follow the input specifications displayed inside of the terminal. 

## What Is Accomplished In This Project
This phase of the project involved the team continuing to utilize the Olivetti Faces dataset of images to extract Color Moments (CM), Extended Binary Local Patterns (ELBP), and Histograms of Oriented Gradients (HOG) feature sets. These are represented as vector models, and then further utilized to perform several tasks related to similarity matrices, dimensional reduction via Principal Component Analysis (PCA), Singular-Valued Decomposition (SVD), Latent Dirichlet Allocation (LDA), and K-Means, graph representation, and graphical node analysis via the ASCOS++ and Personalized PageRank (PPR) measures.


###Task 1 - task_1.py

Run this by command "python3 task_1.py". This task takes following input:
- Feature Model Name -> String {color_moment, histogram_of_oriented_gradients, local_binary_pattern}
- Values of X -> String -> {cc, con, detail, emboss, jitter, neg, noise1, noise2, original, poster, rot, smooth, stipple}
- K-Value -> Integer
- Dimensionality Reduction Model Name -> String {PCA, SVD, LDA, KMeans}


###Task 2 - task_2.py

Run this by command "python3 task_2.py". This task takes following input:
- Feature Model Name -> String {color_moment, histogram_of_oriented_gradients, local_binary_pattern}
- Values of Y -> Integer (1-40)
- K-Value -> Integer
- Dimensionality Reduction Model Name -> String {PCA, SVD, LDA, KMeans}


###Task 3 - task_3.py
Run this by command "python3 task_3.py". This task takes following input:
- Feature Model Name -> String {color_moment, histogram_of_oriented_gradients, local_binary_pattern}
- K-Value -> Integer
- Dimensionality Reduction Model Name -> String {PCA, SVD, LDA, KMeans}



###Task 4 - task_4.py
Run this by command "python3 task_4.py". This task takes following input:
- Feature Model Name -> String {color_moment, histogram_of_oriented_gradients, local_binary_pattern}
- K-Value -> Integer
- Dimensionality Reduction Model Name -> String {PCA, SVD, LDA, KMeans}


###Task 5 - task_5.py
Run this by command "python3 task_5.py". This task takes following input:
- Filepath of Query Image -> String 
- Filepath of Latent Feature File -> String
- N-Value -> Integer


###Task 6 - task_6.py
Run this by command "python3 task_6.py". This task takes following input:
- Filepath of Query Image -> String 
- Filepath of Latent Feature File -> String


###Task 7 - task_7.py
Run this by command "python3 task_7.py". This task takes following input:
- Filepath of Query Image -> String 
- Filepath of Latent Feature File -> String


###Task 8 - task_8.py
Run this by command "python3 task_8.py". This task takes following input:
- N-Value -> Integer
- M-Value -> Integer
- Filename of Similarity Matrix File -> String 


###Task 9 - task_9.py
Run this by command "python3 task_9.py". This task takes following input:
- Feature Model Name -> String {color_moment, histogram_of_oriented_gradients, local_binary_pattern}
- K-Value -> Integer
- Dimensionality Reduction Model Name -> String {PCA, SVD, LDA, KMeans}
- N-Value -> Integer
- M-Value -> Integer
- 1st subject ID -> Integer 
- 2nd subject ID -> Integer 
- 3rd subject ID -> Integer