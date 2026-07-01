<div style='text-align: center;'><img src='https://maas-watermark-prod-new.cn-wlcb.ufileos.com/ocr%2Fcrop%2F20260630162926cc9d729a28ed4f31%2Fcrop_1_1782808213308.png?UCloudPublicKey=TOKEN_6df395df-5d8c-4f69-90f8-a4fe46088958&Signature=kW8R%2BVE3kb5Ru78yDiQ%2B%2BT0hWic%3D&Expires=1783413013' alt='OCR图片'/></div>

Article

<div align="center">

# Utilizing Supervised and Unsupervised Machine Learning Techniques for Crop Yield Prediction, Pest Detection, And Precision Farming

</div>

Article History:

Name of Author: Dr. Inumarthi V. Srinivas $ ^{1} $ , Megha Dhotay $ ^{2} $ , Dr. J. Sridevi $ ^{3} $ , Ritika Sanwal $ ^{4} $ , Nikita Jain $ ^{5} $

## Affiliation:

1Associate Professor, Prin. L. N. Welingkar Institute of Management Development and Research (PGDM), Lakhamsi Napoo Rd, Opposite Matunga Gymkhana, Matunga East, Mumbai - 400019

2Lecturer, Department of Polytechnic and Skill Development, Dr. Vishwanath Karad MIT World Peace University Pune, Maharashtra, India

3Associate Professor, School of Commerce, Vel Tech Rangarajan Dr. Sagunthala R&D Institute of Science and Technology, Chennai, Tamil Nadu, India - 600062

4Assistant Professor, Department of Media and Mass Communication, Graphic Era Hill University, Haldwani Campus, Uttarakhand

5Assistant Professor, Faculty of Engineering, Teerthanker Mahaveer University, Moradabad, India

Corresponding Author: Dr. Inumarthi V. Srinivas

## How to cite this article:

Dr. Inumarthi V. Srinivas, et, al, Utilizing Supervised and Unsupervised Machine Learning Techniques for Crop Yield Prediction, Pest Detection, And Precision Farming. J Int Commer Law Technol. 2026;7(1):660-669.

Received: 28-01-2026

Revised: 16-02-2026

Accepted: 20-02-2026

Published: 02-03-2026

©2026 the Author(s). This is an open access article distributed under the terms of the Creative Commons Attribution License (http://creativecommons.org/licenses/by/4.0

Abstract: Crop recommendations and agricultural decision-making can be improved with the use of machine learning methods. It is analysed using the 2,200 records on soil nutrients and environmental factors including "temperature, pH, humidity, rainfall phosphorus, nitrogen, as well as potassium" that Kaggle gave. Exploratory data analysis determines trends of crop appropriateness. Crop classification is done by the supervised models Logistic Regression and Random Forest to give an accuracy of 0.9727 and 0.9955, respectively. Clustering unsupervised procedures, such as K-Means and Agglomerative Clustering, show silhouette scores of 0.3229 and 0.3468 between clusters representing environmental groups.

Keywords: Machine Learning, Precision Agriculture, Crop Recommendation, Random Forest, Logistic Regression, K-Means Clustering, Agglomerative Clustering, Agricultural Data Analysis.

## INTRODUCTION

Agriculture is very important in bringing about food security, economic stabilization, and sustainable development.

<div style='text-align: center;'><img src='https://maas-watermark-prod-new.cn-wlcb.ufileos.com/ocr%2Fcrop%2F20260630162926cc9d729a28ed4f31%2Fcrop_2_1782808213404.png?UCloudPublicKey=TOKEN_6df395df-5d8c-4f69-90f8-a4fe46088958&Signature=onRi8s1kaFWuHNFQih07zFhjOtQ%3D&Expires=1783413013' alt='OCR图片'/></div>

The dynamism in the climate conditions, soil fertility, and water availability presents farmers with difficulties in choosing the right crops and enhancing productivity. Logistic Regression and Random Forest are some of the supervised learning models that are used to classify crops based on environmental factors. The methods of unsupervised learning, including K-Means and Agglomerative Clustering, are employed to identify latent groupings of agricultural data. Combining both procedures would offer an in-depth insight into the crop patterns and environmental impact.

## Aim and Objectives

This study will employ machine learning to predict crops and analyze the patterns of the agricultural field by using both supervised and unsupervised machine learning. The aims consist of analyzing agricultural data with the help of the exploratory data analysis, creating models to classify data to predict the right crops, clustering data with the help of the cluster methods, measuring the performance of the models with the help of the relevant metrics, and exploring how these methods can be used in the context of the precision farming practice.

## Literature Review

## Machine Learning in Crop Prediction

Several publications indicate that algorithms of supervised learning are useful in crop prediction. The use of the Logistic Regression is usually applied in matters of classification since it is easy and understandable. It assists in determining the likelihood of a crop being within a certain category depending on the environmental characteristics. Nevertheless, more complicated agronomical data may demand sophisticated models.

<div style='text-align: center;'><img src='https://maas-watermark-prod-new.cn-wlcb.ufileos.com/ocr%2Fcrop%2F20260630162926cc9d729a28ed4f31%2Fcrop_1_1782808213423.png?UCloudPublicKey=TOKEN_6df395df-5d8c-4f69-90f8-a4fe46088958&Signature=qGouD4IqCdbKwvCoXkLNlU3CS70%3D&Expires=1783413013' alt='OCR图片'/></div>

<div align="center">

Fig 1: Process model for a machine learning

</div>

The random forest is also utilized in the field of agriculture research since it addresses non-linear relations and massive data with ease. It minimizes overfitting, as it trains a combination of several decision trees and enhances the prediction (Amini and Rahmani, 2023). All the studies carried out show that Random Forest has been effective in crop recommendation system because it can handle various agricultural factors like the nutrients in soil and rainfall. Classification algorithms such as Support Vector Machines are also used in crop yield prediction. Under these models, there is more precision when it comes to predicting the right crops under varying environmental conditions. Supervised learning approaches offer classical remedies in crop classification and estimation of crop yields.

## Unsupervised Learning in Agricultural Data Analysis

Hidden patterns in agronomical data are learnt with the help of the unsupervised learning methods. These methods do not need labelled data as opposed to the supervised ones. One of the most frequently applicable algorithms to the analysis of agricultural data is K-Means clustering. It clusters alike pieces of data similarity, such as the amount of rainfall or the concentration of nutrients in soil.

<div style='text-align: center;'><img src='https://maas-watermark-prod-new.cn-wlcb.ufileos.com/ocr%2Fcrop%2F20260630162926cc9d729a28ed4f31%2Fcrop_2_1782808213460.png?UCloudPublicKey=TOKEN_6df395df-5d8c-4f69-90f8-a4fe46088958&Signature=YdZ0IhwFTL%2Bj%2FBvIDEr5E0Rbb8E%3D&Expires=1783413013' alt='OCR图片'/></div>

## Fig 2: Machine Learning Model Development Workflow for Crop Prediction

Agglomerative Clustering, as well as Hierarchical clustering, is implemented to discover structured clusters of the farming data (Panigrahi et al. 2023). The approach uses clusters in a sequence and assists in learning about what environmental variables are related to one another. The purpose of clustering is to be able to segment agricultural fields, determine the types of soils, and determine crop patterns.

## Methodology

Dataset Description

<div style='text-align: center;'><img src='https://maas-watermark-prod-new.cn-wlcb.ufileos.com/ocr%2Fcrop%2F20260630162926cc9d729a28ed4f31%2Fcrop_1_1782808213491.png?UCloudPublicKey=TOKEN_6df395df-5d8c-4f69-90f8-a4fe46088958&Signature=2pZB%2BUV7v5HBTT%2BoO8VXuob4YH8%3D&Expires=1783413013' alt='OCR图片'/></div>

<div align="center">

Fig 3: Displaying the information

</div>

The data employed in this research was collected on Kaggle and comprises agricultural data for crop suggestion. It has 2,200 records and 8 columns, which are "nitrogen (N), phosphorus (P), potassium (K), temperature, humidity, pH, rainfall, and crop label". The data has numerical and categorical variables. Checking data reveals that 2,200 non -elements exist in all columns, meaning that there are no missing numbers. This guarantees high-quality analysis and model construction.

Dataset Link: "https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset/data"

Data Preprocessing

<div style='text-align: center;'><img src='https://maas-watermark-prod-new.cn-wlcb.ufileos.com/ocr%2Fcrop%2F20260630162926cc9d729a28ed4f31%2Fcrop_2_1782808213531.png?UCloudPublicKey=TOKEN_6df395df-5d8c-4f69-90f8-a4fe46088958&Signature=6ejfN7VxvQaHFO3klgVouYWc12I%3D&Expires=1783413013' alt='OCR图片'/></div>

<div align="center">

Fig 4: Handling Missing Values

</div>

As depicted in the figure, zero missing values are found in all the variables, such as N, P, K, temperature, humidity, pH, rainfall, and label. This confirms the completeness of databanks. The figure and the data in this paper guarantee the reliability of the data and aid proper training of the model without the necessity of imputation and elimination of the data.

<div style='text-align: center;'><img src='https://maas-watermark-prod-new.cn-wlcb.ufileos.com/ocr%2Fcrop%2F20260630162926cc9d729a28ed4f31%2Fcrop_1_1782808213569.png?UCloudPublicKey=TOKEN_6df395df-5d8c-4f69-90f8-a4fe46088958&Signature=%2FeAABLVL3qEJ%2FU9GZDa13tNQCig%3D&Expires=1783413013' alt='OCR图片'/></div>

<div align="center">

Fig 5: Splitting data into train and test

</div>

The figure shows how the dataset is split into a training and a testing set and on an 80:20 proportion. This guarantees objective model assessment. In this paper, data splitting eliminates overfitting and enables appropriate validation of crop prediction performance.

<div style='text-align: center;'><img src='https://maas-watermark-prod-new.cn-wlcb.ufileos.com/ocr%2Fcrop%2F20260630162926cc9d729a28ed4f31%2Fcrop_2_1782808213588.png?UCloudPublicKey=TOKEN_6df395df-5d8c-4f69-90f8-a4fe46088958&Signature=%2Fv%2BiKQp%2FsH2uibZmMkvVVcpH3vE%3D&Expires=1783413013' alt='OCR图片'/></div>

<div align="center">

Fig 6: Performing Feature scaling

</div>

The figure demonstrates the use of standardization on training and testing data, Standard Scaler. Scaling brings down agricultural variables, rainfall and temperature. Feature scaling gives stability to the model and increases the performance of both supervised and unsupervised learning methods.

Performing EDA

<div style='text-align: center;'><img src='https://maas-watermark-prod-new.cn-wlcb.ufileos.com/ocr%2Fcrop%2F20260630162926cc9d729a28ed4f31%2Fcrop_3_1782808213620.png?UCloudPublicKey=TOKEN_6df395df-5d8c-4f69-90f8-a4fe46088958&Signature=saoRfebseaVjixDBUsTybv4gr1s%3D&Expires=1783413013' alt='OCR图片'/></div>

<div align="center">

Fig 7: Displaying summary statistics of the dataset

</div>

<div style='text-align: center;'><img src='https://maas-watermark-prod-new.cn-wlcb.ufileos.com/ocr%2Fcrop%2F20260630162926cc9d729a28ed4f31%2Fcrop_4_1782808213638.png?UCloudPublicKey=TOKEN_6df395df-5d8c-4f69-90f8-a4fe46088958&Signature=WpqtVf30aKCMxEnKpcIu%2FsOhGsE%3D&Expires=1783413013' alt='OCR图片'/></div>

<div align="center">

Fig 8: Scatter plot of temperature and rainfall by crop type

</div>

The data has 2,200 observations, which include the important agricultural variables, including nitrogen, phosphorus, potassium, rainfall, and temperature (Durai and Shamili, 2022). The levels of nutrients and climatic conditions are widely varied, with the temperature of $ 1 0 \mathrm{~}^0 \mathrm{C} $ to $ 4 5 \mathrm{~}^0 \mathrm{C} $ and the rainfall of 20 mm and 300 mm, respectively to favor accurate crop prediction modelling.

<div style='text-align: center;'><img src='https://maas-watermark-prod-new.cn-wlcb.ufileos.com/ocr%2Fcrop%2F20260630162926cc9d729a28ed4f31%2Fcrop_1_1782808213655.png?UCloudPublicKey=TOKEN_6df395df-5d8c-4f69-90f8-a4fe46088958&Signature=j%2Bx9%2F9aii5bCK4WxIv7oSWG8nsM%3D&Expires=1783413013' alt='OCR图片'/></div>

<div align="center">

Fig 9: Boxplot of rainfall distribution

</div>

<div style='text-align: center;'><img src='https://maas-watermark-prod-new.cn-wlcb.ufileos.com/ocr%2Fcrop%2F20260630162926cc9d729a28ed4f31%2Fcrop_2_1782808213674.png?UCloudPublicKey=TOKEN_6df395df-5d8c-4f69-90f8-a4fe46088958&Signature=cOyajFixolkPLY5olsGwC4%2FW2gs%3D&Expires=1783413013' alt='OCR图片'/></div>

<div align="center">

Fig 10: Line plot of average rainfall requirement per crop

</div>

The rainfall varies between almost 20mm and more than 260 mm across the types of crops. Mothbean and lentil have low median rainfall (between 25 and 50 mm), but rice and coconut have high median values of above 200 mm. The sorted average rainfall plot indicates that rice has the highest value of about 215 mm. Such insights are applied to crop suitability analysis and also reinforce precision farming decisions.

<div style='text-align: center;'><img src='https://maas-watermark-prod-new.cn-wlcb.ufileos.com/ocr%2Fcrop%2F20260630162926cc9d729a28ed4f31%2Fcrop_3_1782808213708.png?UCloudPublicKey=TOKEN_6df395df-5d8c-4f69-90f8-a4fe46088958&Signature=OLTgM7dqecQYuudfbJVnUcl5QnM%3D&Expires=1783413013' alt='OCR图片'/></div>

<div align="center">

Fig 11: Frequency distribution of crop rainfall

</div>

<div style='text-align: center;'><img src='https://maas-watermark-prod-new.cn-wlcb.ufileos.com/ocr%2Fcrop%2F20260630162926cc9d729a28ed4f31%2Fcrop_4_1782808213733.png?UCloudPublicKey=TOKEN_6df395df-5d8c-4f69-90f8-a4fe46088958&Signature=5LfCknEKE2RoA%2FXGNS8aN0nxf24%3D&Expires=1783413013' alt='OCR图片'/></div>

<div align="center">

Fig 12: Correlational analysis

</div>

Frequency of rainfall reveals that most crops need 50 mm to 150 mm of rainfall, with a mean of 103.46 mm and a median of 94.87 mm, thus indicating average water demands. Correlation analysis indicates that phosphorus and potassium have strong positive relations (0.74), yet other variables have a weak correlation. The findings in these domains influence the feature understanding and aid in the sound classification and cluster analysis of crops.

## Supervised Learning Models

Supervised learning methods are used to categorize appropriate crops based on the nutrients of soils and the environmental conditions that are present in the crop recommendation dataset of Kaggle. The dataset consists of

nitrogen, phosphorus, potassium, temperature, humidity, pH, and rainfall as predictive variables, and crop label is the target variable (Getahun et al. 2024). The model estimates class probabilities using the sigmoid function:

$$
\cdot \quad " P (y = 1 | x) = 1 / \left(1 + e ^ {- \left(\beta_ {0} + \beta_ {1} x _ {1} + \dots + \beta_ {n} x _ {n}\right)}\right) " \dots \dots \dots (1)
$$

The model parameters are trained through maximizing the likelihood of the training data. Furthermore, a Random Forest classifier is utilized to enhance the predictive performance with an assembly of decision trees produced with the help of bootstrapped samples. Accuracy, precision, recall, F1-score, and confusion matrices are used to assess model performance.

## 3.4 Unsupervised Learning Models

Discovering hidden patterns in the agricultural conditions without crop labels can be done using unsupervised learning methods. During distance-based clustering, feature variables are standardized so that their contribution is the same. K-Means clustering algorithm separates the observations into K clusters by minimizing the within-cluster sum of squares:

$$
\cdot \quad " J = \Sigma_ {i = 1} ^ {K} \Sigma_ {x} \in C _ {i} \left| \left| x - \mu_ {i} \right| \right| ^ {2} " \dots \dots

where $ C_{i} $ represents cluster members, and $ \mu_{i} $ denotes the cluster centroid.

The number of generated clusters is determined by using the algorithm of Elbow, which yields K=4. The Silhouette Score is used to compute cluster quality. Principal Component Analysis (PCA) creates two principal components of the multidimensional variables of agriculture to assist in visualizing agricultural data, without losing key variance (Shams et al. 2024). A comparison of the outcomes of the clustering results will give a better understanding of the similarities in the environment which determine the suitability of crops in the precision farming systems.

## Results and Analysis

4. 1 Supervised Model Results

<div style='text-align: center;'><img src='https://maas-watermark-prod-new.cn-wlcb.ufileos.com/ocr%2Fcrop%2F20260630162926cc9d729a28ed4f31%2Fcrop_1_1782808213753.png?UCloudPublicKey=TOKEN_6df395df-5d8c-4f69-90f8-a4fe46088958&Signature=tBQ97ff0vQH3GLEEodADDvNU33E%3D&Expires=1783413013' alt='OCR图片'/></div>

<div align="center">

Fig 13: Performing Logistic Regression

</div>

<div style='text-align: center;'><img src='https://maas-watermark-prod-new.cn-wlcb.ufileos.com/ocr%2Fcrop%2F20260630162926cc9d729a28ed4f31%2Fcrop_2_1782808213769.png?UCloudPublicKey=TOKEN_6df395df-5d8c-4f69-90f8-a4fe46088958&Signature=0M0iKUtXOzum7Vv848KIlER03oY%3D&Expires=1783413013' alt='OCR图片'/></div>

<div align="center">

Fig 14: Performing Logistic Regression and Performing Random Forest

</div>

The accuracy of the Logistic Regression of the 22 crop classes is 0.9727 with most of the values of precision and recalls significantly close to 1.00. Random Forest has a higher accuracy of 0.9955 with good and similar precision, recall and F1-score. These findings assess the supervised learning performance and prove that Random Forest is a more stable model to use in the correct crop prediction in precision farming.

<div style='text-align: center;'><img src='https://maas-watermark-prod-new.cn-wlcb.ufileos.com/ocr%2Fcrop%2F20260630162926cc9d729a28ed4f31%2Fcrop_1_1782808213787.png?UCloudPublicKey=TOKEN_6df395df-5d8c-4f69-90f8-a4fe46088958&Signature=%2F9%2BmGJ7Cohn7YRgUUaZkkJ%2BXpNo%3D&Expires=1783413013' alt='OCR图片'/></div>

<div align="center">

Fig 15: Confusion matrix for Logistic regression

</div>

<div style='text-align: center;'><img src='https://maas-watermark-prod-new.cn-wlcb.ufileos.com/ocr%2Fcrop%2F20260630162926cc9d729a28ed4f31%2Fcrop_2_1782808213802.png?UCloudPublicKey=TOKEN_6df395df-5d8c-4f69-90f8-a4fe46088958&Signature=aIFaTanISIndVZKbUtR5oF54Wuw%3D&Expires=1783413013' alt='OCR图片'/></div>

<div align="center">

Fig 16: Confusion matrix for Random Forest

</div>

The strong values of the diagonal of most of the 22 crop classes at the confusion matrices denote correct classifications. The misclassifications of Logistic Regression are few of which whereas the Random Forest has practically perfect diagonal consistency. These findings confirm the reliability of classification and validate the fact that Random Forest is a more reliable model in the crop recommendation process under precision farming applications.

<div style='text-align: center;'><img src='https://maas-watermark-prod-new.cn-wlcb.ufileos.com/ocr%2Fcrop%2F20260630162926cc9d729a28ed4f31%2Fcrop_3_1782808213833.png?UCloudPublicKey=TOKEN_6df395df-5d8c-4f69-90f8-a4fe46088958&Signature=Z1g1AqQIO8e3BfhnkOi9C5qsDYU%3D&Expires=1783413013' alt='OCR图片'/></div>

<div align="center">

Fig 17: Performance comparison of Supervised Models

</div>

Comparative analysis indicates that Logistic Regression has a higher accuracy of 0.9727 and F1-score of 0.9725, whilst the Random Forest has a better result of 0.9955 in terms of accuracy, precision, recall, and F1-score. Random Forest is more appropriate because it is more consistent. This analysis is critical in the selection of the model used and the validation of the most trustworthy supervised model that is used to predict crops precisely in precision farming.

## 4.2 Unsupervised Model Results

<div style='text-align: center;'><img src='https://maas-watermark-prod-new.cn-wlcb.ufileos.com/ocr%2Fcrop%2F20260630162926cc9d729a28ed4f31%2Fcrop_4_1782808213867.png?UCloudPublicKey=TOKEN_6df395df-5d8c-4f69-90f8-a4fe46088958&Signature=KEi6gI1%2Bbml8t2De%2B2S2pJP%2B2gI%3D&Expires=1783413013' alt='OCR图片'/></div>

<div style='text-align: center;'><img src='https://maas-watermark-prod-new.cn-wlcb.ufileos.com/ocr%2Fcrop%2F20260630162926cc9d729a28ed4f31%2Fcrop_5_1782808213884.png?UCloudPublicKey=TOKEN_6df395df-5d8c-4f69-90f8-a4fe46088958&Signature=ScGfIVcXVYjeWyazplLy6Glgfxo%3D&Expires=1783413013' alt='OCR图片'/></div>

<div align="center">

Fig 18: Preparing data and Performing K-means Clustering

</div>

The process of data preparation indicates that the label variables are removed, and StandardScaler is used to normalize the agricultural features, followed by clustering. Proper scaling is necessary to make distance-based algorithms reliable. The application of K-Means with k=4 generates a silhouette score of 0.32296470327629756, meaning moderate cluster segregation. These steps build the analytical basis of the research as an organization of inputs and the confirmation of meaningful discovery patterns to validate the use of pattern discovery in applications of precision farming insights.

<div style='text-align: center;'><img src='https://maas-watermark-prod-new.cn-wlcb.ufileos.com/ocr%2Fcrop%2F20260630162926cc9d729a28ed4f31%2Fcrop_1_1782808213901.png?UCloudPublicKey=TOKEN_6df395df-5d8c-4f69-90f8-a4fe46088958&Signature=nnc%2B8fyROhtgwWswX18hDIjjZm4%3D&Expires=1783413013' alt='OCR图片'/></div>

<div style='text-align: center;'><img src='https://maas-watermark-prod-new.cn-wlcb.ufileos.com/ocr%2Fcrop%2F20260630162926cc9d729a28ed4f31%2Fcrop_2_1782808213918.png?UCloudPublicKey=TOKEN_6df395df-5d8c-4f69-90f8-a4fe46088958&Signature=EKTUFeEbuCs0hwZFJ%2FUAOm%2FRGoI%3D&Expires=1783413013' alt='OCR图片'/></div>

<div align="center">

Fig 19: Performing the Elbow method for finding optimal K and Applying PCA in k-means clustering

</div>

Elbow analysis indicates that inertia sharply decreases between K=1 and 8,000 at K=4 with a smooth increase thereafter, and four clusters is the right option to make. PCA set to n_components = 2 is used to reduce dimensionality to form visual representations of the clusters of agricultural data.

<div style='text-align: center;'><img src='https://maas-watermark-prod-new.cn-wlcb.ufileos.com/ocr%2Fcrop%2F20260630162926cc9d729a28ed4f31%2Fcrop_3_1782808213932.png?UCloudPublicKey=TOKEN_6df395df-5d8c-4f69-90f8-a4fe46088958&Signature=wgOzpPMXTt%2FZL%2F02IhQ5CmpVRu0%3D&Expires=1783413013' alt='OCR图片'/></div>

<div style='text-align: center;'><img src='https://maas-watermark-prod-new.cn-wlcb.ufileos.com/ocr%2Fcrop%2F20260630162926cc9d729a28ed4f31%2Fcrop_4_1782808213950.png?UCloudPublicKey=TOKEN_6df395df-5d8c-4f69-90f8-a4fe46088958&Signature=U0gHbW%2Bw%2Bax1ydo5lpU5AlTX%2Fh8%3D&Expires=1783413013' alt='OCR图片'/></div>

<div align="center">

Fig 20: Visualization of k-means clustering using PCA and Performing Agglomerative clustering

</div>

K-Means classification of data based on visualization by PCA identifies four distinct and distinct clusters along Principal Components 1 and 2, suggesting an orderly pattern of agricultural activities. Agglomerative Clustering, nclusters= 4 and ward linkage gives a silhouette score of 0.34678187148114337, which is marginally better than K-Means (0.3229], implying that clustering has better cohesion and separation.

<div style='text-align: center;'><img src='https://maas-watermark-prod-new.cn-wlcb.ufileos.com/ocr%2Fcrop%2F20260630162926cc9d729a28ed4f31%2Fcrop_5_1782808213966.png?UCloudPublicKey=TOKEN_6df395df-5d8c-4f69-90f8-a4fe46088958&Signature=SNQiuU5dW3bR4CyA6IW12YUoV8M%3D&Expires=1783413014' alt='OCR图片'/></div>

<div style='text-align: center;'><img src='https://maas-watermark-prod-new.cn-wlcb.ufileos.com/ocr%2Fcrop%2F20260630162926cc9d729a28ed4f31%2Fcrop_6_1782808214003.png?UCloudPublicKey=TOKEN_6df395df-5d8c-4f69-90f8-a4fe46088958&Signature=IN%2Btn0sTABeMYTYHmyLsh1ltKkI%3D&Expires=1783413014' alt='OCR图片'/></div>

<div align="center">

Fig 21: Agglomerative clustering using PCA and Implementing centroid

</div>

The agricultural conditions are highly hierarchical, with four distinct groups clearly defined in agglomerative clustering as visualized using Principal Component 1 and Principal Component 2. K-Means centroid clustering to define centers on clusters also translates cluster centers with clear highlights, which explains group compactness and separation. These visualizations are very important in validating pattern discovery and reinforcing unsupervised learning evaluation in the study.

<div style='text-align: center;'><img src='https://maas-watermark-prod-new.cn-wlcb.ufileos.com/ocr%2Fcrop%2F20260630162926cc9d729a28ed4f31%2Fcrop_1_1782808214022.png?UCloudPublicKey=TOKEN_6df395df-5d8c-4f69-90f8-a4fe46088958&Signature=ZpL12SOn0YWB%2B6o6yt%2BW2MzbkZ0%3D&Expires=1783413014' alt='OCR图片'/></div>

<div align="center">

Fig 22: Summary of the results of unsupervised models

</div>

<div align="center">

The K-Means clustering results have revealed 4 clusters with inertia of 8673.9306 and silhouette score of 0.3230, which means a moderate separation. There are 804, 611, 585, and 200 observations in cluster distribution. This is an output that backs up unmonitored validation, plus pattern-based agricultural choice examination in the study.

</div>

Pest Detection Flowchart

<div style='text-align: center;'><img src='https://maas-watermark-prod-new.cn-wlcb.ufileos.com/ocr%2Fcrop%2F20260630162926cc9d729a28ed4f31%2Fcrop_2_1782808214049.png?UCloudPublicKey=TOKEN_6df395df-5d8c-4f69-90f8-a4fe46088958&Signature=pMVX%2BWTWcHoxad6PFKdZOLlpDWI%3D&Expires=1783413014' alt='OCR图片'/></div>

<div align="center">

Fig 23: Flowchart of Pest Detection

</div>

## Model Comparison

Performance comparison assesses the presence of supervised and unsupervised methods of learning used in the crop recommendation dataset provided by Kaggle. The Logistic Regression has an accuracy of 0.9727 and it exhibits good classification among several classes of crops. Random Forest shows increased predictive power in aspect accuracy of 0.9955, which means it can deal with more nonlinear relationships between the variables of

soil and climatic conditions (Sharma et al. 2022). Unsupervised analysis gives K-Means clustering a silhouette score of 0.3229 and Agglomerative Clustering a score of 0.3468, indicating a slightly higher separation of clusters. The overall findings suggest that random Forest and Agglomerative procedures provide the best results when it comes to analyzing the agricultural pattern and recommending crops.

## Discussion

Agricultural features analysis shows that climatic variables and soil nutrients significantly contribute to the fitness of crops. Supervised models indicate great predictive potential, the top predictive performance of 0.9955 objectively obtained with the implementation of the Random Forest model, which is a predictor of complex interactions among features commercially capable. The accuracy of the Logistic Regression at 0.9727 is also reliable. The further clustering analysis shows significant groupings of the environment, with Agglomerative Clustering yielding a silhouette score of 0.3468, a little more than K-Means (0.3229]. The results emphasize the importance of using predictive modelling and pattern discovery data to make better choices in precision farming.

## Conclusion

Machine learning offers a useful method of analyzing agricultural conditions and assisting in crop

Machine learning offers a useful method of analyzing agricultural conditions and assisting in crop recommendations. The crop dataset analysis of the Kaggle reveals that soil nutrients, rainfall, temperature, humidity, and pH are important factors that affect the suitability of crops. Supervised learning models classify the types of crops successfully with the best accuracy of 0.9955 for Random Forest and 0.9727 for Logistic Regression. Unsupervised methods indicate significant environmental groupings in which Agglomerative Clustering has a score of 0.3468, a little greater than K-Means (0.3229). Comprehensively, the combination of predictive modelling and clustering enables the crafting of decisions that rely on data and enhance precision farming.

## References

1. Amini, M. and Rahmani, A., 2023. Agricultural databases evaluation with machine learning procedure. Australian Journal of Engineering and Applied Science, 8(2023), pp.39-50.

2. Panigrahi, B., Kathala, K.C.R. and Sujatha, M., 2023. A machine learning-based comparative approach to predict the crop yield using supervised learning with regression models. Procedia Computer Science, 218, pp.2684-2693.

3. Durai, S.K.S. and Shamili, M.D., 2022. Smart farming uses machine learning and deep learning techniques. Decision Analytics Journal, 3, p.100041.

4. Getahun, S., Kefale, H. and Gelaye, Y., 2024. Application of precision agriculture technologies for sustainable crop production and environmental sustainability: A systematic review. The Scientific World Journal, 2024(1), p.2126734.

5. Shams, M.Y., Gamel, S.A. and Talaat, F.M., 2024. Enhancing crop recommendation systems with explainable artificial intelligence: a study on agricultural decision-making. Neural Computing and Applications, 36(11), pp.5695-5714.

6. Sharma, K., Sharma, C., Sharma, S. and Asenso, E., 2022. Broadening the research pathways in smart agriculture: predictive analysis using semiautomatic information modeling. Journal of Sensors, 2022(1), p.5442865.