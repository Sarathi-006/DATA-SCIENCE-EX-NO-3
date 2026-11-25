## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.

STEP 2:Clean the Data Set using Data Cleaning Process.

STEP 3:Apply Feature Encoding for the feature in the data set.

STEP 4:Apply Feature Transformation for the feature in the data set.

STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.

2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.

3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.

4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation

• Reciprocal Transformation

• Square Root Transformation

• Square Transformation

  # 2. POWER TRANSFORMATION
• Boxcox method

• Yeojohnson method

# CODING AND OUTPUT:
```
import pandas as pd 
df=pd.read_csv("Encoding Data.csv") 
df
```
<img width="866" height="260" alt="image" src="https://github.com/user-attachments/assets/67450ece-6d5d-4196-8f04-5a6854c101aa" />


```
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
pm=['Hot','Warm','Cold'] 
e1=OrdinalEncoder(categories=[pm]) 
e1.fit_transform(df[["ord_2"]])
```
<img width="865" height="190" alt="image" src="https://github.com/user-attachments/assets/1ced6518-c592-43e6-9c28-7c0acc93a358" />

```
df['bo2']=e1.fit_transform(df[["ord_2"]])
df
```
<img width="862" height="272" alt="image" src="https://github.com/user-attachments/assets/f3f42441-9f9e-431b-9d6d-0eb8c212b2ac" />

```
le=LabelEncoder()
dfc=df.copy()
dfc['ord_2']=le.fit_transform(dfc['ord_2'])
dfc
```
<img width="867" height="270" alt="image" src="https://github.com/user-attachments/assets/2f140045-4d1b-4764-b091-8679e6c71c86" />

```
from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(sparse_output=False)
df2=df.copy()
enc=pd.DataFrame(ohe.fit_transform(df2[["nom_0"]]))
df2=pd.concat([df2,enc],axis=1)
df2
```
<img width="856" height="270" alt="image" src="https://github.com/user-attachments/assets/bc04e4ac-6ec1-4b12-b839-bb82357f8bf3" />

```
pd.get_dummies(df2,columns=["nom_0"])
```
<img width="842" height="271" alt="image" src="https://github.com/user-attachments/assets/7e1b59c4-2ae1-4fad-a8a2-a0b507a2334e" />

```
pip install --upgrade category_encoders
```
<img width="1065" height="217" alt="image" src="https://github.com/user-attachments/assets/66569585-6ca7-477e-8de4-0cb4b7728857" />

```
import pandas as pd 
from category_encoders import BinaryEncoder 
df=pd.read_csv("data.csv") 
df
```
<img width="871" height="282" alt="image" src="https://github.com/user-attachments/assets/bc659273-f5d2-4d94-986d-dc8336c52038" />

```
be=BinaryEncoder() 
nd=be.fit_transform(df['Ord_2']) 
dfb=pd.concat([df,nd],axis=1) 
dfb
```
<img width="886" height="261" alt="image" src="https://github.com/user-attachments/assets/2aac4d1c-7c35-4db9-bb2a-adce52328b88" />

```
from category_encoders import TargetEncoder 
te=TargetEncoder() 
CC=df.copy() 
new=te.fit_transform(X=CC["City"],y=CC["Target"]) 
CC=pd.concat([CC,new],axis=1) 
CC
```
<img width="852" height="267" alt="image" src="https://github.com/user-attachments/assets/badb6385-38a5-41f7-b658-be1654af731e" />

```
import pandas as pd 
from scipy import stats 
import numpy as np 
df=pd.read_csv("Data_to_Transform.csv") 
df
```
<img width="906" height="317" alt="image" src="https://github.com/user-attachments/assets/b9675616-b68a-4360-89f6-a727865ec45f" />

```
df.skew()
```
<img width="651" height="100" alt="image" src="https://github.com/user-attachments/assets/ce081e82-f55f-4c2b-bbb9-57a51f9a250f" />

```
np.log(df["Highly Positive Skew"])
```
<img width="687" height="212" alt="image" src="https://github.com/user-attachments/assets/eced82a8-1e12-4eda-8e83-83d06fb0b148" />

```
np.reciprocal(df["Moderate Positive Skew"])
```
<img width="671" height="212" alt="image" src="https://github.com/user-attachments/assets/5b7bb8a4-2a18-4740-90d4-148349e3c31d" />

```
np.sqrt(df["Highly Positive Skew"])
```
<img width="682" height="210" alt="image" src="https://github.com/user-attachments/assets/42eedbc1-04f7-4cf9-af49-f4642e8a28b8" />

```
np.square(df["Highly Positive Skew"])
```
<img width="682" height="213" alt="image" src="https://github.com/user-attachments/assets/ba5c5989-83c8-4678-b0cf-fe14731f6068" />

```
df["Highly Positive Skew_boxcox"], parameters=stats.boxcox(df["Highly Positive Skew"]) 
df
```
<img width="872" height="313" alt="image" src="https://github.com/user-attachments/assets/bfab9e2e-1066-434e-8bd8-6c2616b1be3d" />

```
df["Highly Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Highly Negative Skew"]) 
df.skew()
```
<img width="620" height="157" alt="image" src="https://github.com/user-attachments/assets/80a4c838-5878-4237-a090-0ed850259a21" />

```
from sklearn.preprocessing import QuantileTransformer 
qt=QuantileTransformer(output_distribution='normal') 
df["Moderate Negative Skew_1"]=qt.fit_transform(df[["Moderate Negative Skew"]]) 
df
```
<img width="1302" height="307" alt="image" src="https://github.com/user-attachments/assets/93de8733-3f84-4929-9834-a54bbfd072b0" />

```
import seaborn as sns 
import statsmodels.api as sm # STATS MODEL- STATISTICAL MODEL TO VISUALIZE DISTRIBUTION 
import matplotlib.pyplot as plt 
sm.qqplot(df["Moderate Negative Skew"],line='45') # QQ - QUANTILE QUANTILE PLOT 
plt.show()
```
<img width="997" height="393" alt="image" src="https://github.com/user-attachments/assets/4e54edb0-ecfd-4617-bd84-e4374af41fc4" />


```
sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45') # RECIPROCAL 
plt.show()
```
<img width="1163" height="387" alt="image" src="https://github.com/user-attachments/assets/5e4eead4-dc46-4b65-9dd4-6e47bf838870" />

```
from sklearn.preprocessing import QuantileTransformer 
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891) 
df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]]) 
sm.qqplot(df["Moderate Negative Skew"],line='45') 
plt.show()
```
<img width="1283" height="380" alt="image" src="https://github.com/user-attachments/assets/4e7e8fcc-23d2-4100-9f3b-df965ec8cc75" />

# RESULT:
       To read the given data and perform Feature Encoding and Transformation process and save the data to a file is successfully completed
# SUMMARY:
The given datasets were successfully read using Pandas, and various Feature Encoding and Feature Transformation techniques were applied. Categorical features were converted into numerical values using Ordinal Encoding, Label Encoding, One-Hot Encoding, Binary Encoding, and Target Encoding. Numerical features were transformed using Log, Reciprocal, Square Root, Square, Box-Cox, Yeo-Johnson, and Quantile Transformation techniques to reduce skewness and improve data distribution. The skewness of the features was analyzed, and Q-Q plots were used to visualize the distribution before and after transformation. Finally, the processed dataset was prepared for further analysis and machine learning applications.
