
############################
# HOUSE PREDICTION SOLUTION
############################

# Ames, Lowa’dakikonut evlerinden oluşan bu veri seti içerisinde 79 açıklayıcı değişken bulunduruyor.
# Kaggle üzerinde bir yarışması da bulunan projenin veri seti ve yarışma sayfasına aşağıdaki linkten ulaşabilirsiniz.
# Veri seti bir kaggle yarışmasına ait olduğundan dolayı train ve test olmak üzere iki farklı csv dosyası vardır.
# Test veri setinde ev fiyatları boş bırakılmış olup, bu değerleri bizim  tahmin etmemiz beklenmektedir.
# https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/overview/evaluation

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import datetime as dt
from scipy import stats
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score,GridSearchCV

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter("ignore", category=ConvergenceWarning)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)
warnings.simplefilter(action='ignore', category=Warning)

train = pd.read_csv("datasets_feature/train_house.csv")
test = pd.read_csv("datasets_feature/test_house.csv")
train.head()
test.head()
print(train.shape)
print(test.shape)
df = pd.concat([train, test], axis=0)
df.head()

def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head())
    print("##################### Tail #####################")
    print(dataframe.tail())
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

check_df(df)
def grab_col_names(dataframe, cat_th=10, car_th=20):
        """
        grab_col_names for given dataframe

        :param dataframe:
        :param cat_th:
        :param car_th:
        :return:
        """

        cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]

        num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                       dataframe[col].dtypes != "O"]

        cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                       dataframe[col].dtypes == "O"]

        cat_cols = cat_cols + num_but_cat
        cat_cols = [col for col in cat_cols if col not in cat_but_car]

        num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
        num_cols = [col for col in num_cols if col not in num_but_cat]

        print(f"Observations: {dataframe.shape[0]}")
        print(f"Variables: {dataframe.shape[1]}")
        print(f'cat_cols: {len(cat_cols)}')
        print(f'num_cols: {len(num_cols)}')
        print(f'cat_but_car: {len(cat_but_car)}')
        print(f'num_but_cat: {len(num_but_cat)}')

        # cat_cols + num_cols + cat_but_car = değişken sayısı.
        # num_but_cat cat_cols'un içerisinde zaten.
        # dolayısıyla tüm şu 3 liste ile tüm değişkenler seçilmiş olacaktır: cat_cols + num_cols + cat_but_car
        # num_but_cat sadece raporlama için verilmiştir.

        return cat_cols, cat_but_car, num_cols, num_but_cat


cat_cols, cat_but_car, num_cols, num_but_cat = grab_col_names(df)
num_but_cat

######################################
# 2. Kategorik Değişken Analizi (Analysis of Categorical Variables)
######################################

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))

    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()


for col in cat_cols:
    cat_summary(df, col)

######################################
# 3. Sayısal Değişken Analizi (Analysis of Numerical Variables)
######################################

def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=50)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show()

    print("#####################################")


for col in num_cols:
    num_summary(df, col)

df["MSSubClass"].value_counts()
# We convert to datetime our numeric variable
# df["YearBuilt"] = pd.to_datetime(df["YearBuilt"], format='%Y').dt.year
# df["YearBuilt"] = df["YearBuilt"].astype('int64')
# df["YearRemodAdd"] = pd.to_datetime(df["YearRemodAdd"], format='%Y').dt.year
# df["YearRemodAdd"].head()
### MSSUBCLASS CONVERT TO CAT
num_cols = df.select_dtypes(exclude=['object']).drop(['MSSubClass'], axis=1)
cat_cols = df.select_dtypes(include=['object'])
cat_cols['MSSubClass'] = df['MSSubClass']
# We convert ordinal variable to cat variable because of this variable into the numeric type.
num_cols = df.select_dtypes(exclude=['object']).drop(['OverallQual', 'OverallCond'], axis=1)
cat_cols = df.select_dtypes(include=['object'])
cat_cols['OverallQual'] = df['OverallQual']
cat_cols['OverallCond'] = df['OverallCond']
cat_cols.head()
# 4. Hedef Değişken Analizi (Analysis of Target Variable)
######################################
cat_cols, cat_but_car, num_cols, num_but_cat = grab_col_names(df)

def target_summary_with_cat(dataframe, target, categorical_col):
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean()}), end="\n\n\n")


for col in cat_cols:
    target_summary_with_cat(df,"SalePrice",col)

df.info()

# Bağımlı değişkenin incelenmesi
df["SalePrice"].hist(bins=100)
plt.show(block=True)

# Bağımlı değişkenin logaritmasının incelenmesi
np.log1p(df['SalePrice']).hist(bins=50)
plt.show(block=True)

# 5. VERİ GÖRSELLEŞTİRMELERİNİN YAPILMASI
sns.regplot(x="GrLivArea", y="SalePrice", data=train)
plt.ylim(0,)

sns.regplot(x="GarageArea", y="SalePrice", data=train)
plt.ylim(0,)

sns.regplot(x="MSSubClass", y="SalePrice", data=train)
plt.ylim(0,)

plt.subplots(figsize=(6,4))
sns.distplot(train['SalePrice'], fit=stats.norm)
# NUMERIC VARIABLE PLOT
for col in num_cols:
    fig = plt.figure(figsize=(5, 4))
    ax = fig.gca()
    feature = train[col]
    feature.hist(bins=50, ax = ax)
    ax.axvline(feature.mean(), color='magenta', linestyle='dashed', linewidth=2)
    ax.axvline(feature.median(), color='cyan', linestyle='dashed', linewidth=2)
    ax.set_title(col)
plt.show()

# SALES AND YEARBUILT ANALYSIS
plt.figure(figsize=(20,5))
sns.lineplot(data=df, x="YearBuilt", y="SalePrice")
# SCATTERPLOT
numeric_train = train.select_dtypes(exclude=['object'])

fig = plt.figure(figsize=(30, 180))
for index, col in enumerate(num_cols):
    plt.subplot(20, 3, index + 1)

    sns.scatterplot(x=numeric_train.iloc[:, index], y='SalePrice', data=numeric_train.dropna(), color='navy')
    plt.ylabel('COUNT', size=25, color="black")
    plt.xlabel(col, fontsize=25, color="black")
    plt.xticks(size=20, color="black", rotation=45)
    plt.yticks(size=20, color="black")

fig.tight_layout(pad=1.0)

# CAT COLUMNS VISUALIZATION
sns.color_palette("deep")

fig = plt.figure(figsize=(20, 140))
for index, col in enumerate(cat_cols.columns):
    plt.subplot(26, 3, index + 1)
    sns.countplot(x=cat_cols.iloc[:, index], data=cat_cols.dropna(), palette='Set3')
    plt.ylabel('COUNT', size=18, color="orange")
    plt.xlabel(col, fontsize=18, color="orange")

    plt.xticks(size=15, color="orange", rotation=45)
    plt.yticks(size=15, color="orange")

fig.tight_layout(pad=1.0)

# 6. CORRELATION MATRIX
num_cols = df.select_dtypes(exclude=['object'])
def high_correlated_cols(dataframe, plot=False, corr_th=0.75):
    corr = dataframe.corr()
    cor_matrix = corr.abs()
    upper_triangle_matrix = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(np.bool_))
    drop_list = [col for col in upper_triangle_matrix.columns if any(upper_triangle_matrix[col] > corr_th)]
    if plot:
        sns.set(rc={'figure.figsize': (20, 12)})
        sns.heatmap(corr, linewidth=0.5, cmap="seismic", vmin=-1, vmax=1, fmt='.1f')
        plt.show(block=True)
        return drop_list
        print(drop_list)

drop_list = high_correlated_cols(num_cols, plot=True)

drop_list = [col for col in drop_list if col != 'SalePrice']
df.drop(drop_list, axis=1, inplace=True)
df.shape

# 7. OUTLIER DETECTION
for col in num_cols:
    sns.boxplot(x=col, data=num_cols.dropna(), color='navy')
    plt.ylabel('COUNT', size=25, color="black")
    plt.xlabel(col, fontsize=25, color="black")
    plt.xticks(size=20, color="black", rotation=45)
    plt.yticks(size=20, color="black")
plt.show()

def outlier_thresholds(dataframe, variable, low_quantile=0.10, up_quantile=0.90):
    quantile_one = dataframe[variable].quantile(low_quantile)
    quantile_three = dataframe[variable].quantile(up_quantile)
    interquantile_range = quantile_three - quantile_one
    up_limit = quantile_three + 1.5 * interquantile_range
    low_limit = quantile_one - 1.5 * interquantile_range
    return low_limit, up_limit
# CHECK WHETHER OUTLIER AVAILABLE OR NOT
def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

for col in num_cols:
    if col != "SalePrice" and "Id":
      print(col, check_outlier(df, col))

# Supress the outlier values
def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

for col in num_cols:
    if col != "SalePrice":
        replace_with_thresholds(df,col)

df.describe().T

# 8. MISSING VALUE ANALYSIS
###################

msno.bar(df)
plt.show()

msno.matrix(df)
plt.show()

msno.heatmap(df)
plt.show()
# Creat the na table
def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)

    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)

    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])

    print(missing_df, end="\n")

    if na_name:
        return na_columns

missing_values_table(df)

df["Alley"].value_counts()
df["BsmtQual"].value_counts()
# We should drop the useless datas. We'll drop the NA or useless info rate from the our data.

df.drop(["PoolQC", "MiscFeature", "Alley"], axis=1, inplace=True)
df.head(), df.shape
no_cols = ["BsmtQual","BsmtCond","BsmtExposure","BsmtFinType1","BsmtFinType2","FireplaceQu",
           "GarageType","GarageFinish","GarageQual","GarageCond","Fence"]
# To fill the NO or NA statement to the Nulls
for col in no_cols:
    df[col].fillna("No",inplace=True)
missing_values_table(df)
df["LotFrontage"].value_counts()

# To examine missing values with the target values
def missing_vs_target(dataframe, target, na_columns):
    temp_df = dataframe.copy()

    for col in na_columns:
        temp_df[col + '_NA_FLAG'] = np.where(temp_df[col].isnull(), 1, 0)

    na_flags = temp_df.loc[:, temp_df.columns.str.contains("_NA_")].columns

    for col in na_flags:
        print(pd.DataFrame({"TARGET_MEAN": temp_df.groupby(col)[target].mean(),
                            "Count": temp_df.groupby(col)[target].count()}), end="\n\n\n")


missing_vs_target(df, "SalePrice", no_cols)

# To fill with one function missing value.
def quick_missing_imp(data, num_method="median", cat_length=20, target="SalePrice"):
    variables_with_na = [col for col in data.columns if data[col].isnull().sum() > 0]  # Eksik değere sahip olan değişkenler listelenir

    temp_target = data[target]

    print("# BEFORE")
    print(data[variables_with_na].isnull().sum(), "\n\n")  # Uygulama öncesi değişkenlerin eksik değerlerinin sayısı

    # değişken object ve sınıf sayısı cat_lengthe eşit veya altındaysa boş değerleri mode ile doldur
    data = data.apply(lambda x: x.fillna(x.mode()[0]) if (x.dtype == "O" and len(x.unique()) <= cat_length) else x, axis=0)

    # num_method mean ise tipi object olmayan değişkenlerin boş değerleri ortalama ile dolduruluyor
    if num_method == "mean":
        data = data.apply(lambda x: x.fillna(x.mean()) if x.dtype != "O" else x, axis=0)
    # num_method median ise tipi object olmayan değişkenlerin boş değerleri ortalama ile dolduruluyor
    elif num_method == "median":
        data = data.apply(lambda x: x.fillna(x.median()) if x.dtype != "O" else x, axis=0)

    data[target] = temp_target

    print("# AFTER \n Imputation method is 'MODE' for categorical variables!")
    print(" Imputation method is '" + num_method.upper() + "' for numeric variables! \n")
    print(data[variables_with_na].isnull().sum(), "\n\n")

    return data
df = quick_missing_imp(df, num_method="median", cat_length=17)

# 9. RARE ENCODING PROCESS
cat_cols, cat_but_car, num_cols, num_but_cat = grab_col_names(df)
for col in cat_cols:
    cat_summary(df, col)

def rare_analyser(dataframe, target, cat_cols):
    for col in cat_cols:
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                            "RATIO": dataframe[col].value_counts() / len(dataframe),
                            "TARGET_MEAN": dataframe.groupby(col)[target].mean()}), end="\n\n\n")

rare_analyser(df, "SalePrice", cat_cols)

# We will give a threshold value counts to make rare encoding
def rare_encoder(dataframe, rare_perc):
    temp_df = dataframe.copy()

    rare_columns = [col for col in temp_df.columns if temp_df[col].dtypes == 'O'
                    and (temp_df[col].value_counts() / len(temp_df) < rare_perc).any(axis=None)]

    for var in rare_columns:
        tmp = temp_df[var].value_counts() / len(temp_df)
        rare_labels = tmp[tmp < rare_perc].index
        temp_df[var] = np.where(temp_df[var].isin(rare_labels), 'Rare', temp_df[var])

    return temp_df


df = rare_encoder(df,0.01)
df['Functional'].value_counts()

# 8. FEATURE EXTRACTION
###################
# First of all, we should understand properly, before generate new variable.

"""
MSSubClass: Identifies the type of dwelling involved in the sale.

        20	1-STORY 1946 & NEWER ALL STYLES
        30	1-STORY 1945 & OLDER
        40	1-STORY W/FINISHED ATTIC ALL AGES
        45	1-1/2 STORY - UNFINISHED ALL AGES
        50	1-1/2 STORY FINISHED ALL AGES
        60	2-STORY 1946 & NEWER
        70	2-STORY 1945 & OLDER
        75	2-1/2 STORY ALL AGES
        80	SPLIT OR MULTI-LEVEL
        85	SPLIT FOYER
        90	DUPLEX - ALL STYLES AND AGES
       120	1-STORY PUD (Planned Unit Development) - 1946 & NEWER
       150	1-1/2 STORY PUD - ALL AGES
       160	2-STORY PUD - 1946 & NEWER
       180	PUD - MULTILEVEL - INCL SPLIT LEV/FOYER
       190	2 FAMILY CONVERSION - ALL STYLES AND AGES

MSZoning: Identifies the general zoning classification of the sale.
		
       A	Agriculture
       C	Commercial
       FV	Floating Village Residential
       I	Industrial
       RH	Residential High Density
       RL	Residential Low Density
       RP	Residential Low Density Park 
       RM	Residential Medium Density
	
LotFrontage: Linear feet of street connected to property # (LotArea * LotFrontage)

LotArea: Lot size in square feet # (LotArea * LotFrontage)

Street: Type of road access to property # Drop 

       Grvl	Gravel	
       Pave	Paved
       	
Alley: Type of alley access to property # Drop

       Grvl	Gravel
       Pave	Paved
       NA 	No alley access
		
LotShape: General shape of property

       Reg	Regular	
       IR1	Slightly irregular
       IR2	Moderately Irregular
       IR3	Irregular
       
LandContour: Flatness of the property

       Lvl	Near Flat/Level	
       Bnk	Banked - Quick and significant rise from street grade to building
       HLS	Hillside - Significant slope from side to side
       Low	Depression
		
Utilities: Type of utilities available
		
       AllPub	All public Utilities (E,G,W,& S)	
       NoSewr	Electricity, Gas, and Water (Septic Tank)
       NoSeWa	Electricity and Gas Only
       ELO	Electricity only	
	
LotConfig: Lot configuration

       Inside	Inside lot
       Corner	Corner lot
       CulDSac	Cul-de-sac
       FR2	Frontage on 2 sides of property
       FR3	Frontage on 3 sides of property
	
LandSlope: Slope of property # DROP
		
       Gtl	Gentle slope
       Mod	Moderate Slope	
       Sev	Severe Slope
	
Neighborhood: Physical locations within Ames city limits # DROP

       Blmngtn	Bloomington Heights
       Blueste	Bluestem
       BrDale	Briardale
       BrkSide	Brookside
       ClearCr	Clear Creek
       CollgCr	College Creek
       Crawfor	Crawford
       Edwards	Edwards
       Gilbert	Gilbert
       IDOTRR	Iowa DOT and Rail Road
       MeadowV	Meadow Village
       Mitchel	Mitchell
       Names	North Ames
       NoRidge	Northridge
       NPkVill	Northpark Villa
       NridgHt	Northridge Heights
       NWAmes	Northwest Ames
       OldTown	Old Town
       SWISU	South & West of Iowa State University
       Sawyer	Sawyer
       SawyerW	Sawyer West
       Somerst	Somerset
       StoneBr	Stone Brook
       Timber	Timberland
       Veenker	Veenker
			
Condition1: Proximity to various conditions
	
       Artery	Adjacent to arterial street
       Feedr	Adjacent to feeder street	
       Norm	Normal	
       RRNn	Within 200' of North-South Railroad
       RRAn	Adjacent to North-South Railroad
       PosN	Near positive off-site feature--park, greenbelt, etc.
       PosA	Adjacent to postive off-site feature
       RRNe	Within 200' of East-West Railroad
       RRAe	Adjacent to East-West Railroad
	
Condition2: Proximity to various conditions (if more than one is present)
		
       Artery	Adjacent to arterial street
       Feedr	Adjacent to feeder street	
       Norm	Normal	
       RRNn	Within 200' of North-South Railroad
       RRAn	Adjacent to North-South Railroad
       PosN	Near positive off-site feature--park, greenbelt, etc.
       PosA	Adjacent to postive off-site feature
       RRNe	Within 200' of East-West Railroad
       RRAe	Adjacent to East-West Railroad
	
BldgType: Type of dwelling
		
       1Fam	Single-family Detached	
       2FmCon	Two-family Conversion; originally built as one-family dwelling
       Duplx	Duplex
       TwnhsE	Townhouse End Unit
       TwnhsI	Townhouse Inside Unit
	
HouseStyle: Style of dwelling
	
       1Story	One story
       1.5Fin	One and one-half story: 2nd level finished
       1.5Unf	One and one-half story: 2nd level unfinished
       2Story	Two story
       2.5Fin	Two and one-half story: 2nd level finished
       2.5Unf	Two and one-half story: 2nd level unfinished
       SFoyer	Split Foyer
       SLvl	Split Level
	
OverallQual: Rates the overall material and finish of the house 

       10	Very Excellent
       9	Excellent
       8	Very Good
       7	Good
       6	Above Average
       5	Average
       4	Below Average
       3	Fair
       2	Poor
       1	Very Poor
	
OverallCond: Rates the overall condition of the house

       10	Very Excellent
       9	Excellent
       8	Very Good
       7	Good
       6	Above Average	
       5	Average
       4	Below Average	
       3	Fair
       2	Poor
       1	Very Poor
		
YearBuilt: Original construction date # Age = (Present - YearBuilt)

YearRemodAdd: Remodel date (same as construction date if no remodeling or additions) # (YearRemodAdd - YearBuilt)

RoofStyle: Type of roof

       Flat	Flat
       Gable	Gable
       Gambrel	Gabrel (Barn)
       Hip	Hip
       Mansard	Mansard
       Shed	Shed
		
RoofMatl: Roof material

       ClyTile	Clay or Tile
       CompShg	Standard (Composite) Shingle
       Membran	Membrane
       Metal	Metal
       Roll	Roll
       Tar&Grv	Gravel & Tar
       WdShake	Wood Shakes
       WdShngl	Wood Shingles
		
Exterior1st: Exterior covering on house

       AsbShng	Asbestos Shingles
       AsphShn	Asphalt Shingles
       BrkComm	Brick Common
       BrkFace	Brick Face
       CBlock	Cinder Block
       CemntBd	Cement Board
       HdBoard	Hard Board
       ImStucc	Imitation Stucco
       MetalSd	Metal Siding
       Other	Other
       Plywood	Plywood
       PreCast	PreCast	
       Stone	Stone
       Stucco	Stucco
       VinylSd	Vinyl Siding
       Wd Sdng	Wood Siding
       WdShing	Wood Shingles
	
Exterior2nd: Exterior covering on house (if more than one material)

       AsbShng	Asbestos Shingles
       AsphShn	Asphalt Shingles
       BrkComm	Brick Common
       BrkFace	Brick Face
       CBlock	Cinder Block
       CemntBd	Cement Board
       HdBoard	Hard Board
       ImStucc	Imitation Stucco
       MetalSd	Metal Siding
       Other	Other
       Plywood	Plywood
       PreCast	PreCast
       Stone	Stone
       Stucco	Stucco
       VinylSd	Vinyl Siding
       Wd Sdng	Wood Siding
       WdShing	Wood Shingles
	
MasVnrType: Masonry veneer type

       BrkCmn	Brick Common
       BrkFace	Brick Face
       CBlock	Cinder Block
       None	None
       Stone	Stone
	
MasVnrArea: Masonry veneer area in square feet

ExterQual: Evaluates the quality of the material on the exterior # Encoding (ExterQual + ExterCond)
		
       Ex	Excellent - 5
       Gd	Good - 4
       TA	Average/Typical - 3
       Fa	Fair - 2
       Po	Poor - 1
		
ExterCond: Evaluates the present condition of the material on the exterior # Encoding (ExterQual + ExterCond)
		
       Ex	Excellent - 5
       Gd	Good - 4
       TA	Average/Typical - 3
       Fa	Fair - 2
       Po	Poor - 1
		
Foundation: Type of foundation
		
       BrkTil	Brick & Tile
       CBlock	Cinder Block
       PConc	Poured Contrete	
       Slab	Slab
       Stone	Stone
       Wood	Wood
		
BsmtQual: Evaluates the height of the basement # Encoding (BsmtQual + BsmtCond + BsmtExposure)

       Ex	Excellent (100+ inches)	
       Gd	Good (90-99 inches)
       TA	Typical (80-89 inches)
       Fa	Fair (70-79 inches)
       Po	Poor (<70 inches
       NA	No Basement
		
BsmtCond: Evaluates the general condition of the basement # Encoding (BsmtQual + BsmtCond + BsmtExposure)

       Ex	Excellent
       Gd	Good
       TA	Typical - slight dampness allowed
       Fa	Fair - dampness or some cracking or settling
       Po	Poor - Severe cracking, settling, or wetness
       NA	No Basement
	
BsmtExposure: Refers to walkout or garden level walls # Encoding (BsmtQual + BsmtCond + BsmtExposure)

       Gd	Good Exposure
       Av	Average Exposure (split levels or foyers typically score average or above)	
       Mn	Mimimum Exposure
       No	No Exposure
       NA	No Basement
	
BsmtFinType1: Rating of basement finished area # Encoding - (BsmtFinType1 + BsmtFinType2)

       GLQ	Good Living Quarters
       ALQ	Average Living Quarters
       BLQ	Below Average Living Quarters	
       Rec	Average Rec Room
       LwQ	Low Quality
       Unf	Unfinshed
       NA	No Basement
		
BsmtFinSF1: Type 1 finished square feet # (BsmtFinSF1 + BsmtFinSF2) 

BsmtFinType2: Rating of basement finished area (if multiple types) # Encoding (BsmtFinType1 + BsmtFinType2)

       GLQ	Good Living Quarters
       ALQ	Average Living Quarters
       BLQ	Below Average Living Quarters	
       Rec	Average Rec Room
       LwQ	Low Quality
       Unf	Unfinshed
       NA	No Basement

BsmtFinSF2: Type 2 finished square feet (BsmtFinSF1 + BsmtFinSF2) 

BsmtUnfSF: Unfinished square feet of basement area

TotalBsmtSF: Total square feet of basement area

Heating: Type of heating # Drop
		
       Floor	Floor Furnace
       GasA	Gas forced warm air furnace
       GasW	Gas hot water or steam heat
       Grav	Gravity furnace	
       OthW	Hot water or steam heat other than gas
       Wall	Wall furnace
		
HeatingQC: Heating quality and condition # Encoding

       Ex	Excellent
       Gd	Good
       TA	Average/Typical
       Fa	Fair
       Po	Poor
		
CentralAir: Central air conditioning 

       N	No
       Y	Yes
		
Electrical: Electrical system

       SBrkr	Standard Circuit Breakers & Romex
       FuseA	Fuse Box over 60 AMP and all Romex wiring (Average)	
       FuseF	60 AMP Fuse Box and mostly Romex wiring (Fair)
       FuseP	60 AMP Fuse Box and mostly knob & tube wiring (poor)
       Mix	Mixed
		
1stFlrSF: First Floor square feet
 
2ndFlrSF: Second floor square feet

LowQualFinSF: Low quality finished square feet (all floors)

GrLivArea: Above grade (ground) living area square feet

BsmtFullBath: Basement full bathrooms # TotalBath = (BsmtFullBath + BsmtHalfBath)

BsmtHalfBath: Basement half bathrooms # TotalBath = (BsmtFullBath + BsmtHalfBath)

FullBath: Full bathrooms above grade

HalfBath: Half baths above grade

Bedroom: Bedrooms above grade (does NOT include basement bedrooms)

Kitchen: Kitchens above grade

KitchenQual: Kitchen quality # Encoding

       Ex	Excellent
       Gd	Good
       TA	Typical/Average
       Fa	Fair
       Po	Poor
       	
TotRmsAbvGrd: Total rooms above grade (does not include bathrooms)

Functional: Home functionality (Assume typical unless deductions are warranted)

       Typ	Typical Functionality
       Min1	Minor Deductions 1
       Min2	Minor Deductions 2
       Mod	Moderate Deductions
       Maj1	Major Deductions 1
       Maj2	Major Deductions 2
       Sev	Severely Damaged
       Sal	Salvage only
		
Fireplaces: Number of fireplaces

FireplaceQu: Fireplace quality # Encoding

       Ex	Excellent - Exceptional Masonry Fireplace
       Gd	Good - Masonry Fireplace in main level
       TA	Average - Prefabricated Fireplace in main living area or Masonry Fireplace in basement
       Fa	Fair - Prefabricated Fireplace in basement
       Po	Poor - Ben Franklin Stove
       NA	No Fireplace
		
GarageType: Garage location
		
       2Types	More than one type of garage
       Attchd	Attached to home
       Basment	Basement Garage
       BuiltIn	Built-In (Garage part of house - typically has room above garage)
       CarPort	Car Port
       Detchd	Detached from home
       NA	No Garage
		
GarageYrBlt: Year garage was built
		
GarageFinish: Interior finish of the garage

       Fin	Finished
       RFn	Rough Finished	
       Unf	Unfinished
       NA	No Garage
		
GarageCars: Size of garage in car capacity

GarageArea: Size of garage in square feet

GarageQual: Garage quality # Encoding

       Ex	Excellent
       Gd	Good
       TA	Typical/Average
       Fa	Fair
       Po	Poor
       NA	No Garage
		
GarageCond: Garage condition # Encoding

       Ex	Excellent
       Gd	Good
       TA	Typical/Average
       Fa	Fair
       Po	Poor
       NA	No Garage
		
PavedDrive: Paved driveway

       Y	Paved 
       P	Partial Pavement
       N	Dirt/Gravel
		
WoodDeckSF: Wood deck area in square feet

OpenPorchSF: Open porch area in square feet # TotalProch = (OpenPorchSF + EnclosedPorch)

EnclosedPorch: Enclosed porch area in square feet # TotalProch = (OpenPorchSF + EnclosedPorch + 3SsnPorch + ScreenPorch)

3SsnPorch: Three season porch area in square feet # TotalProch = (OpenPorchSF + EnclosedPorch + 3SsnPorch + ScreenPorch)

ScreenPorch: Screen porch area in square feet # TotalProch = (OpenPorchSF + EnclosedPorch + 3SsnPorch + ScreenPorch)

PoolArea: Pool area in square feet 

PoolQC: Pool quality # Encoding
		
       Ex	Excellent
       Gd	Good
       TA	Average/Typical
       Fa	Fair
       NA	No Pool
		
Fence: Fence quality # Encoding
		
       GdPrv	Good Privacy
       MnPrv	Minimum Privacy
       GdWo	Good Wood
       MnWw	Minimum Wood/Wire
       NA	No Fence
	
MiscFeature: Miscellaneous feature not covered in other categories # Drop
		
       Elev	Elevator
       Gar2	2nd Garage (if not described in garage section)
       Othr	Other
       Shed	Shed (over 100 SF)
       TenC	Tennis Court
       NA	None
		
MiscVal: $Value of miscellaneous feature

MoSold: Month Sold (MM)

YrSold: Year Sold (YYYY)

SaleType: Type of sale
		
       WD 	Warranty Deed - Conventional
       CWD	Warranty Deed - Cash
       VWD	Warranty Deed - VA Loan
       New	Home just constructed and sold
       COD	Court Officer Deed/Estate
       Con	Contract 15% Down payment regular terms
       ConLw	Contract Low Down payment and low interest
       ConLI	Contract Low Interest
       ConLD	Contract Low Down
       Oth	Other
		
SaleCondition: Condition of sale

       Normal	Normal Sale
       Abnorml	Abnormal Sale -  trade, foreclosure, short sale
       AdjLand	Adjoining Land Purchase
       Alloca	Allocation - two linked properties with separate deeds, typically condo with a garage unit	
       Family	Sale between family members
       Partial	Home was not completed when last assessed (associated with New Homes)
"""

##################
# ENCODİNG BY CATEGORIES
##################
qual_col = ["ExterQual", "ExterCond", "BsmtQual", "BsmtCond", "HeatingQC", "KitchenQual", "FireplaceQu", "GarageQual", "GarageCond"]
qual_encoding = {'Ex': 5,'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1, 'NA':0}
for col in qual_col:
    df[col] = df[col].replace(qual_encoding)
df.head()
##################
# BSMT FINTYPE
##################
bsmt_encoding = {'GLQ': 6,'ALQ': 5,'BLQ': 4,'Rec': 3,'LwQ': 2,'Unf': 1, 'NA': 0}
bsmt_col = ['BsmtFinType1','BsmtFinType2']
for col in bsmt_col:
    df[col] = df[col].replace(bsmt_encoding)
    df[col] = pd.to_numeric(df[col], errors='coerce')
df["NEW_Bsmtfintype"] = df["BsmtFinSF1"] + df["BsmtFinSF2"]
df["NEW_Bsmtfintype2"] = df["BsmtFinType1"] + df["BsmtFinType2"]
# ExterQual and ExterCond
exter_col = ["ExterQual", "ExterCond"]
for col in exter_col:
    df[col] = pd.to_numeric(df[col], errors='coerce')
df["NEW_EXTER"] = df["ExterQual"] + df["ExterCond"]
##################
# ENCODING EXPOSURE
##################
expose_encoding = {'Gd': 4, 'Av': 3, 'Mn': 2, 'No': 1, 'NA': 0}
df["BsmtExposure"] = df["BsmtExposure"].replace(expose_encoding)
df.head()
bsmt = ["BsmtQual", "BsmtCond", "BsmtExposure"]
for col in bsmt:
    df[col] = pd.to_numeric(df[col], errors='coerce')
df["NEW_Bsmt"] = df["BsmtQual"] + df["BsmtCond"] + df["BsmtExposure"]
##################
# BUILDING AGE
##################
df["YearBuilt"].max()
today_date = dt.datetime(2010, 1, 1)
today_date = today_date.year
df["BUILD_AGE"] = today_date - df["YearBuilt"]
##################
# RESTORATION DATE
##################
df["NEW_RESTORATION"] = df["YearRemodAdd"] - df["YearBuilt"]

##################
# BSMT BATH
##################
df["BSMT_Bath"] = df["BsmtFullBath"] + df["HalfBath"]

##################
# TOTAL PORCH
##################
df["NEW_TotalPorch"] = df["OpenPorchSF"] + df["EnclosedPorch"] + df["3SsnPorch"] + df["ScreenPorch"]

# Convert to dtype a variable
df['MSSubClass'].dtypes
df['MSSubClass'] = df['MSSubClass'].apply(str)
##################
# DROP PROCESS
##################
drop_list = ["Street", "LandSlope", "Neighborhood", "Heating", "Neighborhood"]
df.drop(drop_list, axis=1, inplace=True)
df.head()
##################
# TOTAL QUAL
##################
df["TotalQual"] = df[["OverallQual", "OverallCond", "ExterQual", "ExterCond", "BsmtCond", "BsmtFinType1",
                      "BsmtFinType2", "HeatingQC", "KitchenQual", "Functional", "FireplaceQu", "GarageQual", "GarageCond", "Fence"]].sum(axis = 1) # 42
##################
# Total Sq Feet
##################
df["NEW_TotalSqFeet"] = df.GrLivArea + df.TotalBsmtSF

##################
# Lot Ratio
##################
df["NEW_LotRatio"] = df.GrLivArea / df.LotArea

##################
# Overall Grade
df["NEW_OverallGrade"] = df["OverallQual"] * df["OverallCond"]
##################
df.columns = [col.upper() for col in df.columns]
df.head(), df.shape

# 9. ENCODING AND SCALING PROCESS
###################
cat_cols, cat_but_car, num_cols, num_but_cat = grab_col_names(df)

def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

binary_cols = [col for col in df.columns if df[col].dtypes == "O" and len(df[col].unique()) == 2]

for col in binary_cols:
    label_encoder(df, col)

# ONE HOT ENCODING

def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

df = one_hot_encoder(df, cat_cols, drop_first=True)
df.head(), df.shape

## SCALING PROCESS
from sklearn.preprocessing import RobustScaler
scaling_cols = [col for col in num_cols if col not in "ID, SALEPRICE"]
scaler = RobustScaler()
df[scaling_cols] = scaler.fit_transform(df[scaling_cols])
df["ENCLOSEDPORCH"] = df["ENCLOSEDPORCH"].apply(lambda x: 1 if x > 0 else 0)
df.head()
df.isnull().sum()
df["NEW_BSMTFINTYPE2"] = df["NEW_BSMTFINTYPE2"].fillna(0)
##################################
# MODELLING
##################################
# Train data has sales price datas. Test data has not.
train_df = df[df['SALEPRICE'].notnull()]
train_df.head()
test_df = df[df['SALEPRICE'].isnull()]
test_df.head()
y = train_df['SALEPRICE'] # np.log1p(df['SalePrice'])
X = train_df.drop(["ID", "SALEPRICE"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1905)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

models = [('LR', LinearRegression()),
          #("Ridge", Ridge()),
          #("Lasso", Lasso()),
          #("ElasticNet", ElasticNet()),
          ('KNN', KNeighborsRegressor()),
          ('CART', DecisionTreeRegressor()),
          ('RF', RandomForestRegressor()),
          #('SVR', SVR()),
          ('GBM', GradientBoostingRegressor()),
          ("XGBoost", XGBRegressor(objective='reg:squarederror')),
          ("LightGBM", LGBMRegressor())]
          # ("CatBoost", CatBoostRegressor(verbose=False))]

for name, regressor in models:
    rmse = np.mean(np.sqrt(-cross_val_score(regressor, X, y, cv=5, scoring="neg_mean_squared_error")))
    print(f"RMSE: {round(rmse, 4)} ({name}) ")

"""
RMSE: 40829226730.7714 (LR) 
RMSE: 46832.4654 (KNN) 
RMSE: 40104.2759 (CART) 
RMSE: 29764.4835 (RF) 
RMSE: 26006.8459 (GBM) 
RMSE: 28641.7347 (XGBoost) 
RMSE: 28080.7182 (LightGBM) 
"""

df['SALEPRICE'].mean()
df['SALEPRICE'].std()

##################
# We should inverse the log convert.
##################
train_df = df[df['SALEPRICE'].notnull()]
test_df = df[df['SALEPRICE'].isnull()]

y = np.log1p(train_df['SALEPRICE'])
X = train_df.drop(["ID", "SALEPRICE"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1905)

lgbm = LGBMRegressor().fit(X_train, y_train)
y_pred = lgbm.predict(X_test)

y_pred

# Yapılan LOG dönüşümünün tersinin (inverse'nin) alınması
new_y = np.expm1(y_pred)
new_y
new_y_test = np.expm1(y_test)
new_y_test

np.sqrt(mean_squared_error(new_y_test, new_y))
#  23065.95990927671

##################
# HYPERPARAMETERS OPTIMIZATION
##################

lgbm_model = LGBMRegressor(random_state=46)

rmse = np.mean(np.sqrt(-cross_val_score(lgbm_model, X, y, cv=5, scoring="neg_mean_squared_error")))


lgbm_params = {"learning_rate": [0.01, 0.1],
               "n_estimators": [500, 1500],
               "colsample_bytree": [0.5, 0.7, 1]}

lgbm_gs_best = GridSearchCV(lgbm_model,
                            lgbm_params,
                            cv=3,
                            n_jobs=-1,
                            verbose=True).fit(X, y)



final_model = lgbm_model.set_params(**lgbm_gs_best.best_params_).fit(X, y)

rmse = np.mean(np.sqrt(-cross_val_score(final_model, X, y, cv=5, scoring="neg_mean_squared_error")))
# 0.12568071872785785
lgbm_gs_best.get_params()
lgbm_gs_best.best_params_
# {'colsample_bytree': 0.5, 'learning_rate': 0.01, 'n_estimators': 1500}

# FEATURE IMPORTANCE
def plot_importance(model, features, num=len(X), save=False):

    feature_imp = pd.DataFrame({"Value": model.feature_importances_, "Feature": features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False)[0:num])
    plt.title("Features")
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig("importances.png")

model = LGBMRegressor()
model.fit(X, y)

plot_importance(model, X_train, num=10)

########################################
# We will predict the test data
########################################

model = LGBMRegressor()
model.fit(X, y)
predictions = model.predict(test_df.drop(["ID","SALEPRICE"], axis=1))

dictionary = {"Id":test_df.index, "SalePrice":predictions}
dfSubmission = pd.DataFrame(dictionary)
dfSubmission.to_csv("housePricePredictions.csv", index=False)
test_df.head()