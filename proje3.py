import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report
from imblearn.over_sampling import ADASYN
from sklearn.model_selection import cross_validate
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv("creditcard.csv")

print(df.head(10))
print(df.describe())

print('No Frauds', round(df['Class'].value_counts()[0]/len(df) * 100,2), '% of the dataset')
print('Frauds', round(df['Class'].value_counts()[1]/len(df) * 100,2), '% of the dataset')

sns.countplot('Class', data=df)
plt.title('Başlangıçta, 0: No Fraud || 1: Fraud', fontsize=14)
plt.show()
#Fraud ve no fraud verilerimizin görsel görünümü


X = df.drop('Class', axis=1)
y = df['Class']

log_reg = LogisticRegression()
X_eğitim, X_test, y_eğitim, y_test = train_test_split(X, y, test_size=0.20, random_state=113)
log_reg.fit(X_eğitim, y_eğitim)
egitim_dogruluk = log_reg.score(X_eğitim, y_eğitim)
test_dogruluk = log_reg.score(X_test, y_test)
print('One-vs-rest', '-'*20,
      'Modelin eğitim verisindeki doğruluğu : {:.2f}'.format(egitim_dogruluk),
      'Modelin test verisindeki doğruluğu   : {:.2f}'.format(test_dogruluk), sep='\n')
#Modelimiz şu an imbalanced olduğu için doğruluk değerleri 1 gibi gözükse de aslında yanlış çalışıyor.

C_değerleri = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
dogruluk_df = pd.DataFrame(columns=['C_Değeri', 'Doğruluk'])

dogruluk_değerleri = pd.DataFrame(columns=['C Değeri', 'Eğitim Doğruluğu', 'Test Doğruluğu'])

for c in C_değerleri:
    lr = LogisticRegression(penalty='l2', C=c, random_state=0)
    lr.fit(X_eğitim, y_eğitim)
    dogruluk_değerleri = dogruluk_değerleri.append({'C Değeri': c,
                                                    'Eğitim Doğruluğu': lr.score(X_eğitim, y_eğitim),
                                                    'Test Doğruluğu': lr.score(X_test, y_test)
                                                    }, ignore_index=True)
print(dogruluk_değerleri)
#C değerini değiştirmemiz şu an hiçbir anlam ifade etmiyor.



log_reg_mnm = LogisticRegression(multi_class='multinomial', solver='lbfgs')
log_reg_mnm.fit(X_eğitim, y_eğitim)
egitim_dogruluk = log_reg_mnm.score(X_eğitim, y_eğitim)
test_dogruluk = log_reg_mnm.score(X_test, y_test)
print('Multinomial (Softmax)', '-'*20,
      'Modelin eğitim verisindeki doğruluğu : {:.2f}'.format(egitim_dogruluk),
      'Modelin test verisindeki doğruluğu   : {:.2f}'.format(test_dogruluk), sep='\n')
#multinominal yönteminde de aynı şekilde çıkıyor.



#Biraz düzenleme yapabiliriz.
#----------------------------------------------------------------------------------------------------------------
#Modelimizi ilk önce UNDER-SAMPLING yaparak bir daha gözden geçirelim.
df = df.sample(frac=1)

# amount of fraud classes 492 rows.
fraud_df = df.loc[df['Class'] == 1]
non_fraud_df = df.loc[df['Class'] == 0][:492]

normal_distributed_df = pd.concat([fraud_df, non_fraud_df])

new_df = normal_distributed_df.sample(frac=1, random_state=42)

print('Under sampling yaptıktan sonra fraud ve no-fraud verilerin dağılımı:')
print(new_df['Class'].value_counts()/len(new_df))

sns.countplot('Class', data=new_df)
plt.title('Under-samplingten sonra veriler', fontsize=14)
plt.show()
#yeni_df'te 492 fraud 492 no fraud veri var.

X = new_df.drop('Class', axis=1)
y = new_df['Class']

X_eğitim, X_test, y_eğitim, y_test = train_test_split(X, y, test_size=0.20, random_state=113)
log_reg.fit(X_eğitim, y_eğitim)
egitim_dogruluk = log_reg.score(X_eğitim, y_eğitim)
test_dogruluk = log_reg.score(X_test, y_test)

tahmin_eğitim = log_reg.predict(X_eğitim)
tahmin_test = log_reg.predict(X_test)

print('\nUnder-sample yaptığımızda: ', '-'*20,
      'Modelin eğitim verisindeki doğruluğu : {:.2f}'.format(egitim_dogruluk),
      'Modelin test verisindeki doğruluğu   : {:.2f}'.format(test_dogruluk), sep='\n')


C_değerleri = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
dogruluk_df = pd.DataFrame(columns=['C_Değeri', 'Doğruluk'])
dogruluk_değerleri = pd.DataFrame(columns=['C Değeri', 'Eğitim Doğruluğu', 'Test Doğruluğu'])
for c in C_değerleri:
    # Apply logistic regression model to training data
    lr = LogisticRegression(penalty='l2', C=c, random_state=0)
    lr.fit(X_eğitim, y_eğitim)
    dogruluk_değerleri = dogruluk_değerleri.append({'C Değeri': c,
                                                    'Eğitim Doğruluğu': lr.score(X_eğitim, y_eğitim),
                                                    'Test Doğruluğu': lr.score(X_test, y_test)
                                                    }, ignore_index=True)
print(dogruluk_değerleri)

print("under-sample f1_score() değeri        : {:.2f}".format(f1_score(y_test, tahmin_test)))
print("under-sample recall_score() değeri    : {:.2f}".format(recall_score(y_test, tahmin_test)))
print("under-sample precision_score() değeri : {:.2f}".format(precision_score(y_test, tahmin_test)))

#Under sampling yaparak gerçekçi veriler elde ettik.
#Yüksek c değerlerinde daha başarılı değerler elde edebiliyoruz.
#Performans metriklerimiz iyi durumdalar.

#Şimdi bir de roc curve'e bakalım.
tahmin_test_ihtimal = log_reg.predict_proba(X_test)[:,1]

fpr, tpr, thresholds  = roc_curve(y_test, tahmin_test_ihtimal)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(' Under Samplingten sonra ROC Curve')
plt.show()
print('Under sampling AUC Değeri : ', roc_auc_score(y_test, tahmin_test_ihtimal))

#----------------------------------------------------------------------------------------------------------------

# Over-Sampling yaparak modelimizi bir daha gözden geçirelim.

no_fraud = df[df.Class == 0]
fraud = df[df.Class == 1]

sahte_alısveris_artırılmış = resample(fraud,
                                     replace = True,
                                     n_samples = len(no_fraud),
                                     random_state = 111)

artırılmıs_df = pd.concat([no_fraud, sahte_alısveris_artırılmış])
print("\n Over sampling veri sayıları: (1 fraud, 0 no-fraud)")
print(artırılmıs_df.Class.value_counts())

X = artırılmıs_df.drop('Class', axis=1)
y = artırılmıs_df['Class']

X_eğitim, X_test, y_eğitim, y_test = train_test_split(X, y, test_size=0.20, random_state=113)
log_reg.fit(X_eğitim, y_eğitim)
egitim_dogruluk = log_reg.score(X_eğitim, y_eğitim)
test_dogruluk = log_reg.score(X_test, y_test)

tahmin_eğitim = log_reg.predict(X_eğitim)
tahmin_test = log_reg.predict(X_test)

print('\nOver-sample yaptığımızda: ', '-'*20,
      'Modelin eğitim verisindeki doğruluğu : {:.2f}'.format(egitim_dogruluk),
      'Modelin test verisindeki doğruluğu   : {:.2f}'.format(test_dogruluk), sep='\n')


C_değerleri = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
dogruluk_df = pd.DataFrame(columns=['C_Değeri', 'Doğruluk'])
dogruluk_değerleri = pd.DataFrame(columns=['C Değeri', 'Eğitim Doğruluğu', 'Test Doğruluğu'])
for c in C_değerleri:
    lr = LogisticRegression(penalty='l2', C=c, random_state=0)
    lr.fit(X_eğitim, y_eğitim)
    dogruluk_değerleri = dogruluk_değerleri.append({'C Değeri': c,
                                                    'Eğitim Doğruluğu': lr.score(X_eğitim, y_eğitim),
                                                    'Test Doğruluğu': lr.score(X_test, y_test)
                                                    }, ignore_index=True)
print(dogruluk_değerleri)

#Elimizde daha çok veri doluğu için C değerleri değişse de sonuçlar birbirine çok yakın çıkıyor.


print("over-sample f1_score() değeri        : {:.2f}".format(f1_score(y_test, tahmin_test)))
print("over-sample recall_score() değeri    : {:.2f}".format(recall_score(y_test, tahmin_test)))
print("over-sample precision_score() değeri : {:.2f}".format(precision_score(y_test, tahmin_test)))

#under-sample yaptığımızda çıkan değerlere yakın değerler çıktı.


#----------------------------------------------------------------------------------------------------------------
#Smote ile verilerimizi çoğalttığımızda

y = df.Class
X = df.drop('Class', axis=1)

sm = SMOTE(random_state=27)
X_smote, y_smote = sm.fit_sample(X, y)

X_eğitim, X_test, y_eğitim, y_test =  train_test_split(X_smote, y_smote, test_size=0.20, random_state=111)
log_reg.fit(X_eğitim, y_eğitim)

tahmin_eğitim = log_reg.predict(X_eğitim)
tahmin_test = log_reg.predict(X_test)

print("\nSmote kullandıktan sonra: ")
print("Modelin doğruluk değeri : ",  log_reg.score(X_test, y_test))
print("Test veri kümesi")
print(classification_report(y_test,tahmin_test))

print("smote f1_score() değeri        : {:.2f}".format(f1_score(y_test, tahmin_test)))
print("smote recall_score() değeri    : {:.2f}".format(recall_score(y_test, tahmin_test)))
print("smote precision_score() değeri : {:.2f}".format(precision_score(y_test, tahmin_test)))

#Smote kullandığımızda f1,recall,precision değerlerimiz ve modelin doğruluk değeri beklenildiği gibi arttı.



#----------------------------------------------------------------------------------------------------------------
# Adasyn kullandığımızda


y = df.Class
X = df.drop('Class', axis=1)

ad = ADASYN()
X_adasyn, y_adasyn = ad.fit_sample(X, y)

X_eğitim, X_test, y_eğitim, y_test =  train_test_split(X_adasyn, y_adasyn, test_size=0.20, random_state=111)
log_reg.fit(X_eğitim, y_eğitim)

tahmin_eğitim = log_reg.predict(X_eğitim)
tahmin_test = log_reg.predict(X_test)

print("\nAdasyn kullandıktan sonra: ")
print("Modelin doğruluk değeri : ",  log_reg.score(X_test, y_test))
print("Test veri kümesi")
print(classification_report(y_test,tahmin_test))

print("Adasyn f1_score() değeri        : {:.2f}".format(f1_score(y_test, tahmin_test)))
print("Adasyn recall_score() değeri    : {:.2f}".format(recall_score(y_test, tahmin_test)))
print("Adasyn precision_score() değeri : {:.2f}".format(precision_score(y_test, tahmin_test)))

#Adasyn kullandığımızda da bütün değerlerimiz bir hayli fazla çıktı. Az olmakla beraber smote'dan daha yüksek değerler alıyoruz.


#----------------------------------------------------------------------------------------------------------------
#Modelimizde under-sampling yaptıktan sonra çapraz doğrulama uygulayalım

fraud_df = df.loc[df['Class'] == 1]
non_fraud_df = df.loc[df['Class'] == 0][:492]

normal_distributed_df = pd.concat([fraud_df, non_fraud_df])

new_df = normal_distributed_df.sample(frac=1, random_state=42)

print('Under sampling yaptıktan sonra fraud ve no-fraud verilerin dağılımı:')

print(new_df.Class.value_counts())

X = new_df.drop('Class', axis=1)
y = new_df['Class']

X_egitim, X_test, y_egitim, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
log_reg.fit(X_egitim, y_egitim)
tahmin_eğitim = log_reg.predict(X_egitim)
tahmin_test = log_reg.predict(X_test)
print("Under-sampling sonrası modelin skoru: ",log_reg.score(X_test, y_test))




##cross_val_score ve cross_validate ile çapraz doğrulama
lrm = LogisticRegression()
cv = cross_validate(estimator=lrm,
                     X=X,
                     y=y,
                     cv=10,
                    return_train_score=True
                    )


print('\nCross validate ile Test Kümesi   Ortalaması : ', cv['test_score'].mean())
print('Cross validate ile Eğitim Kümesi Ortalaması : ', cv['train_score'].mean())

cv = cross_validate(estimator=lrm,
                     X=X,
                     y=y,
                     cv=10,
                     scoring = ['accuracy', 'precision', 'r2'],
                    return_train_score=True
                    )
print('Cross validate ile Test Kümesi Doğruluk Ortalaması     : {:.2f}'.format(cv['test_accuracy'].mean()))
print('Cross validate ile Test Kümesi R-kare  Ortalaması      : {:.2f}'.format(cv['test_r2'].mean()))
print('Cross validate ile Test Kümesi Hassasiyet Ortalaması   : {:.2f}'.format(cv['test_precision'].mean()))
print('Cross validate ile Eğitim Kümesi Doğruluk Ortalaması   : {:.2f}'.format(cv['train_accuracy'].mean()))
print('Cross validate ile Eğitim Kümesi R-kare  Ortalaması    : {:.2f}'.format(cv['train_r2'].mean()))
print('Cross validate ile Eğitim Kümesi Hassasiyet Ortalaması : {:.2f}'.format(cv['train_precision'].mean()))

#grid search ile parametre ayarlama

parametreler = {"C": [10 ** x for x in range (-5, 5, 1)],
                "penalty": ['l1', 'l2']
                }

grid_cv = GridSearchCV(estimator=log_reg,
                       param_grid = parametreler,
                       cv = 10
                      )
grid_cv.fit(X, y)

print("Grid search ile en iyi parametreler : ", grid_cv.best_params_)
print("Grid search ile en iyi skor         : ", grid_cv.best_score_)

#RandomizedSearchCV

parametreler = {"C": [10 ** x for x in range (-5, 5, 1)],
                "penalty": ['l1', 'l2']
                }
rs_cv = RandomizedSearchCV(estimator=log_reg,
                           param_distributions = parametreler,
                           cv = 10,
                           n_iter = 10,
                           random_state = 111,
                           scoring = 'precision'
                      )
rs_cv.fit(X, y)

print("Randomized search ile en iyi parametreler        : ", rs_cv.best_params_)
print("Randomized search ile en iyi hassasiyet değeri   : ", rs_cv.best_score_)













