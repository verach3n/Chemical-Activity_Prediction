# Chemical-Activity_Prediction
 
## Project Introduction

The task is to develop a predictive model, using a suitable representation (feature set) and learning algorithm, in order to maximize the area under ROC curve (AUC) on the test set (for which you are not given the labels). The task also contains an estimate is provided on what AUC is expected on the test set.

The training set can be found [here](https://canvas.kth.se/courses/36229/files/5822929/download?wrap=1) and the test set here Download [here](https://canvas.kth.se/courses/36229/files/5822930/download?wrap=1).

In this study, three datasets were used and two different normalization techniques were applied to the data. A total of four different machine learning models were trained and tested, with six different model configurations for each model type. This resulted in a total of 24 models being trained and evaluated. In order to select the best performing model among the 24 trained models, a technique called GridSearchCV was used. This method involves training and evaluating each model with a range of different hyperparameters, and then selecting the model configuration that performs the best. To do this, the dataset was split into 10 folds, and each model was trained and evaluated 10 times, with a different fold being used as the test set each time. This allowed the models to be thoroughly evaluated and ensured that the results were reliable. The hyperparameter that yields the highest average AUC performance is selected for further analysis. After training and evaluating all of the models using GridSearchCV, the best hyperparameters for each model were identified. These hyperparameters were then used to train new models on the entire dataset, without using cross-validation. These models were then used to make predictions on the test data, and their performance was evaluated. The best performing model was then selected for the final model to be used for prediction on the test instances.

After using a template to obtain 24 result files from the test dataset, they were combined into a dataframe with "{model_name}_{norm_name}_{dataset_name}_Result.csv" as the column name and the first row as AUC_test. The following figure provides an overview.

As a result, the largest AUC_est is 0.870776423112391, obtained using the random forest algorithm with min-max normalization, and the features used are a combination of those generated from subpackages Chemical Descriptors and Morgan Fingerprints.

## Introduction to SMILE

Chemical compounds can be represented by text strings using Simplified Molecular Input Line Entry Specification (SMILES) - for a brief introduction, see [here](https://sv.wikipedia.org/wiki/Simplified_Molecular_Input_Line_Entry_Specification).

In this project, a predictive model was developed, using a training set with 156 258 chemical compounds represented by SMILES together with their activity label (0.0 = inactive, 1.0 =active) w.r.t. some biological activity, on the following form:

 ```
 INDEX,SMILES,ACTIVE
 1,CC(C)N1CC(=O)C(c2nc3ccccc3[nH]2)=C1N,0.0
 2,COc1ccc(-c2ccc3c(N)c(C(=O)c4ccc(OC)c(OC)c4)sc3n2)cc1,0.0
 3,CCc1ccc(C(=O)COC(=O)CCc2nc(=O)c3ccccc3[nH]2)cc1,0.0
 ...
 ```

and apply it to 52 086 instances in a test set on the form:

```
INDEX,SMILES
156259,COCCCNc1ncnc2c1cnn2-c1ccc(C)cc1C
156260,Cc1cccc(Nc2nnc(SCC(=O)NCc3cccs3)s2)c1C
156261,O=C1/C(=C/c2cccnc2)CC/C1=C\c1cccnc1
...
```

## Toolkit Setup

In order to generate features from SMILES, you may employ the open source toolkit for cheminformatics `RDKit`, see [here](https://www.rdkit.org/) and [here](https://rdkit.readthedocs.io/en/latest/).

You may install `RDKit` by entering (on the command line):

```
 pip install rdkit
```

Then you may import and work with the package from Python, e.g.,

```
 from rdkit import Chem

 m = Chem.MolFromSmiles('Cc1ccccc1')
```

'Cc1ccccc1' is a SMILES string and m will be assigned an object representing the corresponding chemical compound, for which various properties may be derived, e.g.,

```
 m.GetNumAtoms()
7
```

`RDKit` contains several subpackages that was used to generate features, including:

- **rdMolDescriptors** (see [here](https://www.rdkit.org/docs/source/rdkit.Chem.rdMolDescriptors.html) for details)

```
 import rdkit.Chem.rdMolDescriptors as d
 d.CalcExactMolWt(m)
92.062600256
```

- **Fragments** (see [here](https://www.rdkit.org/docs/source/rdkit.Chem.Fragments.html) for details)

```
 import rdkit.Chem.Fragments as f
 f.fr_Al_COO(m)
 0
```

- **Lipinski** (see [here](https://www.rdkit.org/docs/source/rdkit.Chem.Lipinski.html) for details)

```
 import rdkit.Chem.Lipinski as l
 l.HeavyAtomCount(m)
 7
```

- **Fingerprints**

A special class of features is the so-called fingerprints, which represent presence or absence of substructures. They can be derived in many different ways. One of these that is included in RDKit is the so-called Morgan fingerprints, which can be generated as follows:

```
 from rdkit.Chem import AllChem

 fp = AllChem.GetMorganFingerprintAsBitVect(m,2,nBits=124)
 np.array(fp)
 array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0,0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0, 0, 0, 0, 0,   0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0, 1,0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0])
```

Here, the second argument corresponds to the size of the substructures and the third argument corresponds to how many dimensions to map the substructures to (length of the bit vector).

## Directory structure

This repository is structured as follows:

- `Features.ipynb`:

- `KNeighborsClassifier_Result`: 

- `LogisticRegression_Result`

- `RandomForestClassifier_Result`

- `XGBoost_Result`

- `Template.ipynb`

- `FinalSelection.ipynb`

- `Template.ipynb` 
