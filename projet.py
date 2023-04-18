import math
import numpy as np
import utils
import pandas as pd 
from scipy.stats import chi2_contingency 
from IPython.display import Image
import pydotplus
import matplotlib
import matplotlib.pyplot as plt  

#-------------------------Question 1-------------------------------------------
def getPrior(df) :
    """
    Calcule la probabilité a priori de la classe 1 ainsi que l'intervalle de
    confiance à 95% pour l'estimation de cette probabilité.
    
    :param df: Dataframe contenant les données. Doit contenir une colonne
               nommée "target" ne contenant que des 0 et 1.
    :type df: pandas dataframe
    :return: Dictionnaire contennant la moyenne et les extremités de l'intervalle
             de confiance. Clés 'estimation', 'min5pourcent', 'max5pourcent'.
    :rtype: Dictionnaire
    """
    p = df["target"].mean()
    n = len(df)
    return {"estimation": p,
            "min5pourcent": p-1.96*math.sqrt(p*(1-p)/n),
            "max5pourcent": p+1.96*math.sqrt(p*(1-p)/n),              
           }

#-------------------------Question 2---------------------------------------

class APrioriClassifier(utils.AbstractClassifier) :
    """
    Estime très simplement la classe de chaque individu par la classe majoritaire.
    """
    def __init__(self):
        pass
    def estimClass(self, attrs):
        """
        à partir d'un dictionanire d'attributs, estime la classe 0 ou 1
        
        Pour ce APrioriClassifier, la classe vaut toujours 1.
        
        :param attrs: le  dictionnaire nom-valeur des attributs
        :return: la classe 0 ou 1 estimée
        """
        return 1
   
    def statsOnDF(self, df):
        """
        à partir d'un pandas.dataframe, calcule les taux d'erreurs de classification et rend un dictionnaire.
        
        :param df:  le dataframe à tester
        :return: un dictionnaire incluant les VP,FP,VN,FN,précision et rappel
        
        VP : nombre d'individus avec target=1 et classe prévue=1
        VN : nombre d'individus avec target=0 et classe prévue=0
        FP : nombre d'individus avec target=0 et classe prévue=1
        FN : nombre d'individus avec target=1 et classe prévue=0
        Précision : combien de candidats sélectionnés sont pertinents (VP/(VP+FP))
        Rappel : combien d'éléments pertinents sont sélectionnés (VP/(VP+FN))
        """        
         
        dico = {'VP': 0, 'VN': 0, 'FP': 0, 'FN': 0 , 'Précision': 0 , 'Rappel': 0 }
       
        for t in df.itertuples():
            dic = t._asdict()
            estimClass = self.estimClass(dic)
           
            if dic['target'] == 1 and estimClass == 1:
                dico['VP'] += 1
            elif dic['target'] == 0 and estimClass == 0:
                dico['VN'] += 1
            elif dic['target'] == 0 and estimClass == 1:
                dico['FP'] += 1
            else:
                dico['FN'] += 1
                   
        dico['Précision'] = dico['VP']/(dico['VP'] + dico['FP'])
        dico['Rappel'] = dico['VP']/ (dico['VP'] + dico['FN'])
       
        return dico

#-------------------------Question 3---------------------------------------
#********** Question 3.a **********#  

def P2D_l(df,attr):
    """
    Calcul de la probabilité conditionnelle P(attribut | target).
    
    :param df: dataframe avec les données. Doit contenir une colonne nommée "target".
    :param attr: attribut à utiliser, nom d'une colonne du dataframe.
    :return: dictionnaire de dictionnaire dico. dico[t][a] contient P(attribut = a | target = t).
    :rtype: dictionnaire de dictionnaire.
    """

    # initialisation des valeurs du dictionnaire avec les valeurs de target
    res = dict()
    for target in df['target'].unique():
        tmp = dict()
        for val_attr in df[attr].unique():
            tmp[val_attr] = 0
        res[target] = tmp
   
    group = df.groupby(["target", attr]).groups
    for t, val in group:
        res[t][val] = len(group[(t, val)])
   
    group = df.groupby(["target"])[attr].count()
    
    # calcul de la probabilité
    for target in res.keys():
        for val_attribut in res[target].keys():
            res[target][val_attribut] /= group[target]

    return res


def P2D_p(df, attr):
    """
    Calcul de la probabilité conditionnelle P(target | attribut).
    
    :param df: dataframe avec les données. Doit contenir une colonne nommée "target".
    :param attr: attribut à utiliser, nom d'une colonne du dataframe.
    :return: dictionnaire de dictionnaire dico. dico[a][t] contient P(target = t | attribut = a).
    :rtype: dictionnaire de dictionnaire.
    """

    # initialisation des valeurs du dictionnaire avec les valeurs de target
    res = dict()
    for val_attr in df[attr].unique():
        tmp = dict()
        for val_targ in df['target'].unique():
            tmp[val_targ] = 0
        res[val_attr] = tmp
       
    group = df.groupby([attr, "target"]).groups
    for t, val in group:
        res[t][val] = len(group[(t, val)])

    group = df.groupby([attr])['target'].count()
    # calcul de la probabilité
    for val_attr in res.keys():
        for val_targ in res[val_attr].keys():
            res[val_attr][val_targ] /= group[val_attr]
    return res  
   
#********** Question 3.b **********#  

class ML2DClassifier(APrioriClassifier):
    """
    Classifieur 2D par maximum de vraisemblance à partir d'une seule colonne du dataframe.
    """
    def __init__(self, df, attr):
        """
        Initialise le classifieur. Crée un dictionnaire de dictionnaires contenant
        les probabilités conditionnelles P(attribut | target).
        
        :param df: dataframe. Doit contenir une colonne appelée "target" .
        :param attr: le nom d'une colonne du dataframe df.
        """       
        self.attr = attr
        self.dico = P2D_l(df, attr)

    def estimClass(self, attrs):
        """
        à partir d'un dictionanire d'attributs, estime la classe.        
        L'estimée est faite par maximum de vraisemblance à partir de dico.
        
        :param attrs: le  dictionnaire nom-valeur des attributs
        :return: la classe estimée
        """
        val_attr = attrs[self.attr]
        classe = 0
        proba = 0
        for i in self.dico.keys():
            if self.dico[i][val_attr] >= proba :
                classe = i
                proba = self.dico[i][val_attr]
        return classe

#********** Question 3.c **********#  

class MAP2DClassifier(APrioriClassifier):
    """
    Classifieur 2D par maximum a posteriori à partir d'une seule colonne du dataframe.
    """
    def __init__(self, df, attr):
        """
        Initialise le classifieur. Crée un dictionnaire de dictionnaires contenant
        les probabilités conditionnelles P(target | attribut).
        
        :param df: dataframe. Doit contenir une colonne appelée "target".
        :param attr: le nom d'une colonne du dataframe df.
        """
        self.attr = attr
        self.dico = P2D_p(df, attr)
       
    def estimClass(self, attrs):
        """
        à partir d'un dictionanire d'attributs, estime la classe.        
        L'estimée est faite par maximum de vraisemblance à partir de dico.
        
        :param attrs: le  dictionnaire nom-valeur des attributs
        :return: la classe estimée
        """
        val_attr = attrs[self.attr]
        classe = 0
        proba = 0
        for i in self.dico[val_attr].keys():
            if self.dico[val_attr][i] >= proba :
                classe = i
                proba = self.dico[val_attr][i]
        return classe

#-------------------------Question 4---------------------------------------

#********** Question 4.1 **********#  

def nbParams(df, liste = None):
    """
    Affiche la taille mémoire de tables P(target|attr1,..,attrk) étant donné un
    dataframe df et la liste [target,attr1,...,attrl], en supposant qu'un float
    est représenté sur 8 octets.
   
    :param df: Dataframe contenant les données.
    :param liste: liste contenant les colonnes prises en considération pour le calcul.
    """
    taille = 8
    if liste is None:
        liste = list(df.columns)
    for attr in liste:
        taille *= np.unique(df[attr]).size
    print("{} variable(s) : {} octets".format(len(liste), taille))

#********** Question 4.2 **********#  

def nbParamsIndep(df):
    """
    Affiche la taille mémoire de tables P(target|attr1,..,attrk) étant donné un
    dataframe df et la liste [target,attr1,...,attrl], en supposant qu'un float
    est représenté sur 8 octets.
   
    :param df: Dataframe contenant les données.
    :param liste: liste contenant les colonnes prises en considération pour le calcul.
    """
    taille = 0
    liste = list(df.columns)
    for attr in liste:
        taille += np.unique(df[attr]).size * 8
    print("{} variable(s) : {} octets".format(len(liste), taille))

#-------------------------Question 5---------------------------------------

#********** Question 5.3 **********#  

def drawNaiveBayes(df, root):
    """
    Construit un graphe orienté représentant naïve Bayes.
   
    :param df: Dataframe contenant les données.  
    :param root: le nom de la colonne du Dataframe utilisée comme racine.
    """
    chaine = ""

    attrs = list(df.columns); #recuperer tous les attributs  
    attrs.remove(root);

    for attr in attrs:
        chaine += root + "->" + attr + ";"
    return utils.drawGraph(chaine)

def nbParamsNaiveBayes(df, root, attrs = None):
    """
    Affiche la taille mémoire de tables P(target), P(attr1|target),.., P(attrk|target)
    étant donné un dataframe df, la colonne racine col_pere et la liste [target,attr1,...,attrk],
    en supposant qu'un float est représenté sur 8 octets.
    :param df: Dataframe contenant les données.
    :param root: le nom de la colonne du Dataframe utilisée comme racine.
    :param attrs: liste contenant les colonnes prises en considération pour le calcul.
    """
    taille = np.unique(df[root]).size
    size = taille*8
   
    if attrs is None:
        attrs = list(df.columns)

    for attr in attrs:
        if attr == root:
            continue
        tmp = (taille * np.unique(df[attr]).size) * 8
        size += tmp
   
    print("{} variable(s) : {} octets".format(len(attrs), size))

#********** Question 5.4 **********#  
   
class MLNaiveBayesClassifier(APrioriClassifier):
    """
    Classifieur par maximum de vraissemblance utilisant le modèle naïve Bayes.
    """
    def __init__(self, df):
        """
        Initialise le classifieur. Crée un dictionnarie où la clé est le nom de
        chaque attribut et la valeur est un dictionnaire de dictionnaires contenant
        les probabilités conditionnelles P(attribut | target).
       
        :param df: dataframe. Doit contenir une colonne appelée "target".
        """
        self.df = df
        self.dico_P2D_l = {}
        attrs = list(df.columns)
        attrs.remove("target")
        for attr in attrs:
            self.dico_P2D_l[attr] = P2D_l(df, attr)
   
    def estimClass(self, attrs):
        """
        À partir d'un dictionanire d'attributs,on estime la classe.        
       
        :param attrs: le dictionnaire nom-valeur des attributs
        :return: la classe estimée
        """
        vraissemblance = self.estimProbas(attrs)
        classe = 0
        proba = 0.0
        for i in vraissemblance.keys():
            if vraissemblance[i] >= proba :
                classe = i
                proba = vraissemblance[i]
        return classe
       
    def estimProbas(self, attrs):
        """
        Calcule la vraisemblance par naïve Bayes : P(attr1, ..., attrk | target).
        :param attrs: le dictionnaire nom-valeur des attributs
        """    
        val_target = self.df['target'].unique() #target [1,0]
        dico_proba = dict.fromkeys(val_target, 1) # dico_proba = {1:1 , 0:1}
       
        for attr in self.dico_P2D_l.keys() :
            dico_tmp  =  self.dico_P2D_l[attr]
            for key_target in dico_proba.keys():
                """
                Si la clé n'existe pas dans dico_tmp
                Alors on fait une anticipé en renvoyant {1:0.0, 0:0.0}
                """
                if attrs[attr] not in dico_tmp[key_target]:  
                    res = dict.fromkeys(val_target, 0.0)
                    return res
                dico_proba[key_target] *= dico_tmp[key_target][attrs[attr]]
        return dico_proba
           
class MAPNaiveBayesClassifier(APrioriClassifier):
    """
    Classifieur par le maximum a posteriori en utilisant le modèle naïve Bayes.
    """
    def __init__(self, df):
        """
        Initialise le classifieur. Crée un dictionnarie où la clé est le nom de
        chaque attribut et la valeur est un dictionnaire de dictionnaires contenant
        les probabilités conditionnelles P(attribut | target). Cree aussi un
        dictionnaire avec les probabilités de target.
       
        :param df: dataframe. Doit contenir une colonne appelée "target".
        """
        self.df = df
        self.dico_P2D_l = {}
        tab_col = list(df.columns.values)
        tab_col.remove("target")
        for attr in tab_col:
            self.dico_P2D_l[attr] = P2D_l(df, attr)
   
    def estimClass(self, attrs):
        """
        À partir d'un dictionanire d'attributs,on estime la classe.         
       
        :param attrs: le dictionnaire nom-valeur des attributs
        :return: la classe estimée
        """
        vraissemblance = self.estimProbas(attrs)
        classe = 0
        proba = 0.0
        for i in vraissemblance.keys():
            if vraissemblance[i] >= proba :
                classe = i
                proba = vraissemblance[i]
        return classe
       

    def estimProbas(self, attrs):
        """
        Calcule la probabilité à posteriori par naïve Bayes : P(target | attr1, ..., attrk).
       
        :param attrs: le dictionnaire nom-valeur des attributs
        """    
        val_target = self.df['target'].unique() #target [1,0]
        df = self.df.target.value_counts()/self.df.target.count()
        dico_proba = df.to_dict() #convertir le df en dictionnaire dico_proba = {1:xxx, 0:xxx}
       
        for attr in self.dico_P2D_l.keys() :
            dico_tmp  =  self.dico_P2D_l[attr]
            for key_target in dico_proba.keys():
                """
                Si la clé n'existe pas dans dico_tmp
                Alors on fait une anticipé en renvoyant {1:0.0, 0:0.0}
                """
                if attrs[attr] not in dico_tmp[key_target]:  
                    res = dict.fromkeys(val_target, 0.0)
                    return res
                dico_proba[key_target] *= dico_tmp[key_target][attrs[attr]]
       
        somme_proba = sum(dico_proba.values()) # contient la somme des probas calculées
       
        for prob in dico_proba.keys() :
            dico_proba[prob] /= somme_proba
       
        return dico_proba

#-------------------------Question 6---------------------------------------     

def isIndepFromTarget(df,attr,x) :
    """
    Vérifie si attr est indépendant de target au seuil de x%.
    
    :param df: dataframe. Doit contenir une colonne appelée "target".
    :param attr: le nom d'une colonne du dataframe df.
    :param x: seuil de confiance.
    """
    liste = pd.crosstab(df[attr], df.target) #transformer le df en liste
    _,p,_,_ = chi2_contingency(liste) # calculer le p_value
    if p < x : # on rejette l'hypothese si p_value est inferieur au seuil de x%
        return False
    return True

class ReducedMLNaiveBayesClassifier(APrioriClassifier):
    """
    Classifieur par maximum de vraissemblance utilisant le modèle naïve Bayes reduit. 
    """   
    def __init__(self, df, x):
        """
        Initialise le classifieur. Crée un dictionnarie où la clé est le nom de
        chaque attribut et la valeur est un dictionnaire de dictionnaires contenant
        les probabilités conditionnelles P(attribut | target).
       
        :param df: dataframe. Doit contenir une colonne appelée "target".
        """
        self.df = df
        self.dico_P2D_l = {}
        attrs = list(df.columns) # recupere tous les attributs
        attrs.remove("target")
        for attr in attrs:
            if not isIndepFromTarget(df, attr, x):
                self.dico_P2D_l[attr] = P2D_l(df, attr)
   
    def estimClass(self, attrs):
        """
        À partir d'un dictionanire d'attributs, on estime la classe.        
       
        :param attrs: le dictionnaire nom-valeur des attributs
        :return: la classe estimée
        """
        vraissemblance = self.estimProbas(attrs)
        classe = 0
        proba = 0.0
        for i in vraissemblance.keys():
            if vraissemblance[i] > proba :
                classe = i
                proba = vraissemblance[i]
        return classe
       
    def estimProbas(self, attrs):
        """
        Calcule la vraisemblance par naïve Bayes : P(attr1, ..., attrk | target).
        :param attrs: le dictionnaire nom-valeur des attributs
        """    
        val_target = self.df['target'].unique() #target [1,0]
        dico_proba = dict.fromkeys(val_target, 1) # dico_proba = {1:1 , 0:1}
       
        for attr in self.dico_P2D_l.keys() :
            dico_tmp  =  self.dico_P2D_l[attr]
            for key_target in dico_proba.keys():
                """
                Si la clé n'existe pas dans dico_tmp
                Alors on fait une anticipé en renvoyant {1:0.0, 0:0.0}
                """
                if attrs[attr] not in dico_tmp[key_target]:  
                    res = dict.fromkeys(val_target, 0.0)
                    return res
                dico_proba[key_target] *= dico_tmp[key_target][attrs[attr]]
        return dico_proba
   
    def draw(self) :
        chaine = ""
        for attr in self.dico_P2D_l.keys():
            chaine += "target" + "->" + attr + ";"
        return utils.drawGraph(chaine)
       
class ReducedMAPNaiveBayesClassifier(APrioriClassifier):
    """
    Classifieur par le maximum a posteriori en utilisant le modèle naïve Bayes.
    """
    def __init__(self, df, x):
        """
        Initialise le classifieur. Crée un dictionnarie où la clé est le nom de
        chaque attribut et la valeur est un dictionnaire de dictionnaires contenant
        les probabilités conditionnelles P(attribut | target). Cree aussi un
        dictionnaire avec les probabilités de target.
       
        :param df: dataframe. Doit contenir une colonne appelée "target".
        """
        self.df = df
        self.dico_P2D_l = {}
        tab_col = list(df.columns.values)
        tab_col.remove("target")
        for attr in tab_col:
            if not isIndepFromTarget(df, attr, x):
                self.dico_P2D_l[attr] = P2D_l(df, attr)
   
    def estimClass(self, attrs):
        """
        À partir d'un dictionanire d'attributs, on estime la classe.        
       
        :param attrs: le dictionnaire nom-valeur des attributs
        :return: la classe estimée
        """
        vraissemblance = self.estimProbas(attrs)
        classe = 0
        proba = 0.0
        for i in vraissemblance.keys():
            if vraissemblance[i] > proba :
                classe = i
                proba = vraissemblance[i]
        return classe
       

    def estimProbas(self, attrs):
        """
        Calcule la probabilité à posteriori par naïve Bayes : P(target | attr1, ..., attrk).
       
        :param attrs: le dictionnaire nom-valeur des attributs
        """    
        val_target = self.df['target'].unique() #target [1,0]
        df = self.df.target.value_counts()/self.df.target.count()
        dico_proba = df.to_dict() #convertir le df en dictionnaire dico_proba = {1:xxx, 0:xxx}
       
        for attr in self.dico_P2D_l.keys() :
            dico_tmp  =  self.dico_P2D_l[attr]
            for key_target in dico_proba.keys():
                """
                Si la clé n'existe pas dans dico_tmp
                Alors on fait une anticipé en renvoyant {1:0.0, 0:0.0}
                """
                if attrs[attr] not in dico_tmp[key_target]:  
                    res = dict.fromkeys(val_target, 0.0)
                    return res
                dico_proba[key_target] *= dico_tmp[key_target][attrs[attr]]
       
        somme_proba = sum(dico_proba.values()) # contient la somme des probas calculées
       
        for prob in dico_proba.keys() :
            dico_proba[prob] /= somme_proba
       
        return dico_proba
   
    def draw(self) :
        chaine = ""
        for attr in self.dico_P2D_l.keys():
            chaine += "target" + "->" + attr + ";"
        return utils.drawGraph(chaine)

#-------------------------Question 7---------------------------------------   

def mapClassifiers(dic, df):
    """
    Représente graphiquement les classifiers à partir d'un dictionnaire dic de 
    {nom:instance de classifier} et d'un dataframe df, dans l'espace (précision,rappel). 
    
    :param dic: dictionnaire {nom:instance de classifier}
    :param df: dataframe. Doit contenir une colonne appelée "target".
    """
    precision = np.empty(len(dic))
    rappel = np.empty(len(dic))
    
    for i, nom in enumerate(dic):
         dico_stats = dic[nom].statsOnDF(df)
         precision[i] = dico_stats["Précision"]
         rappel[i] = dico_stats["Rappel"]
    
    fig, ax = plt.subplots()
    ax.grid(True)
    ax.set_axisbelow(True)
    ax.set_xlabel("Précision")
    ax.set_ylabel("Rappel")
    ax.scatter(precision, rappel, marker = 'x', c = 'red') 
    
    for i, nom in enumerate(dic):
        ax.annotate(nom, (precision[i], rappel[i]))
    
    plt.show()

#-------------------------Question 8---------------------------------------  

#********** Question 8.1 **********#   

def MutualInformation(df, x, y):
    """
    Calcule l'information mutuelle entre les colonnes x et y du dataframe.
    
    :param df: Dataframe contenant les données. 
    :param x: nom d'une colonne du dataframe.
    :param y: nom d'une colonne du dataframe.
    """
    list_x = np.unique(df[x].values) # Valeurs possibles de x.
    list_y = np.unique(df[y].values) # Valeurs possibles de y.
    
    dico_x = {list_x[i]: i for i in range(list_x.size)} 
    #un dictionnaire associant chaque valeur a leur indice en list_x.
    
    dico_y = {list_y[i]: i for i in range(list_y.size)} 
    #un dictionnaire associant chaque valeur a leur indice en list_y.
    
    mat_xy = np.zeros((list_x.size, list_y.size), dtype = int)
    #matrice des valeurs P(x,y)
    
    group = df.groupby([x, y]).groups
    
    for i, j in group:
        mat_xy[dico_x[i], dico_y[j]] = len(group[(i, j)]) 
    
    mat_xy = mat_xy / mat_xy.sum()
    
    mat_x = mat_xy.sum(1)
    #matrice des P(x)
    mat_y = mat_xy.sum(0)
    #matrice des P(y)
    mat_px_py = np.dot(mat_x.reshape((mat_x.size, 1)),mat_y.reshape((1, mat_y.size))) 
    #matrice des P(x)P(y)
    
    mat_res = mat_xy / mat_px_py
    mat_res[mat_res == 0] = 1
    #pour éviter des problèmes avec le log de zero
    mat_res = np.log2(mat_res)
    mat_res *= mat_xy
    
    return mat_res.sum()

def ConditionalMutualInformation(df,x,y,z):
    """
    Calcule l'information mutuelle conditionnelle entre les colonnes x et y du 
    dataframe en considerant les deux comme dependantes de la colonne z.
    
    :param df: Dataframe contenant les données. 
    :param x: nom d'une colonne du dataframe.
    :param y: nom d'une colonne du dataframe.
    :param z: nom d'une colonne du dataframe.
    """
    list_x = np.unique(df[x].values) # Valeurs possibles de x.
    list_y = np.unique(df[y].values) # Valeurs possibles de y.
    list_z = np.unique(df[z].values) # Valeurs possibles de z.
    
    dico_x = {list_x[i]: i for i in range(list_x.size)} 
    #un dictionnaire associant chaque valeur a leur indice en list_x.
    
    dico_y = {list_y[i]: i for i in range(list_y.size)} 
    #un dictionnaire associant chaque valeur a leur indice en list_y.
    
    dico_z = {list_z[i]: i for i in range(list_z.size)} 
    #un dictionnaire associant chaque valeur a leur indice en list_z.
    
    mat_xyz = np.zeros((list_x.size, list_y.size, list_z.size), dtype = int)
    #matrice des valeurs P(x,y,z)
    
    group = df.groupby([x, y, z]).groups
    
    for i, j, k in group:
        mat_xyz[dico_x[i], dico_y[j], dico_z[k]] = len(group[(i, j, k)]) 
    
#    for _, row in df.iterrows():
#        i = row[x]
#        j = row[y]
#        mat_xy[dico_x[i], dico_y[j]]+= 1 
    
    mat_xyz = mat_xyz / mat_xyz.sum()
    
    mat_xz = mat_xyz.sum(1)
    #matrice des P(x, z)
    
    mat_yz = mat_xyz.sum(0)
    #matrice des P(y, z)
    
    mat_z = mat_xz.sum(0)
    #matrice des P(z)
    
    mat_pxz_pyz = mat_xz.reshape((list_x.size, 1, list_z.size)) * mat_yz.reshape((1, list_y.size, list_z.size)) 
    #matrice des P(x, z)P(y, z)
    
    mat_pxz_pyz[mat_pxz_pyz == 0] = 1
    # Certains éléments de max_pxz_pyz peuvent être zéro, ce qui pose un problème
    # pour la division. On les change à 1 pour éviter ce problème. On remarque
    # que, si une case de cette matrice vaut 0, alors la case correspondante de
    # la matrice mat_xyz vaut aussi 0. En effet, si une case vaut 0, alors
    # P(x, z) = 0 ou P(y, z) = 0. Si on est dans le premier cas, comme
    # P(x, z) est la somme de P(x, y, z) sur y et que P(x, y, z) est toujours
    # >= 0, alors forcément P(x, y, z) = 0 pour tout y, et donc les cases
    # correspondantes dans mat_xyz valent 0. Similairement si c'est P(y, z) qui
    # vaut 0.
    
    mat_pz_pxyz = mat_z.reshape((1, 1, list_z.size)) * mat_xyz
    #matrice des P(z)P(x, y, z)
    
    mat_res = mat_pz_pxyz / mat_pxz_pyz
    mat_res[mat_res == 0] = 1
    #pour éviter des problèmes avec le log de zero
    mat_res = np.log2(mat_res)
    mat_res *= mat_xyz
    
    return mat_res.sum()
    
#********** Question 8.2 **********#

def MeanForSymetricWeights(a):   
    """
    Calcule la moyenne des poids pour une matrice a symétrique de diagonale nulle.
    La diagonale n'est pas prise en compte pour le calcul de la moyenne.
    
    :param a: Matrice symétrique de diagonale nulle.  
    """
    return a.sum()/(a.size - a.shape[0])

def SimplifyConditionalMutualInformationMatrix(a):
    """
    Annule toutes les valeurs plus petites que sa moyenne dans une matrice a 
    symétrique de diagonale nulle.
    
    :param a: Matrice symétrique de diagonale nulle.      
    """
    moy = MeanForSymetricWeights(a)
    a[a < moy] = 0
    
#********** Question 8.3 **********#    

def Kruskal(df,a):
    """
    Applique l'algorithme de Kruskal au graphe dont les sommets sont les colonnes
    de df (sauf 'target') et dont la matrice d'adjacence ponderée est a.
    Les indices dans a doivent être dans le même ordre que ceux de df.keys().
    
    :param df: Dataframe contenant les données. 
    :param a: Matrice symétrique de diagonale nulle.    
    """
    list_col = [x for x in df.keys() if x != "target"]
    list_arr = [(list_col[i], list_col[j], a[i, j]) for i in range(a.shape[0]) for j in range(i + 1, a.shape[0]) if a[i, j] != 0]
    
    list_arr.sort(key = lambda x: x[2], reverse = True)
    
    g = Graphe(list_col)
    
    for (u, v, poids) in list_arr:
        if g.find(u) != g.find(v):
            g.addArete(u, v, poids)
            g.union(u, v)
    return g.graphe    

class Graphe:
    """
    Structure de graphe pour l'algorithme de Kruskal. 
    """
  
    def __init__(self, sommets): 
        """
        :param sommets: liste de sommets
        """
        self.S = sommets 
        #Liste de sommets 
        self.graphe = [] 
        #liste representant les aretes du graphe 
        self.parent = {s : s for s in self.S}
        #dictionnaire ou la clé est un sommet et la valeur est sont père
        #dans la forêt utilisée par l'algorithme de kruskal.
        self.taille = {s : 1 for s in self.S}
        #dictionnaire des tailles des arbres dans la forêt
        

    def addArete(self, u, v, poids): 
        """
        Ajoute l'arete (u, v) avec poids.
        
        :param u: le nom d'un sommet.
        :param v: le nom d'un sommet.
        :param poids: poids de l'arete entre les deux sommets.
        """
        self.graphe.append((u,v,poids)) 
  
    def find(self, u): 
        """
        Trouve la racine du sommet u dans la forêt utilisée par l'algorithme de
        kruskal. Avec compression de chemin.

        :param u: le nom d'un sommet.
        """
        racine = u
        #recherche de la racine
        while racine != self.parent[racine]:
            racine = self.parent[u]
        #compression du chemin    
        while u != racine:
            v = self.parent[u]
            self.parent[u] = racine
            u = v
        return racine            
  

    def union(self, u, v):
        """
        Union ponderé des deux arbres contenant u et v. Doivent être dans deux
        arbres differents.
        
        :param u: le nom d'un sommet.
        :param v: le nom d'un sommet.
        """
        u_racine = self.find(u) 
        v_racine = self.find(v) 
  
        if self.taille[u_racine] < self.taille[v_racine]: 
            self.parent[u_racine] = v_racine 
            self.taille[v_racine] += self.taille[u_racine] 
        else: 
            self.parent[v_racine] = u_racine 
            self.taille[u_racine] += self.taille[v_racine] 
 
#********** Question 8.4 **********#  

def ConnexSets(list_arcs):
    """
    Costruit une liste des composantes connexes du graphe dont la liste d'aretes
    est list_arcs.
    
    :param list_arcs: liste de triplets de la forme (sommet1, sommet2, poids).
    """
    res = []
    for (u, v, _) in list_arcs:
        u_set = None
        v_set = None
        for s in res:
            if u in s:
                u_set = s
            if v in s:
                v_set = s
        if u_set is None and v_set is None:
            res.append({u, v})
        elif u_set is None:
            v_set.add(u)
        elif v_set is None:
            u_set.add(v)
        elif u_set != v_set:
            res.remove(u_set)
            v_set = v_set.union(u_set)
    return res

def OrientConnexSets(df, arcs, classe):
    """
    Utilise l'information mutuelle (entre chaque attribut et la classe) pour
    proposer pour chaque ensemble d'attributs connexes une racine et qui rend 
    la liste des arcs orientés.
    
    :param df: Dataframe contenant les données. 
    :param arcs: liste d'ensembles d'arcs connexes.
    :param classe: colonne de réference dans le dataframe pour le calcul de 
    l'information mutuelle.
    """
    arcs_copy = arcs.copy()
    list_sets = ConnexSets(arcs_copy)
    list_arbre = []
    for s in list_sets:
        col_max = ""
        i_max = -float("inf") 
        for col in s:
            i = MutualInformation(df, col, classe)
            if i > i_max:
                i_max = i
                col_max = col
        list_arbre += creeArbre(arcs_copy, col_max)
    return list_arbre
    
def creeArbre(arcs, racine): 
    """
    À partir d'une liste d'arcs et d'une racine, renvoie l'arbre orienté depuis
    cette racine. La liste arcs est modifié par cette fonction.
    
    :param arcs: liste d'ensembles d'arcs connexes.
    :param racine: nom d'un sommet.
    """
    res = []
    file = [racine]
    while file != []:
        sommet = file.pop(0)
        arcs_copy = arcs.copy()
        for (u, v, poids) in arcs_copy:
            if sommet == u:
                res.append((u, v))
                arcs.remove((u, v, poids))
                file.append(v)
            elif sommet == v:
                res.append((v, u))
                arcs.remove((u, v, poids))
                file.append(u)
    return res 
    
#********** Question 8.5 **********#  

class MAPTANClassifier(APrioriClassifier):
    """
    Classifieur par le maximum a posteriori en utilisant le modèle TAN
    (tree-augmented naïve Bayes).
    """
    def __init__(self, df):
        """
        Initialise le classifieur. Crée la matrice de Conditional Mutual Information
        simplifiée, une liste des arcs retenus pour le modèle TAN, un dictionnarie
        pour les probabilités 2D et un dictionnaire pour les probabilités 3D.
        Cree aussi un dictionnaire avec les probabilités de target = 0 et target = 1.
        
        :param df: dataframe. Doit contenir une colonne appelée "target" ne contenant que 0 ou 1.
        """
        self.createCmis(df)
        arcs = Kruskal(df, self.cmis)
        self.liste_arcs = OrientConnexSets(df, arcs, "target")
        
        self.dico_P2D_l = {}
        self.dico_P3D_l = {}
        self.pTarget = {1: df["target"].mean()}
        self.pTarget[0] = 1 - self.pTarget[1] 
        
        tab_col = list(df.columns.values)
        tab_col.remove("target")

        for attr in tab_col:
            pere = self.is3D(attr)
            if pere is not False:
                self.dico_P3D_l[attr] = P3D_l(df, attr, pere)
            else:
                self.dico_P2D_l[attr] = P2D_l(df, attr)
    
    def estimClass(self, attrs):
        """
        À partir d'un dictionanire d'attributs, estime la classe 0 ou 1.        
        L'estimée est faite par maximum à posteriori à partir de dico_res.
        
        :param attrs: le dictionnaire nom-valeur des attributs
        :return: la classe 0 ou 1 estimée
        """
        dico_res = self.estimProbas(attrs)
        if dico_res[0] >= dico_res[1]:
            return 0
        return 1
        

    def estimProbas(self, attrs):
        """
        Calcule la probabilité a posteriori P(target | attr1, ..., attrk) par
        la méthode TAN (tree-augmented naïve Bayes).
             
        :param attrs: le dictionnaire nom-valeur des attributs
        """
        P_0 = self.pTarget[0]
        P_1 = self.pTarget[1]
        for key in self.dico_P2D_l:
            dico_p = self.dico_P2D_l[key]
            if attrs[key] in dico_p[0]:
                P_0 *= dico_p[0][attrs[key]]
                P_1 *= dico_p[1][attrs[key]]
            else:
                #si la valeur de l'attribut n'existe pas dans l'enseble d'apprentissage,
                #alors sa probabilité conditionnelle n'est pas definie et on peut
                #faire une sortie anticipée
                return {0: 0.0, 1: 0.0}
        
        for key in self.dico_P3D_l:
            proba = self.dico_P3D_l[key]
            P_0 *= proba.getProba(attrs[key], attrs[proba.pere], 0)
            P_1 *= proba.getProba(attrs[key], attrs[proba.pere], 1)
        
        if (P_0 + P_1) == 0 : 
            return {0: 0.0, 1: 0.0}
        
        P_0res = P_0 / (P_0 + P_1)
        P_1res = P_1 / (P_0 + P_1)
        return {0: P_0res, 1: P_1res}
    
    def draw(self):
        """
        Construit un graphe orienté représentant le modèle TAN.
        """
        res = ""
        for enfant in self.dico_P2D_l:
            res = res + "target" + "->" + enfant + ";"
        for enfant in self.dico_P3D_l:
            res = res + self.dico_P3D_l[enfant].pere + "->" + enfant + ";"
            res = res + "target" + "->" + enfant + ";"
        return utils.drawGraph(res[:-1])
    
    def createCmis(self, df):
        """
        Crée la matrice de Conditional Mutual Information simplifiée à partir du dataframe df.
        
        :param df: dataframe. Doit contenir une colonne appelée "target" ne contenant que 0 ou 1.
        """
        self.cmis = np.array([[0 if x == y else ConditionalMutualInformation(df, x, y, "target") 
                                 for x in df.keys() if x != "target"] for y in df.keys() if y != "target"])
        SimplifyConditionalMutualInformationMatrix(self.cmis)
        
    def is3D(self, attr):  
        """
        Détermine si l'attribut attr doit être représenté par une matrice 3D,
        c'est-à-dire s'il a un parent outre que "target" dans self.list_arcs.
        
        :param attr: nom d'un attribut du dataframe.
        """
        for pere, fils in self.liste_arcs:
            if fils == attr:
                return pere
        return False
    
    
class P3D_l():
    """
    Classe pour le calcul des probabilités du type P(attr1 | attr2, target).
    """
    def __init__(self, df, attr1, attr2):
        """
        :param df: dataframe. Doit contenir une colonne appelée "target" ne 
                   contenant que 0 ou 1. 
        :param attr1: nom d'une colonne du dataframe.
        :param attr2: nom d'une colonne du dataframe.
        """
        self.pere = attr2
        list_x = np.unique(df[attr1].values) # Valeurs possibles de x.
        list_y = np.unique(df[attr2].values) # Valeurs possibles de y.
         
        self.dico_x = {list_x[i]: i for i in range(list_x.size)} 
        #un dictionnaire associant chaque valeur a leur indice en list_x.      
        self.dico_y = {list_y[i]: i for i in range(list_y.size)} 
        #un dictionnaire associant chaque valeur a leur indice en list_y.
        self.mat = np.zeros((list_x.size, list_y.size, 2))
        #la matrice des probabilités.
        
        group = df.groupby([attr1, attr2, 'target']).groups
        
        #rempli la matrice mat avec la quantité d'élements dans le dataframe
        #ayant les valeurs (i, j, k) pours les attributs attr1, attr2 et target.
        for i, j, k in group:
            self.mat[self.dico_x[i], self.dico_y[j], k] = len(group[(i, j, k)]) 
        
        #on normalise la matrice pour avoirs les probabilités conditionnelles
        #Pour cela, on calcule dans dans quant[j, k] la quantité d'élements dans 
        #le dataframe ayant les valeurs (j, k) pours les attributs attr2 et target.
        quant = self.mat.sum(0)
        quant[quant == 0] = 1 #pour eviter des divisions par zero
        
        self.mat = self.mat / quant.reshape((1, list_y.size, 2))
        
    def getProba(self, i, j, k):
        """
        Renvoie la valeur de P(attr1 = i | attr2 = j, target = k).
        
        :param i: valeur pour l'attribut attr1 de init.
        :param j: valeur pour l'attribut attr2 de init.
        :param k: valeur pour target.
        """
        if i in self.dico_x and j in self.dico_y:
            return self.mat[self.dico_x[i], self.dico_y[j], k]
        return 0.