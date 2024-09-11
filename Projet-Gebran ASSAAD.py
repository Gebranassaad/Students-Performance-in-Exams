import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gs
from sklearn.preprocessing import LabelEncoder

"""
#PieChart School
gp_nb = data[data["school"] == "GP"].count()[1]
ms_nb = data[data["school"] == "MS"].count()[1]
school = [gp_nb,ms_nb]
plt.pie(school , labels = ["GP","MS"],autopct='%1.1f%%', colors = ["#BB8A59","#AAB18E"])
plt.title("School")
plt.show()


#LineChart
GP = data[data["school"] =="GP"]
MS = data[data["school"] == "MS"]
x=['G1','G2','G3']
values1 = [np.mean(GP["G1"]),np.mean(GP["G2"]),np.mean(GP["G3"])]
values2 = [np.mean(MS["G1"]),np.mean(MS["G2"]),np.mean(MS["G3"])]
plt.ylabel("Notes")
plt.plot(x,values1, color = "#BB8A59")
plt.plot(x,values2, color = "#AAB18E")
plt.legend(["Gabriel Pereira","Mousinho da Silveira"])
plt.show()


#Donut Chart
nb_male = data[data["sex"] == "M"].count()[1]
nb_female = data[data["sex"] == "F"].count()[1]
sex = [nb_male,nb_female]
fig, ax = plt.subplots()
ax.pie(sex, labels=["Male","Female"], autopct='%1.1f%%',colors = ["#CDAE46","#9E663A"])
# Draw a circle at the center to create the donut chart
circle = plt.Circle((0, 0), 0.70, fc='#E3E5D2',alpha = 1)  # Adjust the second parameter for the size of the hole
fig.gca().add_artist(circle)
# Add a title
plt.title('Sex of Students')
# Equal aspect ratio ensures that pie is drawn as a circle.
ax.axis('equal')
# Display the chart
plt.show()


#Les attributs Binaires
import matplotlib.gridspec as gs
fig = plt.figure(figsize = (17,15))
g = gs.GridSpec(nrows = 4, ncols = 4, figure = fig)
i = 0
j = 0
for feature in binary:
    if j == 4:
        i = i + 1
        j = 0
    ax1 = plt.subplot(g[i,j])
    ax1 = sns.countplot(df[feature])
    j = j + 1


#BarPlot (Mjob-Fjob-Guardian)
fig,(ax1, ax2 , ax3) = plt.subplots(1, 3, figsize=(15, 5)) 
ax1 = sns.countplot(data["Mjob"],ax = ax1, color ="#C0D6C2")
ax1.set_xlabel("Job")
ax1.set_ylabel("Count")
ax1.set_title("Mother Job")
ax2 = sns.countplot(data["Fjob"],ax = ax2,order = ["at_home","health","other","services","teacher"],color ="#C0D6C2")
ax2.set_xlabel("Job")
ax2.set_ylabel("Count")
ax2.set_title("Father Job")
ax3 = sns.countplot(data["guardian"],ax = ax3, color ="#C0D6C2")
ax3.set_xlabel("Guardian")
ax3.set_ylabel("Count")
ax3.set_title("Student's Guardian")
plt.show()


#BarPlot ( Reason )
sns.countplot(data["reason"])
plt.xlabel("Reason")
plt.ylabel("Count")
plt.title("Reason To Choose This School")
plt.show()





#BarPlot (Traveltime)
sns.countplot(data["traveltime"])
custom_ticks = ['<=15 min', '15 to 30 min', '30 min to 1 h', ">= 1 h"]  # Custom tick labels
tick_positions = [0, 1, 2, 3]
plt.xticks(tick_positions,custom_ticks)
plt.xlabel("Travel Time")
plt.ylabel("Count")
plt.title("Home to School Travel Time")
plt.show()


#BarPlot (Medu-Fedu)
fig,(ax1, ax2) = plt.subplots(1, 2, figsize=(15, 4)) 
ax1 = sns.countplot(data["Medu"],ax = ax1)
custom_ticks = ['None', 'primary education', '5th to 9th grade', "secondary education","higher education"]  # Custom tick labels
tick_positions = [0, 1, 2, 3, 4]
ax1.set_xticks(tick_positions,custom_ticks,rotation = 90)
ax1.set_xlabel("Medu")
ax1.set_ylabel("Count")
ax1.set_title("Mother Education Level")
ax2 = sns.countplot(data["Fedu"],ax = ax2)
custom_ticks = ['None', 'primary education', '5th to 9th grade', "secondary education","higher education"]  # Custom tick labels
tick_positions = [0, 1, 2, 3, 4]
ax2.set_xticks(tick_positions,custom_ticks,rotation = 90)
ax2.set_xlabel("Fedu")
ax2.set_ylabel("Count")
ax2.set_title("Father Education Level")
plt.show()


#BarPlot(Studytime-Failures)
fig,(ax1, ax2) = plt.subplots(1, 2, figsize=(15, 4)) 	
ax1 = sns.countplot(data["studytime"],ax = ax1)
custom_ticks = ['<2 hours', '2 to 5 hours', '5 to 10 hours', ">10 hours"]  # Custom tick labels
tick_positions = [0, 1, 2, 3]
ax1.set_xticks(tick_positions,custom_ticks,rotation = 90)
ax1.set_xlabel("Studytime")
ax1.set_ylabel("Count")
ax1.set_title("Weekly Study Time")
ax2 = sns.countplot(data["failures"],ax = ax2)
custom_ticks = ['0', '1', '2', ">=3"]  # Custom tick labels
tick_positions = [0, 1, 2, 3]
ax2.set_xticks(tick_positions,custom_ticks)
ax2.set_xlabel("Failures")
ax2.set_ylabel("Count")
ax2.set_title("Number of Past Class Failures")
plt.show()







#BarPlot (Famrel – Health)
fig,(ax1, ax2) = plt.subplots(1, 2, figsize=(15, 4)) 
ax1 = sns.countplot(data["famrel"],ax = ax1)
custom_ticks = ['very bad', 'bad', 'moderate', "good","excellent"]  # Custom tick labels
tick_positions = [0, 1, 2, 3, 4]
ax1.set_xticks(tick_positions,custom_ticks,rotation = 90) # or we can write rotation = "vertical"
ax1.set_xlabel("Famrel")
ax1.set_ylabel("Count")
ax1.set_title("Quality of Family Relationships")
ax2 = sns.countplot(data["health"],ax = ax2)
custom_ticks = ['very bad', 'bad', 'moderate', "good","very good"]  # Custom tick labels
tick_positions = [0, 1, 2, 3, 4]
ax2.set_xticks(tick_positions,custom_ticks,rotation = 90)
ax2.set_xlabel("Health")
ax2.set_ylabel("Count")
ax2.set_title("Current Health Status ")
plt.show()


#BarPlot(Freetime-Goout)
fig,(ax1, ax2) = plt.subplots(1, 2, figsize=(15, 4)) 
ax1 = sns.countplot(data["freetime"],ax = ax1)
custom_ticks = ['very low', 'low', 'moderate', "high","very high"]  # Custom tick labels
tick_positions = [0, 1, 2, 3, 4]
ax1.set_xticks(tick_positions,custom_ticks,rotation = 90) # or we can write rotation = "vertical"
ax1.set_xlabel("Freetime")
ax1.set_ylabel("Count")
ax1.set_title("Free Time After School")
ax2 = sns.countplot(data["goout"],ax = ax2)
custom_ticks = ['very low', 'low', 'moderate', "high","very high"]  # Custom tick labels
tick_positions = [0, 1, 2, 3, 4]
ax2.set_xticks(tick_positions,custom_ticks,rotation = 90)
ax2.set_xlabel("GoOut")
ax2.set_ylabel("Count")
ax2.set_title("Going Out With Friends")
plt.show()

#Horizontal BarPlot (Dalc-Walc)
fig,(ax1, ax2) = plt.subplots(1, 2, figsize=(15, 4)) 
ax1 = sns.countplot(y = data["Dalc"],ax = ax1, color ="#C0D6C2")
custom_ticks = ['very low', 'low', 'moderate', "high","very high"]  # Custom tick labels
tick_positions = [0, 1, 2, 3, 4]
ax1.set_yticks(tick_positions,custom_ticks) # or we can write rotation = "vertical"
ax1.set_ylabel("Dalc")
ax1.set_xlabel("Count")
ax1.set_title("Workday Alcohol Consumption")
ax2 = sns.countplot( y = data["Walc"],ax = ax2, color ="#C0D6C2")
custom_ticks = ['very low', 'low', 'moderate', "high","very high"]  # Custom tick labels
tick_positions = [0, 1, 2, 3, 4]
ax2.set_yticks(tick_positions,custom_ticks)
ax2.set_ylabel("Walc")
ax2.set_xlabel("Count")
ax2.set_title("Weekend Alcohol Consumption")
ax2.set_xlim(0,250)
plt.show()
#BoxPlot (Age)
# Create a boxplot
plt.figure(figsize=(15, 5))
sns.boxplot(data["age"], x='age')
plt.xlabel("Age")
plt.title("Boxplot of Age of Students")
plt.show()


#Histogramme (Absences)
plt.hist(data["absences"] ,bins = 9)
plt.xlim(0,90)
plt.xlabel("Day")
plt.ylabel("Count")
plt.title("Absences")

#Histogrammes (G1-G2-G3)
sns.distplot(data["G1"])
plt.title("Distribution de Note 1")
plt.show()

sns.distplot(data["G2"])
plt.title("Distribution de Note 2")
plt.show()

sns.distplot(data["G3"])
plt.title("Distribution de Note 3")
plt.show()
#Violin Plot (G1-G2-G3)
sns.violinplot(data =data[["G1","G2","G3"]])
plt.title("Distribution of Notes")
plt.ylabel("Count")
plt.show()


#Groupement des notes
df = pd.DataFrame(data)
# Define the conditions and corresponding labels
conditions = [
    (df['G3'] >= 15) & (df['G3'] <= 20),
    (df['G3'] >= 10) & (df['G3'] <= 14),
    df['G3'] < 10
]
labels = ['A', 'B', 'C']
# Use numpy.select to create the 'Notes' column
df['Notes'] = np.select(conditions, labels, default=None)
# Display the updated DataFrame
print(df)

#PiePlot(Notes)
nb_A = data[data["Notes"] == "A"].count()[1]
nb_B = data[data["Notes"] == "B"].count()[1]
nb_C = data[data["Notes"] == "C"].count()[1]
notes = [nb_A,nb_B,nb_C]
fig, ax = plt.subplots()
ax.pie(notes, labels=["A","B","C"], autopct='%1.1f%%',explode = [0.2,0,0])

# Draw a circle at the center to create the donut chart
circle = plt.Circle((0, 0), 0.70, fc='white')  # Adjust the second parameter for the size of the hole
fig.gca().add_artist(circle)
# Add a title
plt.title('Grades of Students')
# Equal aspect ratio ensures that pie is drawn as a circle.
ax.axis('equal')
# Display the chart
plt.show()

#Grouped BarPlot(Sex-Notes)
p = sns.countplot(data = data , x = data["sex"],hue = data["Notes"],hue_order = ["A","B","C"])
plt.title("Groupement des notes selon le sexe")
plt.show()

#Grouped BarPlot(Edu-Notes)
fig , (p1,p2) = plt.subplots(1,2,figsize = (15,5))
p1 = sns.countplot(data = data , x = "Medu",hue = "Notes",ax = p1,hue_order = ["A","B","C"])
custom_ticks = ['None', 'primary education', '5th to 9th grade', "secondary education","higher education"]  # Custom tick labels
tick_positions = [0, 1, 2, 3, 4]
p1.set_xticks(tick_positions,custom_ticks,rotation = 90)
p1.set_xlabel("Job")
p1.set_ylabel("Count")
p1.set_title("Mother Education")

p2 =sns.countplot(data = data , x = "Fedu",hue = "Notes",ax = p2,hue_order = ["A","B","C"])
custom_ticks = ['None', 'primary education', '5th to 9th grade', "secondary education","higher education"]  # Custom tick labels
tick_positions = [0, 1, 2, 3, 4]
p2.set_xticks(tick_positions,custom_ticks,rotation = 90)
p2.set_xlabel("Job")
p2.set_ylabel("Count")
p2.set_title("Father Education")
plt.show()


#Stacked BarPlot (Edu-Notes)
fig , (p1,p2) = plt.subplots(1,2,figsize = (15,5))
# Créez un DataFrame à partir de vos données
# Comptez le nombre d'étudiants par éducation maternelle et par note
count_data1 = df.groupby(['Medu', 'Notes']).size().unstack()
# Créez un graphique à barres empilées
count_data1.plot(kind='bar', stacked=True,ax = p1)
custom_ticks = ['None', 'primary education', '5th to 9th grade', "secondary education","higher education"]
tick_positions = [0, 1, 2, 3, 4]
p1.set_xticks(tick_positions,custom_ticks,rotation = 90)
p1.set_xlabel("Edu")
p1.set_ylabel("Count")
p1.set_title("Mother Education")
count_data2 = df.groupby(['Fedu', 'Notes']).size().unstack()
count_data2.plot(kind = "bar",stacked = True,ax = p2)
custom_ticks = ['None', 'primary education', '5th to 9th grade', "secondary education","higher education"]  # Custom tick labels
tick_positions = [0, 1, 2, 3, 4]
p2.set_xticks(tick_positions,custom_ticks,rotation = 90)
p2.set_xlabel("Edu")
p2.set_ylabel("Count")
p2.set_title("Father Education")
plt.show()

#Heatmap(ord-ord)
fig = plt.figure(figsize = (15,10))
sns.heatmap(df[ordin].corr(method = "spearman"), annot = True)
plt.title("Ordinal - Ordinal", size = 20)

#Grouped Boxplots (Mjob-Fjob)
fig,(ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5)) 
ax1 = sns.boxplot(x=data["Mjob"],y = data["G3"],ax = ax1)
ax1.set_xlabel("Job")
ax1.set_title("Mother Job")
ax2 = sns.boxplot(data["Fjob"],y = data["G3"],ax = ax2,order = ["at_home","health","other","services","teacher"])
ax2.set_xlabel("Job")
ax2.set_title("Father Job")
plt.show()

#Grouped Boxplot (Reason)
sns.boxplot(x = data["reason"], y = data["G3"])
plt.title("Reason to Choose This School")
plt.show()


#SwarmPlot (Studytime - Failures)
fig,(ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5)) 
ax1 = sns.swarmplot(data["studytime"],y= data["G3"],ax = ax1,color = "0.25")
custom_ticks = ['<2 hours', '2 to 5 hours', '5 to 10 hours', ">10 hours"]  # Custom tick labels
tick_positions = [0, 1, 2, 3]
ax1.set_xticks(tick_positions,custom_ticks,rotation = 90)
ax1.set_xlabel("Studytime")
ax1.set_title("Weekly Study Time")
ax2 = sns.swarmplot(x = data["failures"],y = data["G3"],ax = ax2,color = "0.25")
custom_ticks = ['0', '1', '2', ">=3"]  # Custom tick labels
tick_positions = [0, 1, 2, 3]
ax2.set_xticks(tick_positions,custom_ticks)
ax2.set_xlabel("Failures")
ax2.set_title("Number of Past Class Failures")
mean1 = data.groupby("studytime")["G3"].mean().values
mean2 = data.groupby("failures")["G3"].mean().values
ax1.scatter(tick_positions, mean1,  color='red', s=70, label='Mean')
ax2.scatter(tick_positions, mean2,  color='red', s=70, label='Mean')
plt.legend()
plt.show()

#Grouped Boxplot(Dalc-Walc)
fig,(ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5)) 
ax1 = sns.boxplot(x = data["Dalc"],y = data["G3"],ax = ax1)
custom_ticks = ['very low', 'low', 'moderate', "high","very high"]  # Custom tick labels
tick_positions = [0, 1, 2, 3, 4]
ax1.set_xticks(tick_positions,custom_ticks) # or we can write rotation = "vertical"
ax1.set_xlabel("Dalc")
ax1.set_title("Workday Alcohol Consumption")
ax2 = sns.boxplot( x = data["Walc"],y = data["G3"],ax = ax2)
custom_ticks = ['very low', 'low', 'moderate', "high","very high"]  # Custom tick labels
tick_positions = [0, 1, 2, 3, 4]
ax2.set_xticks(tick_positions,custom_ticks)
ax2.set_xlabel("Walc")
ax2.set_title("Weekend Alcohol Consumption")
plt.show()

#Transform binary attributs
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for feature in binary:
    data[feature] = le.fit_transform(data[feature])
    print(feature, le.classes_)

#Heatmap(Binary -Num)
fig = plt.subplots(figsize = (20,10))
sns.heatmap( data[num+binary].corr(),annot = True,cmap = "YlGnBu")

#BarPlot(Higher)
sns.boxplot(x = data["higher"] , y = data["G3"])
plt.show()





#ScatterPlot(G1-G2)
plt.subplots(1, 2, figsize=(15, 5)) 
plt.subplot(1,2,1)
plt.scatter(data["G1"],data["G3"])
plt.title("Distribution de Note 3"
plt.subplot(1,2,2)
plt.scatter(data["G2"],data["G3"])
plt.show()

#3D Scatter
# Create a 3D scatter plot
fig = plt.figure(figsize = (10,8))
ax = fig.add_subplot(111 ,projection='3d')
ax.scatter(data["G1"], data["G2"], data["G3"], c="blue", marker='o')
ax.set_xlabel('G1')
ax.set_ylabel('G2')
ax.set_zlabel('G3')
plt.title('3D Scatter Plot Example')
plt.show()

#Afficher les donnees ayant G3 = 0
data["moyenne"] = (data["G1"]+ data["G2"])/2
data[data["G3"] == 0]
data[data["G3"] == 0].iloc[25:38]"""
