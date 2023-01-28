import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import cluster,datasets
#Cargando base de datos Iris
iris=datasets.load_iris()
datos=np.array(pd.DataFrame(iris['data']))

#Proceso de agrupamiento iterativo
resultado=[]
for clases in range(2,10,1):
    grupos =cluster.KMeans(n_clusters=clases)#agrupamiento de datos de acuerdo al número de clases
    grupos.fit(datos)#evalua en todos los datos
    salida=grupos.predict(datos)#predicción de pertenencia de clase
    sigma=[], delta=[]
    #Estimación de sigma y delta por dato predicho
    for j in range(len(salida)):
        ck=[],cl=[]
        for i in range(len(salida)):
            if i!=j:
                diff=datos[j,:]-datos[i,:]
                de=np.linalg.norm(diff)**2
                if salida[i]==salida[j]:
                    ck.append(de)
                else:
                    cl.append(de)
        delta.append(max(ck))
        sigma.append(min(cl))
    D=min(sigma)/max(delta)
    resultado.append(D)
clase_op=resultado.index(max(resultado))+2
print('La clase óptima es '+str(clase_op))
x=range(2,10)
plt.figure(0)
plt.plot(x,resultado,'bx-')
