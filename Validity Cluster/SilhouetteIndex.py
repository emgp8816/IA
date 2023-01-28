import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import cluster,datasets

#Carga de Base de datos
iris=datasets.load_iris()
datos=np.array(pd.DataFrame(iris['data']))


resultado=[]
for clases in range(2,10,1):
    grupos =cluster.KMeans(n_clusters=clases)
    grupos.fit(datos)#evalua en todos los datos
    salida=grupos.predict(datos)
    sil=0
    sumde=[]
    for j in range(len(salida)):
        b=np.zeros(clases)
        contb=np.zeros(clases)
        for i in range(len(salida)):
            diff=datos[j,:]-datos[i,:]
            de=np.linalg.norm(diff)**2
            if i!=j:
                b[salida[i]]=b[salida[i]]+de
                contb[salida[i]]=contb[salida[i]]+1
        a=b[salida[j]]/contb[salida[j]]
        b[salida[j]]=0
        for numclass in range(clases):
            b[numclass]=b[numclass]/contb[numclass]
        b1=b[b!=0]
        b=min(b1)
        sil=sil+((b-a)/max(a,b))
    resultado.append(1/len(salida)*sil)
clase_op=resultado.index(max(resultado))+2
print('La clase óptima es '+str(clase_op))
                       
x=range(2,10)
plt.figure(0)
plt.plot(x,resultado,'bx-')
plt.show()
