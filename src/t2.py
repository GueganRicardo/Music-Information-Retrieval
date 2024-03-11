# -*- coding: utf-8 -*-

import librosa
import librosa.display
import sounddevice as sd  
import warnings
import numpy as np
import matplotlib.pyplot as plt
import os
import scipy as sc
#./MER_audio_taffc_dataset/musicas


n_musicas=900
path_to_directory = "./MER_audio_taffc_dataset/musicas"
path_to_queries = "./Queries"
path_to_metadados = "./MER_audio_taffc_dataset/panda_dataset_taffc_metadata.csv"
path_to_avaliacoesTop100 = "./AvaliacoesTop100.txt"
path_to_avaliacoesMetadados = "./AvaliacoesMetadados.txt"
nomesMusicas = list()

def normalizaTop100():
    if os.path.isfile('./Features/top100_features.csv'):
        # EX 2.1.1
        features = np.genfromtxt('./Features/top100_features.csv', delimiter=',', skip_header=1)
        features = features[:, 1:-1]#tirar o nome e o quadrante da song
        # EX 2.1.2
        features_normalizadas = normaliza_features(features)
        # EX 2.1.3
        np.savetxt("featuresNormalizadasTop100.csv", features_normalizadas, delimiter=",")#guarda as features normalizadas
        # --- Load 
    else:
        print("ficheiro não encontrado.")

#retorna as stats de uma musica por normalizar
def featuresMusica(pathMusica):
    y, fs = librosa.load(pathMusica,mono=True)
    numSong=0
    nFeature=0
    statsMusica = np.zeros(190)
    mfcc = librosa.feature.mfcc(y=y,n_mfcc=13)#perceber o tamanho da frame e como fazemos os safe
    for i in mfcc:
        stats(i,statsMusica,nFeature)
        nFeature+=1
    #add das features dos mfcc
    spec_centroid = librosa.feature.spectral_centroid(y=y).flatten()
    stats(spec_centroid,statsMusica,nFeature)
    nFeature+=1
    spec_bandwidth = librosa.feature.spectral_bandwidth(y=y).flatten()
    stats(spec_bandwidth,statsMusica,nFeature)
    nFeature+=1
    spec_contrast = librosa.feature.spectral_contrast(y=y,n_bands=6)
    for i in spec_contrast:
        stats(i,statsMusica,nFeature)
        nFeature+=1
    #dar add as 7 bandas do spectral contrast
    spec_flatness = librosa.feature.spectral_flatness(y=y).flatten()
    stats(spec_flatness,statsMusica,nFeature)
    nFeature+=1
    spec_rolloff = librosa.feature.spectral_rolloff(y=y).flatten()
    stats(spec_rolloff,statsMusica,nFeature)
    nFeature+=1
    f0 = librosa.core.yin(y=y,fmin=20,fmax=22050/2).flatten()
    stats(f0,statsMusica,nFeature)
    nFeature+=1
    rms = librosa.feature.rms(y=y).flatten() #utiliza os valores default do enunciado
    stats(rms,statsMusica,nFeature)
    nFeature+=1
    zero_cross_rate = librosa.feature.zero_crossing_rate(y=y).flatten()
    stats(zero_cross_rate,statsMusica,nFeature)
    nFeature+=1
    tempo = librosa.feature.tempo(y=y)
    statsMusica[189]=tempo
    return statsMusica


def criaFicheiroFeatures():
    statsTotal = np.zeros((n_musicas,190))
    features_total = list()
    warnings.filterwarnings("ignore")
    numSong = -1
    for fName in os.listdir(path_to_directory):
        nFeature=0
        numSong+=1
        print("a tratar de", fName, numSong)
        y, fs = librosa.load("./MER_audio_taffc_dataset/musicas/"+fName, mono=True)#y freq em cada posicao
        mfcc = librosa.feature.mfcc(y=y,n_mfcc=13)#perceber o tamanho da frame e como fazemos os safe
        for i in mfcc:
            stats(i,statsTotal[numSong],nFeature)
            nFeature+=1
        #add das features dos mfcc
        spec_centroid = librosa.feature.spectral_centroid(y=y).flatten()
        stats(spec_centroid,statsTotal[numSong],nFeature)
        nFeature+=1
        spec_bandwidth = librosa.feature.spectral_bandwidth(y=y).flatten()
        stats(spec_bandwidth,statsTotal[numSong],nFeature)
        nFeature+=1
        spec_contrast = librosa.feature.spectral_contrast(y=y,n_bands=6)
        for i in spec_contrast:
            stats(i,statsTotal[numSong],nFeature)
            nFeature+=1
        #dar add as 7 bandas do spectral contrast
        spec_flatness = librosa.feature.spectral_flatness(y=y).flatten()
        stats(spec_flatness,statsTotal[numSong],nFeature)
        nFeature+=1
        spec_rolloff = librosa.feature.spectral_rolloff(y=y).flatten()
        stats(spec_rolloff,statsTotal[numSong],nFeature)
        nFeature+=1
        f0 = librosa.core.yin(y=y,fmin=20,fmax=22050/2).flatten()
        stats(f0,statsTotal[numSong],nFeature)
        nFeature+=1
        rms = librosa.feature.rms(y=y).flatten() #utiliza os valores default do enunciado
        stats(rms,statsTotal[numSong],nFeature)
        nFeature+=1
        zero_cross_rate = librosa.feature.zero_crossing_rate(y=y).flatten()
        stats(zero_cross_rate,statsTotal[numSong],nFeature)
        nFeature+=1
        tempo = librosa.feature.tempo(y=y)
        statsTotal[numSong][189]=tempo
    np.savetxt("featuresNoNormalizadas.csv", statsTotal, delimiter=",")
    statsTotal = normaliza_features(statsTotal)
    np.savetxt("featuresNormalizadas.csv", statsTotal, delimiter=",")


def distanciaEuclidiana(v1, v2):
    return np.linalg.norm(v1 - v2)

def distanciaManhattan(v1, v2):
    return np.sum(np.abs(v1 - v2))


def distanciaCoseno(vetor1, vetor2):
    dot_prod = np.dot(vetor1, vetor2)
    norm_prod = np.linalg.norm(vetor1) * np.linalg.norm(vetor2)
    cos_sim = dot_prod / norm_prod
    dist = 1 - cos_sim
    return dist

def getNomesMusicas():
    i =0
    for fName in os.listdir(path_to_directory):
        nomesMusicas.append(fName)
        i = i+ 1

def normaliza_features(features):
    min_por_coluna = np.amin(features, axis=0)
    max_por_coluna = np.amax(features, axis=0)
    for i in range(len(max_por_coluna)):
        if min_por_coluna[i] == max_por_coluna[i]:
            max_por_coluna[i] = min_por_coluna[i] + 1
    return (features - min_por_coluna) / (max_por_coluna - min_por_coluna)


def normalizaMusica(musica, features):
    min_por_coluna = np.amin(features, axis=0)
    max_por_coluna = np.amax(features, axis=0)
    for i in range(len(max_por_coluna)):
        if min_por_coluna[i] == max_por_coluna[i]:
            max_por_coluna[i] = min_por_coluna[i] + 1
    return (musica -  min_por_coluna) / (max_por_coluna - min_por_coluna)


##Transformar as feaures em stats
def stats(feature,statsMusica,i):
    statsMusica[i*7+0]=np.mean(feature)
    #print("media")
    statsMusica[i*7+1]=np.std(feature)
    #print("desvio")
    statsMusica[i*7+2]=sc.stats.skew(feature)
    #print("assimetria")
    statsMusica[i*7+3]=sc.stats.kurtosis(feature)
    #print("curtose")
    statsMusica[i*7+4]=np.median(feature)
    #print("mediana")
    statsMusica[i*7+5]=np.max(feature)
    #print("max")
    statsMusica[i*7+6]=np.min(feature)
    #print("min")


def criaMatrizSimilaridade(distancia, ficheiro_features):
    matriz_similaridade = np.zeros((n_musicas,n_musicas))
    if os.path.isfile(ficheiro_features):
        features = np.genfromtxt(ficheiro_features, delimiter=',')
        if distancia == "euc":
            for i in range(n_musicas):
                for f in range(n_musicas):
                    matriz_similaridade[i][f] = distanciaEuclidiana(features[i], features[f])
        elif distancia == "man":
            for i in range(n_musicas):
                for f in range(n_musicas):
                    matriz_similaridade[i][f] = distanciaManhattan(features[i], features[f])
        else:
            for i in range(n_musicas):
                for f in range(n_musicas):
                    matriz_similaridade[i][f] = distanciaCoseno(features[i], features[f])
        np.savetxt(distancia + ficheiro_features, matriz_similaridade, delimiter=",")
        

def rankingSimilaridadeTop100(nomeMusica):
    matchEuclidiana = list()
    matchManhattan = list()
    matchCoseno = list()
    indiceMusica = nomesMusicas.index(nomeMusica)
    #dist Euclidiana
    matrizSimilaridade = np.genfromtxt('./eucfeaturesNormalizadasTop100.csv', delimiter=',', skip_header=0)
    linhaMusica = matrizSimilaridade[indiceMusica][:]
    indicesTop20 = np.argsort(linhaMusica)[:21]
    for i in range(21):
        matchEuclidiana.append(nomesMusicas[indicesTop20[i]])
    print("Euclidean distance matches: ", matchEuclidiana)
    #dist Manhattan
    matrizSimilaridade = np.genfromtxt('./manfeaturesNormalizadasTop100.csv', delimiter=',', skip_header=0)
    linhaMusica = matrizSimilaridade[indiceMusica][:]
    indicesTop20 = np.argsort(linhaMusica)[:21]
    for i in range(21):
        matchManhattan.append(nomesMusicas[indicesTop20[i]])
    print("Manhattan distance matches: ", matchManhattan)
    #dist Cosenos
    matrizSimilaridade = np.genfromtxt('./cosfeaturesNormalizadasTop100.csv', delimiter=',', skip_header=0)
    linhaMusica = matrizSimilaridade[indiceMusica][:]
    indicesTop20 = np.argsort(linhaMusica)[:21]
    for i in range(21):
        matchCoseno.append(nomesMusicas[indicesTop20[i]])
    print("Cosine similarity matches: ", matchCoseno)

    return [matchEuclidiana, matchManhattan, matchCoseno]

def rankingSimilaridade(dadosMusica):
    featuresAntes = np.genfromtxt('./featuresNoNormalizadas.csv', delimiter=',', skip_header=0)
    featuresNossas = np.genfromtxt('./featuresNormalizadas.csv', delimiter=',', skip_header=0)
    musica = normalizaMusica(dadosMusica, featuresAntes)
    dist_euclidiana = np.zeros(n_musicas)
    dist_manhattan = np.zeros(n_musicas)
    dist_coseno = np.zeros(n_musicas)
    for i in range(n_musicas):
        dist_euclidiana[i] = distanciaEuclidiana(musica, featuresNossas[i])
        dist_manhattan[i] = distanciaManhattan(musica, featuresNossas[i])
        dist_coseno[i] = distanciaCoseno(musica, featuresNossas[i])

    indicesEuclidiana = np.argsort(dist_euclidiana)[:21]
    indicesManhattan = np.argsort(dist_manhattan)[:21]
    indicesCoseno = np.argsort(dist_coseno)[:21]

    matchEuclidiana = list()
    matchManhattan = list()
    matchCoseno = list()
    
    for i in range(21):
        matchEuclidiana.append(nomesMusicas[indicesEuclidiana[i]])
        matchManhattan.append(nomesMusicas[indicesManhattan[i]])
        matchCoseno.append(nomesMusicas[indicesCoseno[i]])

    print("Euclidean distance matches: ", matchEuclidiana)
    print("Manhattan distance matches: ", matchManhattan)
    print("Cosine similarity matches: ", matchCoseno)

    return [matchEuclidiana, matchManhattan, matchCoseno]


def remove_quotes(s):
    return s.replace('"', '')


def getMetadados():
    metadados = np.genfromtxt(path_to_metadados, delimiter=',', dtype='U500', skip_header=1, replace_space=remove_quotes)
    metadadosUteis = metadados[:, [1, 3, 9, 11]]
    col2 = np.char.split(metadadosUteis[:, 2], sep="; ")
    col3 = np.char.split(metadadosUteis[:, 3], sep="; ")
    metadadosUteis = np.hstack((metadadosUteis[:, :2], col2.reshape(-1, 1), col3.reshape(-1, 1), metadadosUteis[:, 4:]))
    for i in range(n_musicas):
        metadadosUteis[i][0]=np.char.replace(metadadosUteis[i][0], '"','')
        metadadosUteis[i][1]=np.char.replace(metadadosUteis[i][1], '"','')
        metadadosUteis[i][2]=np.char.replace(metadadosUteis[i][2], '"','')
        metadadosUteis[i][3]=np.char.replace(metadadosUteis[i][3], '"','')
    return metadadosUteis


def rankingMetadados(nomeMusica):
    pontuacoes = np.genfromtxt('./fichPontosMetadados.csv', delimiter=',', skip_header=0)
    indiceMusica = nomesMusicas.index(nomeMusica)
    posicoesRankMetadados = np.argsort(pontuacoes[indiceMusica])[::-1]
    rankingMetadados = list()
    for i in range(21):
        rankingMetadados.append(nomesMusicas[posicoesRankMetadados[i]])
    print("Resultado Ranking Metadados")
    print(rankingMetadados)
    return rankingMetadados

def prepFichMetadados(metadadosUteis):
    pontuacoes = np.zeros((n_musicas,n_musicas))
    numMetadadosUteis=len(metadadosUteis[0])#4
    for h in range (n_musicas):
        print(h)
        for i in range(n_musicas):
            for j in range(numMetadadosUteis):
                if j in [1,0]:
                    if(metadadosUteis[h][j]==metadadosUteis[i][j]):
                        pontuacoes[h][i]+=1
                else:
                    for k in range(len(metadadosUteis[i][j])):
                        if np.isin(metadadosUteis[i][j][k],metadadosUteis[h][j]):
                            pontuacoes[h][i]+=1
    print(pontuacoes)
    np.savetxt("fichPontosMetadados", pontuacoes, delimiter=",")





def calculaPrecisionObjetiva(resultadosSimilaridade1, resultadosSimilaridade2, resultadosMetadados):
    nRelevantes = np.zeros(6)
    for i in range(1,21):
        if resultadosSimilaridade1[0][i] in resultadosMetadados:
            nRelevantes[0]+=1
        if resultadosSimilaridade1[1][i] in resultadosMetadados:
            nRelevantes[1]+=1
        if resultadosSimilaridade1[2][i] in resultadosMetadados:
            nRelevantes[2]+=1
        if resultadosSimilaridade2[0][i] in resultadosMetadados:
            nRelevantes[3]+=1
        if resultadosSimilaridade2[1][i] in resultadosMetadados:
            nRelevantes[4]+=1
        if resultadosSimilaridade2[2][i] in resultadosMetadados:
            nRelevantes[5]+=1
    print(nRelevantes/20)


def calculaPrecisionSubjetiva(pathDasAvaliacoes):
    avaliacoesTop100 =  np.genfromtxt(pathDasAvaliacoes, delimiter=',')
    medias = np.zeros(5)
    desviosPadroes = np.zeros(5)
    precision = np.zeros(4)

    for i in range(4):
        medias[i] = np.mean(avaliacoesTop100[i::4])
        desviosPadroes[i] = np.std(avaliacoesTop100[i::4])
        mediasColunas = np.mean(avaliacoesTop100[i::4], axis=0)
        precision[i] = np.sum(mediasColunas > 2.5)
    medias[4]=np.mean(avaliacoesTop100)
    desviosPadroes[4] = np.std(avaliacoesTop100)
    print("Medias:",medias)
    print("desviosPadroes:",desviosPadroes)
    print(precision/20)

def main():
    #normalizaTop100()
    #criaFicheiroFeatures()
    distancias = ["euc", "man", "cos"]
    diretorias = ["featuresNormalizadas.csv", "featuresNormalizadasTop100.csv"]
    #matrizes de similaridade
    """
    for dist in distancias:
        for fName in diretorias:
            criaMatrizSimilaridade(dist, fName)
    queries = os.listdir("./Queries")
    """
    metadadosUteis = getMetadados()
    getNomesMusicas()
    #prepFichMetadados(metadadosUteis)
    for fName in os.listdir(path_to_queries):
        print("Nome da Musica:", fName)
        path = "./Queries/" + fName
        dadosMusica = featuresMusica(path)  
        print("---------------------------------------------------------------------")
        print("Com as nossas Features:")
        resultadosSimilaridade = rankingSimilaridade(dadosMusica)#usa as nossas features
        print("---------------------------------------------------------------------")
        print("Com as featires do top 100:")
        resultadosSimilaridadeTop100 = rankingSimilaridadeTop100(fName)
        print("---------------------------------------------------------------------")
        resultadosMetadados = rankingMetadados(fName)
        print("---------------------------------------------------------------------")
        print("Precision:")
        calculaPrecisionObjetiva(resultadosSimilaridade, resultadosSimilaridadeTop100, resultadosMetadados)
    print("Resultados avaliacoes Top100")
    calculaPrecisionSubjetiva(path_to_avaliacoesTop100)
    print("Resultados avaliacoes Metadados")
    calculaPrecisionSubjetiva(path_to_avaliacoesMetadados)


if __name__ == "__main__":
    main()
