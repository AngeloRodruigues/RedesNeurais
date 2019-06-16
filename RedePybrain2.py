import numpy as np
from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules import SoftmaxLayer
from pybrain.structure.modules import sigmoidlayer

'''rede = buildNetwork(2, 3, 1, outclass=SoftmaxLayer)
print(rede['in'])
print(rede['hidden0'])
print(rede['out'])
print(rede['bias'])'''

rede = buildNetwork(3, 10, 1)
base = SupervisedDataSet(3, 1)
# ENTRADA
# coluna 1: Vinha em aula ? 1(sim) 2(não)
# coluna 2: Estudo ? 1(sim) 2(não)
# coluna 3: Gosta da materia ? 1(sim) 2(não)
#
# SAIDA
# coluna 1: Vai bem na prova! 1(sim) 2(não)
# coluna 2: Vai tira média! 1(sim) 2(não)
# coluna 3: Vai sair mal! 1(sim) 2(não)
base.addSample((1, 0, 0), (1, ))
base.addSample((0, 1, 0), (1, ))
base.addSample((0, 0, 1), (1, ))
base.addSample((1, 1, 1), (3, ))
base.addSample((2, 0, 0), (2, ))
base.addSample((0, 2, 0), (2, ))
base.addSample((0, 0, 2), (2, ))
base.addSample((2, 2, 2), (6, ))
base.addSample((0, 0, 0), (0, ))
base.addSample((3, 0, 0), (3, ))
base.addSample((0, 3, 0), (3, ))
base.addSample((0, 0, 3), (3, ))
base.addSample((3, 3, 3), (9, ))
base.addSample((4, 0, 0), (4, ))
base.addSample((0, 4, 0), (4, ))
base.addSample((0, 0, 4), (4, ))
base.addSample((4, 4, 4), (12, ))
base.addSample((5, 0, 0), (5, ))
base.addSample((0, 5, 0), (5, ))
base.addSample((0, 0, 5), (5, ))
base.addSample((5, 5, 5), (15, ))
base.addSample((6, 0, 0), (6, ))
base.addSample((0, 6, 0), (6, ))
base.addSample((0, 0, 6), (6, ))
base.addSample((6, 6, 6), (18, ))
base.addSample((7, 0, 0), (7, ))
base.addSample((0, 7, 0), (7, ))
base.addSample((0, 0, 7), (7, ))
base.addSample((7, 7, 7), (21, ))
base.addSample((8, 0, 0), (8, ))
base.addSample((0, 8, 0), (8, ))
base.addSample((0, 0, 8), (8, ))
base.addSample((8, 8, 8), (24, ))
base.addSample((9, 0, 0), (9, ))
base.addSample((0, 9, 0), (9, ))
base.addSample((0, 0, 9), (9, ))
base.addSample((9, 9, 9), (27, ))
base.addSample((10, 0, 0), (10, ))
base.addSample((0, 10, 0), (10, ))
base.addSample((0, 0, 10), (10, ))
base.addSample((10, 10, 10), (30, ))
'''
print(base['input'])
print(base['target'])
'''

treinamento = BackpropTrainer(rede, dataset=base, learningrate=0.003, momentum=0.5)
for i in range(1, 5000):
    erro = treinamento.train()
    if i % i == 0:
        print(f'Erro{i}: {erro}')
print(f'Rede treinada!\nUltima margem de erro foi -> {erro}')
print()
while True:
    p1 = str(input('digite um valor: '))
    p2 = str(input('digite um valor: '))
    p3 = str(input('digite um valor: '))

    resu = rede.activate([p1, p2, p3])
    print(np.round(resu))
    print(resu)

    r = input('Deseja repetir?[s/n] ')
    if r == 'n':
        break
