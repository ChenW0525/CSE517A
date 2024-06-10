from genTrainFeatures import genTrainFeatures
from naivebayesCL import naivebayesCL
from whoareyou import whoareyou

xTr,yTr = genTrainFeatures()
w,b = naivebayesCL(xTr,yTr)
whoareyou(w,b)