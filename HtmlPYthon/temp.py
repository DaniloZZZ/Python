# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import htmlPy
import matplotlib.pyplot as plt
n = np.arange(100)
y = np.sin(n/10.)
plt.plot(n,y)
class bck(htmlPy.Object):
    def __init__(self,app):
        super(bck,self).__init__()
        self.app =  app
    @htmlPy.Slot()
    def hw(self):
        self.app.html = u"Hello World" 