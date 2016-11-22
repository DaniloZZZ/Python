# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 12:44:20 2016

@author: student
"""

import htmlPy
from temp import bck

app = htmlPy.AppGUI(title=u"HW app")
app.maximized = False
app.template_path = "."
app.bind(bck(app))
app.template = ("index.html",{})

if __name__=="__main__":
    app.start()