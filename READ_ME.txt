Pour lancer le programme:

python main.py (-r | -d | -s | -i | -o) (valeur)

paramètres :
 Les paramètres suivants prènent en entrée des entiers.

	-r	> "RECURSION", niveau de recursion pour DWT	 optionnel	valeur par défaut : 1
	-s	> "STEP", marche de quantification		 optionnel	valeur par défaut : 1
	-d	> "DEAD-ZONE", zone de zero pour quantification	 optionnel	valeur par défaut : 0

 Ces deux derniers paramètres prènent des chaines de caractère.
 Dans les deux cas, prenez soin d'inclure l'extension du fichier.

	-i	> "IN", image en entrée 			 optionnel	valeur par défaut : "RGB.jpg"
	-o	> "OUT", image de sortie			 optionnel	valeur par défaut : None