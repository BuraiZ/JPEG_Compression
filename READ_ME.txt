Pour lancer le programme:

python main.py (-r | -d | -s | -i | -o) (valeur)

param�tres :
 Les param�tres suivants pr�nent en entr�e des entiers.

	-r	> "RECURSION", niveau de recursion pour DWT	 optionnel	valeur par d�faut : 1
	-s	> "STEP", marche de quantification		 optionnel	valeur par d�faut : 1
	-d	> "DEAD-ZONE", zone de zero pour quantification	 optionnel	valeur par d�faut : 0

 Ces deux derniers param�tres pr�nent des chaines de caract�re.
 Dans les deux cas, prenez soin d'inclure l'extension du fichier.

	-i	> "IN", image en entr�e 			 optionnel	valeur par d�faut : "RGB.jpg"
	-o	> "OUT", image de sortie			 optionnel	valeur par d�faut : None