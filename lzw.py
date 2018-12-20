import numpy as np

def encode(Message=None):
    dictsymb =[[str(Message[0])]]
    dictbin = ["{:b}".format(0)]
    nbsymboles = 1
    for i in range(1,len(Message)):
        if [str(Message[i])] not in dictsymb:
            dictsymb += [[str(Message[i])]]
            dictbin += ["{:b}".format(nbsymboles)] 
            nbsymboles +=1
    
    lengthPerSymbol = int(np.ceil(np.log2(nbsymboles)))
    longueurOriginale = lengthPerSymbol*len(Message)

    for i in range(nbsymboles):
        dictbin[i] = "{:b}".format(i).zfill(int(np.ceil(np.log2(nbsymboles))))

    NbSymbolCode = format(nbsymboles, 'b').zfill(8)

    DictionnaireCode = ""
    binLength = int(np.ceil(np.log2(nbsymboles)))
    for i in range(0, nbsymboles):
        DictionnaireCode += format(int(dictsymb[i][0]), 'b').zfill(lengthPerSymbol)
        DictionnaireCode += dictbin[i]

    i=0
    MessageCode = []
    longueur = 0
    while i < len(Message):
        precsouschaine = [str(Message[i])] #sous-chaine qui sera codé
        souschaine = [str(Message[i])] #sous-chaine qui sera codé + 1 caractère (pour le dictionnaire)
        
        #Cherche la plus grande sous-chaine. On ajoute un caractère au fur et à mesure.
        while souschaine in dictsymb and i < len(Message):
            i += 1
            precsouschaine = souschaine[:]
            if i < len(Message):  #Si on a pas atteint la fin du message
                souschaine += [str(Message[i])]

        #Codage de la plus grande sous-chaine à l'aide du dictionnaire
        codebinaire = [dictbin[dictsymb.index(precsouschaine)]]
        MessageCode += codebinaire
        longueur += len(codebinaire[0]) 
        #Ajout de la sous-chaine codé + symbole suivant dans le dictionnaire.
        #S'il reste de la place...
        if i < len(Message) and len(dictsymb) < 2**12:
            dictsymb += [souschaine]
            dictbin += ["{:b}".format(nbsymboles)] 
            nbsymboles +=1
            
            #Ajout de 1 bit si requis
            if np.ceil(np.log2(nbsymboles)) > len(MessageCode[-1]):
                for j in range(nbsymboles):
                    dictbin[j] = "{:b}".format(j).zfill(int(np.ceil(np.log2(nbsymboles))))

    '''
    dictionnaire = np.transpose([dictsymb,dictbin])
    print(dictionnaire)
    '''

    print("Longueur = {0}".format(longueur))
    print("Longueur originale = {0}".format(longueurOriginale))

    return NbSymbolCode  + DictionnaireCode  + ''.join(MessageCode)

def decode(code=None):
    dictsymb = []
    dictbin = []

    nbsymboles = int(code[0:8], 2)
    subCodeLength = int(np.ceil(np.log2(nbsymboles)))       # number of bit to decode at a time
    symLength = subCodeLength
    if (subCodeLength == 0):
        subCodeLength = 1
    
    if (symLength == 0):
        symLength = 1

    for i in range(0, nbsymboles):
        sym = str(int(code[symLength + i*(symLength+subCodeLength) : symLength + i*(symLength+subCodeLength) + symLength], 2))
        binValue = code[symLength + (i+1)*symLength + i*subCodeLength : symLength + (i+1)*symLength + i*subCodeLength + subCodeLength]
        dictsymb += [[sym]]
        dictbin += [binValue]

    '''
    listsymb = []
    for i in range(0, len(dictsymb)):
        listsymb += dictsymb[i]
    dictionnaire = np.transpose([dictbin,listsymb])
    print(dictionnaire)
    '''

    i=0
    message = []            # the complete decoded message
    code = code[8 + nbsymboles*(symLength+subCodeLength):]
    while i < len(code):
        subCode = code[i:i+subCodeLength]      # current part to decode

        # decode the subCode
        subMessage = dictsymb[dictbin.index(subCode)]
        message += [subMessage]

        # correct the placeholder of the previous symbol
        if ('_' in dictsymb[len(dictsymb)-1]):
            #if (dictsymb[len(dictsymb)-1].count('_') == 2):
             #   dictsymb[len(dictsymb)-1] = dictsymb[len(dictsymb)-2] + "_"
            dictsymb[len(dictsymb)-1][-1] = subMessage[0]


        # move index to prepare for next part to decode
        i += subCodeLength

        # Add new symbol to the dictionary, if there is space
        # the 0 is a temporary placeholder until the next subMessage is decoded
        if len(dictsymb) < 2**12:
            dictsymb += [subMessage + ["_"]]
            dictbin += ["{:b}".format(len(dictbin))]
            # Update dictionary to have enough bit to encode each symbol
            if np.ceil(np.log2(len(dictsymb))) > subCodeLength:
                subCodeLength += 1
                for j in range(len(dictsymb)):
                    dictbin[j] = "{:b}".format(j).zfill(int(np.ceil(np.log2(len(dictsymb)))))

    '''
    listsymb = []
    for i in range(0, len(dictsymb)):
        listsymb += dictsymb[i]
    dictionnaire = np.transpose([dictbin,listsymb])
    print(dictionnaire)
    print(dictsymb)
    '''

    # reconvert string message to integer
    temp = []
    for i in range(0, len(message)):
        for j in range(0, len(message[i])):
            temp += [int(message[i][j])]
    message = temp

    return message
    
