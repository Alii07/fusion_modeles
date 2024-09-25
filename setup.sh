#!/bin/bash

# Met à jour pip à la dernière version
python -m pip install --upgrade pip

# Vérifie et installe les dépendances système depuis .binder/apt.txt si présent
if [ -f ".binder/apt.txt" ]; then
    echo "Installation des paquets apt depuis .binder/apt.txt"
    sudo apt-get update
    xargs sudo apt-get install -y < .binder/apt.txt
fi

# Installe les dépendances Python depuis .binder/requirements.txt si présent
if [ -f ".binder/requirements.txt" ]; then
    echo "Installation des dépendances Python depuis .binder/requirements.txt"
    pip install -r .binder/requirements.txt
else
    # Installe depuis requirements.txt s'il n'y a pas de fichier .binder/requirements.txt
    echo "Installation des dépendances Python depuis requirements.txt"
    pip install -r requirements.txt
fi
